"""Synthetic feature/selection/callback checks; no model fitting."""
import contextlib
import io
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/upi"))
from challenge import stopping_metric, warmup_stopping, reliability_weights, build_peer_features
from fit_submit import train_member
from make_submission import select_member_policy, predict_labels, prepare_rule_context
from test_podium_predictions import context, zero_offsets


class ChallengeTests(unittest.TestCase):
    def test_member_selection_full_mock_validation_refit_and_submission(self):
        import test_recall_policy
        self.member_selection = True
        test_recall_policy.RecallTests.test_mock_training_refit_and_runner_report_use_saved_blend(self)

    def test_balanced_loss_does_not_change_when_legit_is_repeated(self):
        metric = stopping_metric("balanced_logloss", [0, 4, 11])
        p = np.array([[.9, .05, .05], [.2, .6, .2], [.3, .3, .4]])
        expected = metric([0, 1, 2], p)[1]
        self.assertAlmostEqual(metric([0] * 20 + [1, 2], np.vstack([np.repeat(p[:1], 20, axis=0), p[1:]]))[1], expected)
        self.assertAlmostEqual(stopping_metric("macro_f1", [0, 4, 11])([0, 1, 2], p)[1], 3 / 12)

    def test_real_lightgbm_callback_warmup_without_training(self):
        import lightgbm as lgb
        callback = warmup_stopping(lgb.early_stopping(2, first_metric_only=True, verbose=False), 3)
        scores = [.99, .9, .3, .4, .35, .34]
        with self.assertRaises(lgb.callback.EarlyStopException) as stopped:
            for iteration, score in enumerate(scores):
                env = lgb.callback.CallbackEnv(model=None, params={}, iteration=iteration,
                    begin_iteration=0, end_iteration=10,
                    evaluation_result_list=[("valid_0", "macro_f1", score, True)])
                callback(env)
        self.assertEqual(stopped.exception.best_iteration, 3)  # round4, not round1

    def test_peer_means_exclude_self_and_isolate_split_and_missing_teacher(self):
        base = pd.DataFrame({"txn_id": list("abcdef"), "timestamp": ["2026-08-01 00:01"] * 6,
            "split": ["train"] * 4 + ["test"] * 2, "sender_id": ["S"] * 6,
            "receiver_id": ["R"] * 6, "receiver_type": ["P2P"] * 6, "device_id": ["D"] * 6})
        p = np.full((6, 12), .1 / 11)
        p[:, 0] = .9
        p[1] = np.eye(12)[4]
        p[3] = np.nan  # unavailable teacher query must have no peer evidence
        out = build_peer_features(base, p)
        self.assertEqual(out.peer_dev_count.tolist(), [2, 2, 2, 0, 1, 1])
        self.assertAlmostEqual(out.peer_dev_class_4.iloc[0], (1 + .1 / 11) / 2)
        changed = p.copy()
        changed[0] = np.eye(12)[11]
        changed[4:] = np.eye(12)[2]
        other = build_peer_features(base, changed)
        np.testing.assert_allclose(out.filter(like="peer_").iloc[0], other.filter(like="peer_").iloc[0], equal_nan=True)
        self.assertTrue(np.isnan(out.peer_dev_class_4.iloc[3]))
        reversed_out = build_peer_features(base.iloc[::-1].reset_index(drop=True), p[::-1])
        pd.testing.assert_frame_equal(out, reversed_out.iloc[::-1].reset_index(drop=True))

    def test_peer_entity_namespace_and_month_boundaries(self):
        base = pd.DataFrame({"txn_id": ["a", "b", "c"],
            "timestamp": ["2026-08-31 23:59", "2026-08-31 23:59", "2026-09-01 00:00"],
            "split": "train", "sender_id": "S", "device_id": "D",
            "receiver_id": "R", "receiver_type": ["P2P", "MERCHANT", "P2P"]})
        out = build_peer_features(base, np.tile(np.eye(12)[1], (3, 1)))
        self.assertEqual(out.peer_rcv_count.tolist(), [0, 0, 0])
        self.assertEqual(out.peer_snd_count.tolist(), [1, 1, 0])

    def test_peer_time_cells_normalize_arrow_timestamp_resolution(self):
        b = pd.DataFrame({"txn_id": ["a", "b", "c"],
            "timestamp": np.array(["2026-08-01T00:01", "2026-08-01T00:59", "2026-08-01T01:01"], dtype="datetime64[us]"),
            "split": "train", "sender_id": "S", "receiver_type": "P2P", "receiver_id": "R", "device_id": "D"})
        out = build_peer_features(b, np.tile(np.eye(12)[1], (3, 1)))
        self.assertEqual(out.peer_snd_count.tolist(), [1, 1, 0])

    def test_reliability_preserves_labels_and_missing_teacher(self):
        clean = pd.DataFrame({"y_raw": [1, 1, 1, 0], "y_clean": [1, 1, 1, 4],
            "reason": ["unchanged"] * 3 + ["unreported"], "direct_band": [0, 1, 0, 0],
            "teacher_fraud_probability": [.01, .01, np.nan, .95]})
        before = clean.copy(deep=True)
        np.testing.assert_allclose(reliability_weights(clean), [.35, 1, 1, .6])
        pd.testing.assert_frame_equal(before, clean)

    def test_training_wires_custom_metric_and_minimum_with_mock_estimator(self):
        calls = []
        class Model:
            best_iteration_ = 150
            def __init__(self, **kwargs):
                calls.append(kwargs)
            def fit(self, X, y, **kwargs):
                calls.append(kwargs)
        fake = types.SimpleNamespace(LGBMClassifier=Model, early_stopping=lambda *a, **k: lambda env: None,
                                     log_evaluation=lambda period: ("log_period", period),
                                     basic=types.SimpleNamespace(LightGBMError=RuntimeError))
        with patch.dict(sys.modules, {"lightgbm": fake}):
            _, rounds = train_member(pd.DataFrame({"x": [1, 2]}), np.array([0, 4]), np.ones(2),
                pd.DataFrame({"x": [3, 4]}), np.array([0, 4]), device="cpu",
                spec={"seed": 1, "num_leaves": 31, "min_child_samples": 20,
                      "colsample_bytree": .8, "stop_metric": "macro_f1", "minimum_rounds": 150,
                      "log_period": 100})
        self.assertEqual(calls[0]["metric"], "None")
        self.assertTrue(callable(calls[1]["eval_metric"]))
        self.assertEqual(calls[1]["callbacks"][1], ("log_period", 100))
        self.assertEqual(rounds, 150)

    def test_member_selection_accepts_helpful_individual_and_freezes_weights(self):
        y = np.tile(np.arange(12), 2)
        meta, f3, f5 = context(24)
        meta["timestamp"] = np.repeat(["2026-09-16", "2026-09-20"], 12)
        ctx = prepare_rule_context(meta, f3, f5)
        good = np.eye(12)[y]
        bad = np.eye(12)[(y + 1) % 12]
        groups = np.full(24, "other")
        with patch("make_submission.fit_offsets", return_value=zero_offsets()), contextlib.redirect_stdout(io.StringIO()):
            offsets, selection = select_member_policy({"A": bad, "B": good}, {"A": 1., "B": 0.},
                y, groups, ctx, config={"min_gain": .001})
        self.assertEqual(selection["member_weights"], {"A": 0., "B": 1.})
        np.testing.assert_array_equal(predict_labels(good, groups, offsets, ctx), y)
        ctx["dates"][-1] = np.datetime64("2026-09-24")
        with self.assertRaises(ValueError):
            select_member_policy({"A": bad, "B": good}, {"A": 1., "B": 0.}, y, groups, ctx, config={})


if __name__ == "__main__":
    unittest.main()
