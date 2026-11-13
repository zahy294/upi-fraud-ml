"""Tiny synthetic calibration tests; no real inputs or model training."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/upi"))
from fit_submit import (fraud_class_weights, _profile_run, calendar_dates, split_masks,
                        load_probability_artifact)
from common import LABELS
from make_submission import (fit_offsets, predict_labels, prepare_rule_context,
                             select_recall_policy, evaluate_predictions)
from test_podium_predictions import context


BASE = {"steps": [.4, .2, .1, .05], "max_passes": 1, "conditional": False}
RECALL = {"blend_weights": [0., .5, 1.], "min_gain": .001,
          "calibration": {"coarse_grid": [-1., 0., 1., 2., 3.],
                          "offset_bound": 3., "conditional": False}}


def fixture():
    y = np.tile(np.arange(12), 4)
    meta, f3, f5 = context(len(y))
    meta["timestamp"] = np.repeat(["2026-09-16", "2026-09-19", "2026-09-20", "2026-09-23"], 12)
    ctx = prepare_rule_context(meta, f3, f5)
    groups = np.full(len(y), "other")
    base = np.full((len(y), 12), .2 / 11)
    base[:, 0] = .8
    good = np.full_like(base, .1 / 11)
    good[np.arange(len(y)), y] = .9
    return y, groups, ctx, base, good


class RecallTests(unittest.TestCase):
    def test_coarse_search_crosses_flat_f1_region(self):
        y, groups, ctx, base, _ = fixture()
        # All rows actually mule; small local steps cannot bridge log(.8/.1).
        y[:] = 1
        base[:] = .1 / 10
        base[:, 0], base[:, 1] = .8, .1
        old = fit_offsets(base, y, groups, ctx, config=BASE)
        new = fit_offsets(base, y, groups, ctx, config={**BASE, **RECALL["calibration"]})
        self.assertTrue((predict_labels(base, groups, old, ctx) == 0).all())
        self.assertTrue((predict_labels(base, groups, new, ctx) == 1).all())

    def test_rare_class_weights_are_bounded_and_fit_only(self):
        weights, counts = fraud_class_weights(np.array([0] * 100 + [1] * 25 + [11]))
        self.assertEqual(weights[0], 1.)
        self.assertGreater(weights[11], weights[1])
        self.assertEqual(weights[11], 6.)
        self.assertEqual(weights[7], 1.)  # absent class is not invented
        self.assertEqual(counts.sum(), 126)
        unit, _ = fraud_class_weights(np.arange(12), power=0.)
        np.testing.assert_array_equal(unit, np.ones(12))
        for labels, kwargs in (([], {}), ([12], {}), ([0, 1], {"cap": np.nan})):
            with self.assertRaises(ValueError):
                fraud_class_weights(np.asarray(labels), **kwargs)

    def test_gate_accepts_useful_member_and_serializes(self):
        y, groups, ctx, base, good = fixture()
        offsets, selection = select_recall_policy(base, good, y, groups, ctx,
                                                 baseline_config=BASE, config=RECALL)
        self.assertGreater(selection["recall_weight"], 0.)
        alpha = selection["recall_weight"]
        mixed = (1 - alpha) * base + alpha * good
        restored = json.loads(json.dumps(offsets, allow_nan=False))
        pred = predict_labels(mixed, groups, restored, ctx)
        self.assertEqual(evaluate_predictions(y, pred)["macro_f1"], 1.)
        self.assertEqual(len(selection["trials"]), 4)
        self.assertEqual(selection["gate_days"], "Sep20-23")

    def test_gate_keeps_legacy_when_candidates_do_not_improve(self):
        y, groups, ctx, bad, good = fixture()
        offsets, selection = select_recall_policy(good, bad, y, groups, ctx,
                                                 baseline_config=BASE, config=RECALL)
        self.assertEqual(selection["selected"], "legacy")
        self.assertEqual(selection["recall_weight"], 0.)
        np.testing.assert_array_equal(predict_labels(good, groups, offsets, ctx), y)

    def test_missing_inner_partition_falls_back_and_assessment_is_rejected(self):
        y, groups, ctx, base, good = fixture()
        ctx["dates"] = np.repeat(np.datetime64("2026-09-16"), len(y))
        _, selection = select_recall_policy(base, good, y, groups, ctx,
                                            baseline_config=BASE, config=RECALL)
        self.assertEqual(selection["selected"], "legacy")
        self.assertEqual(selection["fallback_reason"], "missing_inner_partition")
        ctx["dates"][-1] = np.datetime64("2026-09-24")
        with self.assertRaises(ValueError):
            select_recall_policy(base, good, y, groups, ctx,
                                 baseline_config=BASE, config=RECALL)

    def test_direct_band_still_wins_under_large_offsets(self):
        y, groups, ctx, base, good = fixture()
        ctx["band"][:] = 11
        offsets = fit_offsets(base, y, groups, ctx, config={**BASE, **RECALL["calibration"]})
        self.assertTrue((predict_labels(base, groups, offsets, ctx) == 11).all())

    def test_invalid_grid_and_blend_fail(self):
        y, groups, ctx, base, good = fixture()
        for config in ({"coarse_grid": [4.]}, {"offset_bound": float("nan")}):
            with self.assertRaises(ValueError):
                fit_offsets(base, y, groups, ctx, config=config)
        for config in ({"blend_weights": [1.1]}, {"min_gain": -.1}):
            with self.assertRaises(ValueError):
                select_recall_policy(base, good, y, groups, ctx,
                                     baseline_config=BASE, config=config)

    def test_mock_training_refit_and_runner_report_use_saved_blend(self):
        # Real feature/artifact/decision plumbing with a fake estimator only.
        # Model probabilities are predetermined; no fitting or real data reads.
        days = ["2026-07-01", "2026-08-01", "2026-09-05", "2026-09-16",
                "2026-09-20", "2026-09-24", "2026-10-01"]
        labels = np.tile(np.r_[np.zeros(12, dtype=int), np.arange(1, 12)], len(days))
        n = len(labels)
        b = pd.DataFrame({"txn_id": [f"id{i}" for i in range(n)],
                          "timestamp": np.repeat(days, 23), "receiver_type": "P2P",
                          "amount": 1., "split": ["train"] * (n - 23) + ["test"] * 23,
                          "label": np.asarray(LABELS)[labels]})
        _, f3, f5 = context(n)
        f3["txn_id"], f5["txn_id"] = b.txn_id, b.txn_id
        packs = {"v1": b[["txn_id"]].assign(x=1.),
                 "v2": b[["txn_id"]].assign(S_flag=0., Ds_n=0.), "v3": f3,
                 "v4": b[["txn_id"]].assign(is_tiny_p2p_pay=0., bait_prev_dt_h=np.nan),
                 "v5": f5, "v6": b[["txn_id"]].assign(v6_x=1.),
                 "te": b[["txn_id"]].assign(v6_te_x=.1)}
        config = {"splits": {"fit": ["07-01", "09-01"], "early_stop": ["09-01", "09-16"],
                             "calibration": ["09-16", "09-24"], "assessment": ["09-24", "10-01"]},
                  "device": "cpu", "profiles": ["ensemble"], "legit_fraction": 1.,
                  "members": [{"name": name, "seed": seed, "num_leaves": 31,
                               "min_child_samples": 20, "colsample_bytree": .8, "weight": weight}
                              for name, seed, weight in [("A", 17, .5), ("B", 43, .3), ("C", 101, .2)]],
                  "calibration": BASE, "recall": {**RECALL, "enabled": True,
                      "member": {"name": "R", "seed": 211, "num_leaves": 47,
                                 "min_child_samples": 30, "colsample_bytree": .8}}}
        class Booster:
            def save_model(self, path):
                Path(path).write_text("mock only", encoding="utf-8")
        class Model:
            podium_classes_ = np.arange(12)
            podium_backend_ = "mock"
            booster_ = Booster()
            def __init__(self, recall):
                self.recall = recall
            def predict_proba(self, X):
                result = np.full((len(X), 12), .1 / 11 if self.recall else .2 / 11)
                if self.recall:
                    result[np.arange(len(X)), labels[X.index]] = .9
                else:
                    result[:, 0] = .8
                return result
        calls = []
        def mock_train(X, y, weights, *args, **kwargs):
            calls.append((kwargs["spec"]["name"], kwargs["fixed_rounds"], np.asarray(weights)))
            return Model(kwargs["spec"]["name"] == "R"), 7
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "runs/synthetic"
            config["_root"] = str(root)
            dates = calendar_dates(b, config)
            masks, _ = split_masks(b, dates, config)
            raw = root / "data/raw"
            raw.mkdir(parents=True)
            pd.DataFrame({"id": b.txn_id[masks["test"]].to_numpy()[::-1]}).to_csv(raw / "sample_submission.csv", index=False)
            with patch("fit_submit.train_member", side_effect=mock_train):
                for stage, mask in [("validation", masks["fit"]), ("refit", masks["train"])]:
                    cleaned = pd.DataFrame({"txn_id": b.txn_id[mask], "y_clean": labels[mask], "reason": "unchanged"})
                    _profile_run(b, packs, cleaned, dates, masks, config, run, "ensemble", stage, "mock-signature")
            val = run / "ensemble/validation"
            refit = run / "ensemble/refit"
            selection = json.loads((val / "recall_selection.json").read_text())
            self.assertGreater(selection["recall_weight"], 0.)
            self.assertEqual(selection, json.loads((refit / "recall_selection.json").read_text()))
            test_ids = b.txn_id[masks["test"]]
            baseline = load_probability_artifact(refit / "A_proba.npz", test_ids)
            recall = load_probability_artifact(refit / "R_proba.npz", test_ids)
            mixed = load_probability_artifact(refit / "test_proba.npz", test_ids)
            alpha = selection["recall_weight"]
            np.testing.assert_allclose(mixed, (1 - alpha) * baseline + alpha * recall, atol=1e-7)
            self.assertEqual([c[1] for c in calls], [None] * 4 + [7] * 4)
            self.assertTrue((calls[3][2] >= calls[0][2]).all())
            self.assertTrue((calls[3][2] > calls[0][2]).any())
            # Exercise report generation and strict submission reconstruction.
            from test_colab_runner import runner
            (run / "effective_config.json").write_text(json.dumps(config))
            (run / "manifest.json").write_text("{}")
            import fit_submit
            import make_submission
            with patch.object(runner, "_load_api", return_value=(fit_submit, make_submission)):
                report = runner.validate_artifacts(root, run)
            self.assertEqual(report["profiles"]["ensemble"]["recall_selection"], selection)
            self.assertIn("Assessment change versus legacy", (run / "RUN_REPORT.md").read_text())


if __name__ == "__main__":
    unittest.main()
