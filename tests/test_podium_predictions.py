"""Small synthetic verification; no real files, CV or model training."""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/upi"))
from common import LABELS
from fit_submit import (align_proba, sample_training, save_probability_artifact,
                        load_probability_artifact, rebuild_chain_features,
                        calendar_dates, split_masks, train_member, _frozen_features,
                        _profile_run, build_context_te, _bootstrap)
from make_submission import (apply_band_rules, prepare_rule_context, event_groups,
                             fit_offsets, predict_labels, evaluate_predictions,
                             write_submission, slice_rule_context)


def context(n=6):
    meta = pd.DataFrame({"txn_id": [f"t{i}" for i in range(n)],
                         "receiver_type": ["P2P"] * n,
                         "timestamp": pd.date_range("2026-09-16", periods=n, freq="D")})
    f3 = pd.DataFrame({"txn_id": meta.txn_id, "band": np.zeros(n)})
    for k in range(1, 12):
        f3[f"snd_band_{k}_3h"] = np.zeros(n)
        f3[f"rcv_band_{k}_24h"] = np.zeros(n)
    f5 = pd.DataFrame({"txn_id": meta.txn_id, "ch_band": np.zeros(n), "ch_len": np.ones(n)})
    return meta, f3, f5


def p(n, k=0):
    values = np.full((n, 12), .1 / 11)
    values[:, k] = .9
    return values


def zero_offsets():
    return {"global": [0.] * 12, "conditional": {}, "class_order": list(LABELS)}


class ProbabilityTests(unittest.TestCase):
    def test_missing_class_alignment(self):
        out = align_proba(np.array([[.25, .75]]), np.array([4, 0]))
        self.assertEqual(out.shape, (1, 12))
        self.assertEqual(out[0, 4], .25)
        self.assertEqual(out[0, 0], .75)
        self.assertEqual(out[0, 5], 0)
        for probs, classes in (([[np.nan]], [0]), ([[-1]], [0]), ([[0]], [0]), ([[1, 0]], [0, 0])):
            with self.assertRaises(ValueError):
                align_proba(np.array(probs), np.array(classes))

    def test_id_and_class_order_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.npz"
            save_probability_artifact(path, ["a", "b"], p(2))
            np.testing.assert_allclose(load_probability_artifact(path, ["a", "b"]), p(2), atol=1e-7)
            with self.assertRaises(ValueError):
                load_probability_artifact(path, ["b", "a"])
            np.savez(path, ids=["a", "b"], proba=p(2), classes=np.array(LABELS)[::-1], provenance="{}")
            with self.assertRaises(ValueError):
                load_probability_artifact(path, ["a", "b"])
            with self.assertRaises(ValueError):
                save_probability_artifact(path, ["a", None], p(2))

    def test_missing_teacher_rows_and_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "teacher.npz"
            probs = p(2)
            probs[0] = np.nan
            save_probability_artifact(path, ["a", "b"], probs, allow_missing=True, provenance={"fold": 1})
            self.assertTrue(np.isnan(load_probability_artifact(path, ["a", "b"], allow_missing=True)[0]).all())
            with self.assertRaises(ValueError):
                load_probability_artifact(path, ["a", "b"])
            with self.assertRaises(ValueError):
                load_probability_artifact(path, ["a", "b"], allow_missing=True, expected_provenance={"fold": 2})

    def test_sampling_is_id_stable_and_inverse_weighted(self):
        ids = np.array([f"id{i}" for i in range(40)])
        y = np.array([0] * 30 + [4] * 10)
        keep, weight = sample_training(ids, y)
        reverse_keep, reverse_weight = sample_training(ids[::-1], y[::-1])
        np.testing.assert_array_equal(keep, reverse_keep[::-1])
        self.assertTrue(keep[30:].all())
        np.testing.assert_array_equal(weight, np.where(y[keep] == 0, 4., 1.))
        np.testing.assert_array_equal(weight, reverse_weight[::-1])

    def test_lazy_training_argument_guard(self):
        with self.assertRaises(ValueError):
            train_member(None, None, None, object(), np.array([0]), spec={}, device="cpu", fixed_rounds=2)

    def test_training_api_with_mock_only(self):
        # No LightGBM import or learning: fake estimator records kwargs only.
        calls = []
        class FakeModel:
            def __init__(self, **kwargs):
                self.params = kwargs
                self.best_iteration_ = 0
            def fit(self, X, y, **kwargs):
                calls.append((self.params, np.asarray(y), kwargs))
        fake = types.SimpleNamespace(LGBMClassifier=FakeModel,
                                     basic=types.SimpleNamespace(LightGBMError=RuntimeError))
        spec = {"seed": 17, "num_leaves": 63, "min_child_samples": 20,
                "colsample_bytree": .5}
        with patch.dict(sys.modules, {"lightgbm": fake}):
            model, rounds = train_member(pd.DataFrame({"x": [1, 2]}), np.array([0, 4]),
                                          np.ones(2), None, None, spec=spec, device="cpu", fixed_rounds=12)
        self.assertEqual(rounds, 12)
        np.testing.assert_array_equal(model.podium_classes_, [0, 4])
        np.testing.assert_array_equal(calls[0][1], [0, 1])
        self.assertNotIn("eval_set", calls[0][2])
        self.assertEqual(calls[0][0]["objective"], "multiclass")


class RuleTests(unittest.TestCase):
    def test_precedence_and_original_prediction_eligibility(self):
        meta, f3, f5 = context()
        f3.loc[0, "band"] = 4
        f5["ch_band"] = [1, 1, 11, 3, 2, 0]
        f3.loc[1, "snd_band_4_3h"] = 1
        f3.loc[2, "rcv_band_7_24h"] = 1
        meta.loc[2, "receiver_type"] = "MERCHANT"
        out = apply_band_rules(np.array([0, 5, 8, 0, 1, 5]), meta, f3.iloc[::-1], f5.iloc[::-1])
        np.testing.assert_array_equal(out, [4, 1, 11, 0, 2, 5])

    def test_ambiguous_neighbours_do_not_force_type(self):
        meta, f3, f5 = context(2)
        f3.loc[0, ["snd_band_4_3h", "snd_band_5_3h"]] = 1
        f3.loc[1, "rcv_band_3_24h"] = 1
        np.testing.assert_array_equal(apply_band_rules(np.array([5, 1]), meta, f3, f5), [5, 3])

    def test_groups_priority(self):
        meta, f3, f5 = context()
        f3.loc[0, "band"] = 1
        f5.loc[:1, "ch_len"] = 2
        f2 = pd.DataFrame({"txn_id": meta.txn_id, "S_flag": [1, 1, 1, 0, 0, 0],
                           "Ds_n": [2, 2, 2, 2, 0, 0]})
        f4 = pd.DataFrame({"txn_id": meta.txn_id, "is_tiny_p2p_pay": [1] * 5 + [0],
                           "bait_prev_dt_h": [np.nan] * 6})
        np.testing.assert_array_equal(event_groups(meta, f3, f5, f2, f4),
                                      ["direct_band", "chain", "merchant_session", "drain", "bait", "other"])

    def test_all_12_metrics_and_empty_support(self):
        metrics = evaluate_predictions(np.array([0, 4]), np.array([0, 4]))
        self.assertAlmostEqual(metrics["macro_f1"], 2 / 12)
        self.assertEqual(len(metrics["per_class"]), 12)
        self.assertEqual(metrics["per_class"]["mule_account"]["f1"], 0)
        empty = evaluate_predictions(np.array([], dtype=int), np.array([], dtype=int))
        self.assertTrue(empty["empty"])
        self.assertEqual(empty["macro_f1"], 0)

    def test_offset_roundtrip_and_calibration_partition_guard(self):
        meta, f3, f5 = context()
        ctx = prepare_rule_context(meta, f3, f5)
        groups = np.array(["other"] * 6)
        probs = p(6, 4)
        y = np.array([0, 4, 4, 4, 0, 4])
        offsets = fit_offsets(probs, y, groups, ctx, config={"steps": [.1], "max_passes": 1})
        restored = json.loads(json.dumps(offsets, allow_nan=False))
        np.testing.assert_array_equal(predict_labels(probs, groups, offsets, ctx),
                                      predict_labels(probs, groups, restored, ctx))
        self.assertFalse(offsets["conditional_gate"]["enabled"])
        self.assertEqual(offsets["global"][0], 0)
        self.assertEqual(len(slice_rule_context(ctx, np.array([True] * 3 + [False] * 3))["band"]), 3)
        meta["timestamp"] = pd.date_range("2026-09-24", periods=6)
        with self.assertRaises(ValueError):
            fit_offsets(probs, y, groups, {"meta": meta, "f3": f3, "f5": f5}, config={})

    def test_direct_band_cannot_be_overridden_by_offsets(self):
        meta, f3, f5 = context(2)
        f3["band"] = [4, 0]
        offsets = zero_offsets()
        offsets["global"][11] = 1.5
        pred = predict_labels(p(2, 11), np.array(["direct_band", "other"]), offsets,
                              {"meta": meta, "f3": f3, "f5": f5})
        self.assertEqual(pred[0], 4)


class PipelineShapeTests(unittest.TestCase):
    def test_forward_teacher_sequence_with_mock_estimator(self):
        dates = ["2026-07-02", "2026-07-03", "2026-07-16", "2026-08-01",
                 "2026-09-05", "2026-09-17", "2026-09-25", "2026-10-01", "2026-10-02"]
        b = pd.DataFrame({"txn_id": [f"t{i}" for i in range(9)],
                          "timestamp": dates, "month": [d[:7] for d in dates],
                          "split": ["train"] * 8 + ["test"],
                          "label": ["legit", "sim_swap"] * 4 + ["unread"],
                          "amount": [1.] * 9, "receiver_type": ["P2P"] * 9})
        packs = {"v1": b[["txn_id"]].assign(flag1=1.),
                 "v2": b[["txn_id"]].assign(flag2=1.),
                 "v4": b[["txn_id"]].assign(flag4=1.)}
        config = {"device": "cpu", "splits": {
            "fit": ["07-01", "09-01"], "early_stop": ["09-01", "09-16"],
            "calibration": ["09-16", "09-24"], "assessment": ["09-24", "10-01"]},
            "members": [{"name": "A", "seed": 17, "num_leaves": 63,
                         "min_child_samples": 20, "colsample_bytree": .5, "weight": 1.}]}
        calls = []
        class Booster:
            def save_model(self, path):
                Path(path).write_text("mock", encoding="utf-8")
        class Model:
            podium_classes_ = np.arange(12)
            podium_backend_ = "mock"
            booster_ = Booster()
            def __init__(self, positions):
                self.positions = positions
            def predict_proba(self, X):
                # Every predicted training row must follow the latest teacher
                # fitting row; test teacher calls also obey this fixture order.
                self_test = np.array(X.index)
                if len(self_test):
                    assert self.positions.max() < self_test.min()
                return p(len(X), 4)
        def mock_train(X, *args, **kwargs):
            calls.append((X.index.tolist(), kwargs["fixed_rounds"]))
            return Model(np.array(X.index)), 500
        series = calendar_dates(b, config)
        masks, year = split_masks(b, series, config)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            with patch("fit_submit.train_member", side_effect=mock_train):
                _bootstrap(b, packs, series, masks, year, config, folder, "synthetic")
            saved = load_probability_artifact(folder / "forward_teacher.npz", b.txn_id, allow_missing=True)
            self.assertTrue(np.isnan(saved[:2]).all())
            self.assertTrue(np.isfinite(saved[2:8]).all())
            self.assertTrue(np.isnan(saved[8]).all())
        self.assertEqual(len(calls), 6)
        self.assertTrue(all(rounds == 500 for _, rounds in calls))

    def test_october_tails_do_not_encode_earlier_test_rows(self):
        b = pd.DataFrame({"txn_id": ["july", "sept", "tail", "oct_test", "nov_test"],
                          "timestamp": ["2026-07-01", "2026-09-01", "2026-10-03",
                                        "2026-10-01", "2026-11-01"],
                          "month": ["2026-07", "2026-09", "2026-10", "2026-10", "2026-11"],
                          "sender_id": ["A"] * 5, "receiver_id": ["B"] * 5,
                          "receiver_type": ["P2P"] * 5, "device_id": ["D"] * 5,
                          "label": ["legit", "sim_swap", "mule_account", "unread", "unread"]})
        fit = np.array([True, True, True, False, False])
        first = build_context_te(b, fit).set_index("txn_id")
        b.loc[2, "label"] = "merchant_collusion"
        second = build_context_te(b, fit).set_index("txn_id")
        pd.testing.assert_series_equal(first.loc["oct_test"], second.loc["oct_test"])
        self.assertNotEqual(first.loc["nov_test", "v6_te_snd_1"],
                            second.loc["nov_test", "v6_te_snd_1"])

    def test_validation_refit_artifact_shapes_with_mock_estimator(self):
        dates = ["2026-07-01", "2026-08-01", "2026-09-05", "2026-09-06",
                 "2026-09-16", "2026-09-20", "2026-09-24", "2026-09-25",
                 "2026-10-01", "2026-10-02"]
        n = len(dates)
        b = pd.DataFrame({"txn_id": [f"id{i}" for i in range(n)],
                          "timestamp": dates, "month": [date[:7] for date in dates],
                          "receiver_type": ["P2P"] * n, "amount": [1.] * n,
                          "split": ["train"] * (n - 1) + ["test"],
                          "label": ["legit", "sim_swap"] * 4 + ["legit", "unread"]})
        _, f3, f5 = context(n)
        f3["txn_id"], f5["txn_id"] = b.txn_id, b.txn_id
        packs = {"v1": b[["txn_id"]].assign(signature=1.),
                 "v2": b[["txn_id"]].assign(S_flag=0., Ds_n=0.),
                 "v3": f3, "v4": b[["txn_id"]].assign(
                     is_tiny_p2p_pay=0., bait_prev_dt_h=np.nan), "v5": f5}
        config = {"splits": {"fit": ["07-01", "09-01"], "early_stop": ["09-01", "09-16"],
                             "calibration": ["09-16", "09-24"], "assessment": ["09-24", "10-01"]},
                  "device": "cpu", "profiles": ["control"], "legit_fraction": 1.,
                  "members": [{"name": "A", "seed": 17, "num_leaves": 63,
                               "min_child_samples": 20, "colsample_bytree": .5, "weight": 1.}],
                  "calibration": {"steps": [.1], "max_passes": 1}}
        class Booster:
            def save_model(self, path):
                Path(path).write_text("mock estimator, no learning", encoding="utf-8")
        class Model:
            podium_classes_ = np.arange(12)
            podium_backend_ = "mock"
            booster_ = Booster()
            def predict_proba(self, X):
                return p(len(X), 4)
        calls = []
        def mock_train(*args, **kwargs):
            calls.append(kwargs)
            return Model(), 7
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config["_root"] = str(root)
            dates_series = calendar_dates(b, config)
            masks, _ = split_masks(b, dates_series, config)
            raw = root / "data/raw"
            raw.mkdir(parents=True)
            pd.DataFrame({"id": ["id9"], "label": ["legit"]}).to_csv(
                raw / "sample_submission.csv", index=False)
            with patch("fit_submit.train_member", side_effect=mock_train):
                clean = pd.DataFrame({"txn_id": b.txn_id[masks["fit"]],
                                      "y_clean": [0, 4], "reason": ["unchanged"] * 2})
                report = _profile_run(b, packs, clean, dates_series, masks, config,
                                      root, "control", "validation", "synthetic-signature")
                self.assertEqual(report["partitions"]["assessment"]["n"], 2)
                self.assertEqual(report["partitions"]["tail"]["n"], 1)
                clean = pd.DataFrame({"txn_id": b.txn_id[masks["train"]],
                                      "y_clean": [0, 4] * 4 + [0],
                                      "reason": ["unchanged"] * 9})
                result = _profile_run(b, packs, clean, dates_series, masks, config,
                                      root, "control", "refit", "synthetic-signature")
                self.assertEqual(result["rows"], 1)
                loaded = load_probability_artifact(
                    root / "control/refit/test_proba.npz", ["id9"])
                self.assertEqual(loaded.shape, (1, 12))
            self.assertIsNone(calls[0]["fixed_rounds"])
            self.assertEqual(calls[1]["fixed_rounds"], 7)

    def test_calendar_and_tail_partitions(self):
        meta = pd.DataFrame({"timestamp": ["2026-07-10", "2026-08-20", "2026-09-05",
                                          "2026-09-17", "2026-09-25", "2026-10-01", "2026-10-02"],
                              "split": ["train"] * 6 + ["test"]})
        config = {"splits": {"fit": ["07-01", "09-01"], "early_stop": ["09-01", "09-16"],
                              "calibration": ["09-16", "09-24"], "assessment": ["09-24", "10-01"]}}
        dates = calendar_dates(meta, config)
        masks, year = split_masks(meta, dates, config)
        self.assertEqual(year, 2026)
        self.assertEqual(masks["fit"].sum(), 2)
        self.assertEqual(masks["tail"].sum(), 1)
        self.assertEqual(masks["assessment"].sum(), 1)

    def test_chain_context_band_is_rebuilt(self):
        b = pd.DataFrame({"txn_id": ["a", "b", "c"], "t": np.array([0, 10, 20], dtype=np.int64),
                          "sender_id": ["A", "B", "C"], "receiver_id": ["B", "C", "D"],
                          "receiver_type": ["P2P"] * 3, "amount": [100., 98., 96.]})
        f1 = pd.DataFrame({"txn_id": b.txn_id, "snd_vict_in_cnt_48h": [3., 0, 0],
                           "is_1p02": [1., 0, 0], "rcv_vict_cnt_24h": [4., 0, 0]})
        bands = pd.DataFrame({"txn_id": b.txn_id, "band": [1, 0, 0]})
        chain = rebuild_chain_features(b, f1, bands)
        self.assertEqual(chain.ch_band.tolist(), [1, 1, 1])
        self.assertEqual(chain.ch_len.tolist(), [3., 3., 3.])
        bands["band"] = 4
        self.assertEqual(rebuild_chain_features(b, f1, bands).ch_band.tolist(), [4, 4, 4])

    def test_frozen_schema_unknown_category(self):
        b = pd.DataFrame({"txn_id": ["a", "b"], "txn_type": ["PAY", "PAY"], "amount": [1., 2.]})
        X, schema, _ = _frozen_features(b, {})
        changed = b.assign(txn_type=["UNKNOWN", "PAY"])
        X2, _, unknown = _frozen_features(changed, {}, schema)
        self.assertEqual(X2.txn_type.tolist(), [-1, 0])
        self.assertEqual(unknown["txn_type"], 1)


class SubmissionTests(unittest.TestCase):
    def test_sample_order_and_absent_classes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "submission.csv"
            sample = pd.DataFrame({"id": ["b", "a"]})
            report = write_submission(np.array(["a", "b"]), np.array([4, 0]), sample, path)
            result = pd.read_csv(path)
            self.assertEqual(result.columns.tolist(), ["id", "label"])
            self.assertEqual(result.id.tolist(), ["b", "a"])
            self.assertEqual(result.label.tolist(), ["legit", "sim_swap"])
            self.assertEqual(report["rows"], 2)
            self.assertIn("mule_account", report["absent_classes"])
            with self.assertRaises(ValueError):
                write_submission(np.array(["a", "extra"]), np.array([4, 0]), sample, path)
            with self.assertRaises(ValueError):
                write_submission(np.array(["a", "a"]), np.array([4, 0]), sample, path)


if __name__ == "__main__":
    unittest.main()
