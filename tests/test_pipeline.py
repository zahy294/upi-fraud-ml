"""Failure-mode checks for temporal isolation, group windows and keyed outputs."""
import json
from pathlib import Path
import tempfile
import unittest

import lightgbm as lgb
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.metrics import f1_score

from upi_fraud.artifacts import save_probabilities
from upi_fraud.constants import FEATURES, LABELS
from upi_fraud.data import align, check_submission
from upi_fraud.decisions import (apply_combination, evaluate, fit_decisions, predictions)
from upi_fraud.features import Events, build_features
from upi_fraud.model import fit_model, fitting_rows, predict_proba
from upi_fraud.run import read_config

REPO = Path(__file__).resolve().parents[1]


def fixture():
    rows = []
    for i in range(144):
        sender = i % 6 + 1
        receiver = (sender % 6) + 1
        # All twelve classes in July, August and September; identity/timestamp
        # attributes are independent of labels, which are deliberately artificial.
        month = 7 + i // 48
        rows.append({
            "txn_id": f"TX{i:08d}", "timestamp": pd.Timestamp(f"2026-{month:02d}-01") + pd.Timedelta(seconds=i % 48 * 60),
            "sender_id": f"AC{sender:06d}", "receiver_id": f"AC{receiver:06d}" if i % 3 else "MR000001",
            "receiver_type": "P2P" if i % 3 else "MERCHANT", "device_id": f"DV0{sender:06d}",
            "amount": float([5, 500, 4999, 1000][i % 4]), "txn_type": ["PAY", "QR", "REFUND", "COLLECT"][i % 4],
            "merchant_category": "gaming" if i % 3 == 0 else None, "device_city": "CityA",
            "sim_age_hours": 24. if i % 2 else 5000., "device_age_days": 2. if i % 2 else 100.,
            "sender_account_age_days": 500., "new_payee": 1, "pin_retries": 0,
            "login_to_txn_sec": 5., "balance_before": 10000., "app": "appA", "qr_vpa_mismatch": np.nan,
            "label": LABELS[i % 12], "split": "train", "_source_order": i,
        })
    accounts = pd.DataFrame({"account_id": [f"AC{i:06d}" for i in range(1, 7)],
                             "home_city": ["CityA"] * 6, "opened_on": ["2020-01-01"] * 6})
    merchants = pd.DataFrame({"merchant_id": ["MR000001"], "onboarded_on": ["2020-01-01"]})
    return pd.DataFrame(rows), accounts, merchants


class EventsTests(unittest.TestCase):
    def test_closed_boundaries_no_neighbour_and_group_separation(self):
        e = Events(np.array([1, 1, 1, 2]), np.array([0, 300, 301, 300]), np.array([10, 20, 30, 90]))
        np.testing.assert_array_equal(e.count(np.array([1, 2, 3, -1]), np.array([0, 0, 0, 0]), 0, 300), [2, 1, 0, 0])
        got = e.adjacent_value(np.array([1, 2, 3]), np.array([0, 300, 0]), 300, True)
        np.testing.assert_allclose(got, [20, np.nan, np.nan], equal_nan=True)
        empty = Events(np.array([], dtype=int), np.array([], dtype=int), np.array([]))
        self.assertTrue(np.isnan(empty.adjacent_value(np.array([1]), np.array([0]), 10, True)[0]))

    def test_same_timestamp_excluded_and_group_boundary_checked(self):
        e = Events(np.array([1, 1, 1, 2]), np.array([10, 10, 11, 1]), np.array([10, 20, 30, 99]))
        self.assertEqual(e.adjacent_value(np.array([1]), np.array([10]), 2, True)[0], 30)
        self.assertTrue(np.isnan(e.adjacent_value(np.array([1]), np.array([11]), 100, True)[0]))
        self.assertTrue(np.isnan(e.adjacent_value(np.array([2]), np.array([1]), 100, False)[0]))


class FeatureTests(unittest.TestCase):
    def test_shuffle_and_label_perturbation_leave_features_unchanged(self):
        raw, accounts, merchants = fixture()
        expected, schema = build_features(raw, accounts, merchants)
        changed = raw.sample(frac=1, random_state=41).copy()
        changed["label"] = "wrong_label_that_must_never_be_read"
        changed["app_version"], changed["ip_risk_score"] = "99.9", 1.
        actual, actual_schema = build_features(changed, accounts, merchants, expected.txn_id)
        assert_frame_equal(expected, actual)
        self.assertEqual(schema, actual_schema)
        self.assertFalse({"label", "app_version", "ip_risk_score"} & set(actual))

    def test_tiny_flow_and_recipient_namespaces(self):
        raw, accounts, merchants = fixture()
        raw = raw.iloc[:4].copy()
        raw["timestamp"] = pd.to_datetime(["2026-07-01 00:00:00", "2026-07-01 00:30:00", "2026-07-01 00:00:00", "2026-07-01 00:00:00"])
        raw["sender_id"] = ["AC000001", "AC000002", "AC000003", "AC000004"]
        raw["receiver_id"] = ["AC000002", "AC000003", "MR000001", "AC000001"]
        raw["receiver_type"] = ["P2P", "P2P", "MERCHANT", "P2P"]
        raw["amount"] = [10000., 9800., 100., 200.]
        f, _ = build_features(raw, accounts, merchants, raw.txn_id)
        self.assertAlmostEqual(float(f.fwd_amt_ratio_30m.iloc[0]), .98, places=6)
        self.assertEqual(f.rcv_out_30m.iloc[0], 1)
        self.assertEqual(f.rcv_cnt_total.iloc[2], 1)
        self.assertEqual(f.rcv_cnt_total.iloc[3], 1)
        self.assertTrue(np.isnan(f.fwd_amt_ratio_30m.iloc[2]))

    def test_unknown_categories_use_saved_mapping_and_missing_ids_rejected(self):
        raw, accounts, merchants = fixture()
        original, schema = build_features(raw, accounts, merchants)
        raw["app"] = "unseen_app"
        f, frozen = build_features(raw, accounts, merchants, category_schema=schema)
        self.assertTrue((f.app == -1).all())
        self.assertEqual(schema, frozen)
        with self.assertRaises(ValueError):
            align(original, ["missing"])
        with self.assertRaises(ValueError):
            align(pd.concat([original, original.iloc[:1]]), original.txn_id)


class ModelAndDecisionTests(unittest.TestCase):
    def test_training_and_prediction_do_not_read_validation_labels(self):
        raw, accounts, merchants = fixture()
        config = read_config(REPO / "configs/measured.json")
        config["negative_sample_fraction"] = 1.
        config["model"].update(n_estimators=4, min_child_samples=2, n_jobs=1)
        raw = raw[raw.timestamp < pd.Timestamp("2026-09-01")]
        f, schema = build_features(raw, accounts, merchants)
        model, info = fit_model(f, raw[["txn_id", "label"]], "2026-08-01", config)
        query = f[f.timestamp >= pd.Timestamp("2026-08-01")]
        expected = predict_proba(model, query)
        changed = raw.copy()
        changed.loc[changed.timestamp >= pd.Timestamp("2026-08-01"), "label"] = "merchant_collusion"
        altered, _ = build_features(changed, accounts, merchants, category_schema=schema)
        again, _ = fit_model(altered, changed[["txn_id", "label"]], "2026-08-01", config)
        np.testing.assert_array_equal(expected, predict_proba(again, query))
        self.assertLess(pd.Timestamp(info["fit_max_timestamp"]), pd.Timestamp("2026-08-01"))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.txt"
            model.save_model(str(path))
            restored = lgb.Booster(model_file=str(path))
            np.testing.assert_array_equal(expected, predict_proba(restored, query))
        reversed_frame = query.iloc[::-1]
        np.testing.assert_array_equal(expected[::-1], predict_proba(model, reversed_frame))

    def test_october_tails_excluded_and_sampling_targets_align_by_key(self):
        raw, accounts, merchants = fixture()
        raw.loc[raw.index[-1], "timestamp"] = pd.Timestamp("2026-10-02")
        f, _ = build_features(raw, accounts, merchants)
        mask, y, weights = fitting_rows(f, raw[["txn_id", "label"]].sample(frac=1), "2026-09-01", 1., 137)
        self.assertTrue((f.loc[mask, "timestamp"] < pd.Timestamp("2026-09-01")).all())
        self.assertEqual(int(mask.sum()), 96)
        self.assertTrue((weights == 1).all())
        self.assertEqual(y[-1], 11)

    def test_metric_matches_sklearn_and_rule_priority_is_probability_gated(self):
        y = np.arange(120) % 12
        pred = np.arange(120) % 11
        self.assertAlmostEqual(evaluate(y, pred)["macro_f1"], f1_score(y, pred, labels=list(range(12)), average="macro", zero_division=0))
        P = np.full((2, 12), .01)
        P[:, 0] = .89
        rules = np.array([4, 4])
        thresholds = [None] * 12
        thresholds[4] = .02
        self.assertEqual(apply_combination(P, rules, np.ones(12), thresholds).tolist(), [0, 0])
        thresholds[4] = 0.
        self.assertEqual(apply_combination(P, rules, np.ones(12), thresholds).tolist(), [4, 4])

    def test_probability_artifact_and_decision_class_order(self):
        raw, accounts, merchants = fixture()
        frame, _ = build_features(raw.iloc[:12], accounts, merchants)
        P = np.full((12, 12), 1 / 12, dtype="float32")
        config = read_config(REPO / "configs/measured.json")
        decision = fit_decisions(P, np.arange(12), np.zeros(12, dtype=int), config)
        pred = predictions(P, np.zeros(12, dtype=int), decision)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "probabilities.parquet"
            save_probabilities(path, frame, P, np.zeros(12, dtype=int), pred, np.arange(12))
            loaded = pd.read_parquet(path)
            self.assertEqual(loaded.txn_id.tolist(), frame.txn_id.tolist())
            self.assertEqual(json.loads(path.with_suffix(".schema.json").read_text())["label_order"], list(LABELS))
        decision["label_order"] = list(reversed(LABELS))
        with self.assertRaises(ValueError):
            predictions(P, np.zeros(12, dtype=int), decision)


class SubmissionTests(unittest.TestCase):
    def test_reject_bad_ids_order_labels_and_columns(self):
        sample = pd.DataFrame({"id": ["a", "b"], "label": ["legit", "legit"]})
        self.assertEqual(check_submission(sample, sample)["rows"], 2)
        invalid = [sample.iloc[::-1], sample.iloc[:1], sample.assign(id=["a", "a"]),
                   sample.assign(label=["legit", "typo"]), sample.assign(label=["legit", None]),
                   sample.rename(columns={"id": "txn_id"})]
        for frame in invalid:
            with self.subTest(frame=frame.to_dict()), self.assertRaises(ValueError):
                check_submission(frame, sample)


if __name__ == "__main__":
    unittest.main()
