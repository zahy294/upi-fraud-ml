"""No real fits: synthetic alignment, chronology, inference and runner tests."""
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

from src.upi.hierarchical import combine_probabilities, fit_detector, fit_types
from src.upi.motifs import build_motifs
from src.upi.event_rescue import event_evidence, propagate
import hierarchical_train as runner


def rows():
    return pd.DataFrame({"txn_id": list("abcdef"), "split": "train",
        "timestamp": pd.to_datetime(["2026-08-01 00:00", "2026-08-01 00:01",
            "2026-08-01 00:02", "2026-08-01 00:03", "2026-08-01 00:04", "2026-09-01 00:00"]),
        "sender_id": "S", "receiver_id": "R", "device_id": "D",
        "receiver_type": "P2P", "txn_type": "PAY", "new_payee": 1,
        "amount": [50, 50, 50, 50, 50, 50], "balance_before": [10000, 7777, 2500, 1777, 625, 156.25]})


class HierarchyTests(unittest.TestCase):
    def test_combination_preserves_binary_mass_and_mapping(self):
        conditional = np.zeros((3, 12))
        conditional[:, 4], conditional[:, 11] = .25, .75
        p = combine_probabilities(np.array([0, .5, 1]), conditional)
        np.testing.assert_allclose(p.sum(1), 1)
        np.testing.assert_allclose(p[:, 0], [1, .5, 0])
        self.assertEqual(p[1, 11], .375)
        with self.assertRaises(ValueError):
            combine_probabilities([.5], np.ones((1, 12)) / 12)

    def test_motifs_find_skipped_drain_without_labels_and_are_order_stable(self):
        base = rows()
        result = build_motifs(base)
        self.assertAlmostEqual(result.motif_quarter_skip_residual[0], 0)
        self.assertGreater(result.motif_quarter_nearest_residual[0], .1)
        self.assertEqual(result.motif_quarter_matching_links[2], 2)
        self.assertEqual(result.motif_quarter_matching_links[5], 0)
        shuffled = base.iloc[[4, 2, 0, 5, 1, 3]].copy()
        shuffled["label"], shuffled["ip_risk_score"] = "nonsense", -100
        other = build_motifs(shuffled).set_index("txn_id").loc[base.txn_id].reset_index()
        pd.testing.assert_frame_equal(result, other)

    def test_peer_does_not_use_self_or_cross_split(self):
        base = rows()
        base.loc[1:, "split"] = "test"
        p = np.eye(12)[[1, 0, 0, 0, 0, 0]]
        evidence, _ = event_evidence(base, p)
        self.assertFalse(evidence.any())
        base["split"] = "train"
        evidence, _ = event_evidence(base, p)
        self.assertEqual(evidence[0, 0], 0)
        self.assertGreater(evidence[1, 0], 0)
        self.assertEqual(evidence[-1, 0], 0)
        np.testing.assert_allclose(propagate(p, evidence, 0), p, atol=1e-8)
        self.assertTrue(np.isfinite(propagate(p, evidence, 1)).all())

    def test_type_fit_has_no_legitimate_rows_and_refit_has_no_stop_set(self):
        X = pd.DataFrame({"x": np.arange(24)})
        y = np.tile(np.arange(12), 2)
        with patch("src.upi.hierarchical.train_member", return_value=(object(), 7)) as train:
            fit_types(X, y, np.ones(24), X, y, spec={})
            args = train.call_args.args
            self.assertEqual(len(args[0]), 22)
            self.assertTrue((args[1] > 0).all())
            self.assertTrue((args[4] > 0).all())
            fit_types(X, y, np.ones(24), None, None, spec={}, rounds=7)
            self.assertIsNone(train.call_args.args[3])
            self.assertIsNone(train.call_args.args[4])

    def test_detector_uses_binary_objective_and_original_stop_targets(self):
        calls = []
        class Fake:
            best_iteration_ = 8
            def __init__(self, **kwargs):
                calls.append(kwargs)
            def fit(self, X, y, **kwargs):
                calls.append((y, kwargs))
        lgb = types.SimpleNamespace(LGBMClassifier=Fake, early_stopping=lambda *a, **k: None,
                                    log_evaluation=lambda *a: None)
        spec = dict(max_rounds=100, learning_rate=.05, num_leaves=15, min_child_samples=10,
                    colsample_bytree=.8, n_jobs=1, seed=1, early_stopping_rounds=10, categorical_columns=[])
        X = pd.DataFrame({"x": [1, 2, 3]})
        with patch.dict("sys.modules", {"lightgbm": lgb}):
            _, count = fit_detector(X, [0, 4, 11], np.ones(3), X, [0, 1, 7], spec=spec)
            self.assertEqual(count, 8)
            self.assertEqual(calls[0]["objective"], "binary")
            self.assertEqual(calls[0]["metric"], "average_precision")
            np.testing.assert_array_equal(calls[1][0], [0, 1, 1])
            calls.clear()
            fit_detector(X, [0, 4, 11], np.ones(3), None, None, spec=spec, rounds=8)
            self.assertNotIn("eval_set", calls[1][1])

    def test_mock_runner_validation_then_refit_and_checked_csv(self):
        # Real selection, probability combination, metadata and CSV code;
        # estimators are deterministic fakes, so this performs no training.
        class Booster:
            def save_model(self, path):
                Path(path).write_text("fake model")
        class Model:
            booster_ = Booster()
            def __init__(self, kind):
                self.kind = kind
                self.classes_ = np.arange(2)
                self.podium_classes_ = np.arange(1, 12) if kind == "types" else np.arange(12)
            def predict_proba(self, X):
                y = X.x.to_numpy(dtype=int)
                if self.kind == "detector":
                    p = np.where(y > 0, .95, .05)
                    return np.column_stack([1-p, p])
                if self.kind == "types":
                    p = np.full((len(y), 11), .01)
                    p[np.arange(len(y)), np.maximum(y, 1)-1] = .9
                    return p
                p = np.full((len(y), 12), .1/11)
                p[:, 0] = .9
                return p
        config = runner.read(Path(__file__).parents[1] / "configs/hierarchy.json")
        config["calibration"] = {"steps": [.1], "max_passes": 1, "conditional": False}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out = root / "run"
            for stage in ("validation", "refit"):
                cache = out / stage / "cache"
                cache.mkdir(parents=True)
                days = ["2026-07-01", "2026-09-02", "2026-09-16", "2026-09-20", "2026-09-25"] if stage == "validation" else ["2026-07-01", "2026-10-01"]
                y = np.tile(np.arange(12), len(days))
                n = len(y)
                row = pd.DataFrame({"txn_id": [f"{stage}_{i}" for i in range(n)], "timestamp": np.repeat(days, 12),
                    "receiver_type": "P2P", "fit": np.arange(n)<12, "query": np.arange(n)>=12,
                    "y": y, "weight": 1., "early_stop": False, "calibration": False,
                    "assessment": False, "full_september": False, "tail": False})
                if stage == "validation":
                    row.loc[12:23, "early_stop"] = True
                    row.loc[24:47, "calibration"] = True
                    row.loc[48:, "assessment"] = True
                    row.loc[12:, "full_september"] = True
                else:
                    sample = pd.DataFrame({"id": row.txn_id[12:].iloc[::-1], "label": "legit"})
                    (root / "data/raw").mkdir(parents=True)
                    sample.to_csv(root / "data/raw/sample_submission.csv", index=False)
                row.to_parquet(cache / "rows.parquet", index=False)
                pd.DataFrame({"x": y}).to_parquet(cache / "X.parquet", index=False)
                f3 = pd.DataFrame({"txn_id": row.txn_id, "band": 0})
                for k in range(1, 12):
                    f3[f"snd_band_{k}_3h"] = 0
                    f3[f"rcv_band_{k}_24h"] = 0
                f3.to_parquet(cache / "v3.parquet", index=False)
                pd.DataFrame({"txn_id": row.txn_id, "ch_band": 0}).to_parquet(cache / "v5.parquet", index=False)
                np.save(cache / "groups.npy", np.full(n, "other"))
                runner.write(cache / "schema.json", {"categorical_mappings": {}})
                with patch.object(runner, "train_member", return_value=(Model("flat"), 2)), \
                     patch.object(runner, "fit_detector", return_value=(Model("detector"), 3)), \
                     patch.object(runner, "fit_types", return_value=(Model("types"), 4)):
                    runner.train_stage(root, out, config, "sig", stage)
            selection = runner.read(out / "validation/selection.json")
            self.assertEqual(selection["member_weights"]["H"], 1.)
            csv = pd.read_csv(out / "refit/submission.csv")
            self.assertEqual(csv.id.tolist(), sample.id.tolist())
            self.assertEqual(csv.label.tolist(), list(reversed(runner.LABELS)))
            self.assertTrue((out / "refit/complete.json").exists())


if __name__ == "__main__":
    unittest.main()
