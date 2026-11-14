"""End-to-end runner tests with mocked estimators; no real training."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import event_boost_train as tool
from test_podium_predictions import zero_offsets


class Booster:
    def save_model(self, path):
        Path(path).write_text("mock model", encoding="utf-8")


class Model:
    booster_ = Booster()
    podium_classes_ = np.arange(12)


def probabilities(model, X):
    p = np.full((len(X), 12), .1/11)
    p[np.arange(len(X)), X.x.to_numpy(dtype=int)] = .9
    return p


class EventBoostTests(unittest.TestCase):
    def test_feature_join_is_id_aligned_and_rejects_missing_duplicates(self):
        rows = pd.DataFrame({"txn_id": ["a", "b"]})
        X = pd.DataFrame({"x": [10., 20.]})
        features = pd.DataFrame({"txn_id": ["b", "a"], **{f"ec{k}": [2., 1.] for k in range(39)}})
        result = tool.append_features(X, rows, features)
        self.assertEqual(result.ec0.tolist(), [1., 2.])
        for bad in (features.iloc[:1], pd.concat([features, features.iloc[:1]])):
            with self.assertRaises(ValueError):
                tool.append_features(X, rows, bad)
        with self.assertRaises(ValueError):
            tool.append_features(X, rows, features.rename(columns={"ec0": "x"}))

    def test_validation_refit_one_model_fixed_rounds_and_csv_order(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            source, features, out = root/"source", root/"features", root/"out"
            features.mkdir()
            config = {"model_defaults": {"n_jobs": 1}, "flat": {"seed": 17}, "calibration": {"conditional": False}}
            manifest = {"signature": "candidate", "config": config, "reference_f1": .52}
            feature_ids, calls = [], []
            for stage in ("validation", "refit"):
                cache = (source if stage == "validation" else out)/stage/"cache"
                cache.mkdir(parents=True)
                days = (["2026-08-01", "2026-09-02", "2026-09-16", "2026-09-20", "2026-09-25"]
                        if stage == "validation" else ["2026-08-01", "2026-10-01"])
                y = np.tile(np.arange(12), len(days))
                n = len(y)
                row = pd.DataFrame({"txn_id": [f"{stage}_{i}" for i in range(n)],
                    "timestamp": np.repeat(days, 12), "receiver_type": "P2P", "y": y, "weight": 1.,
                    "fit": np.arange(n)<12, "query": np.arange(n)>=12, "early_stop": False})
                if stage == "validation":
                    row.loc[12:23, "early_stop"] = True
                row.to_parquet(cache/"rows.parquet", index=False)
                pd.DataFrame({"x": y}).to_parquet(cache/"X.parquet", index=False)
                f3 = pd.DataFrame({"txn_id": row.txn_id, "band": 0})
                for k in range(1, 12):
                    f3[f"snd_band_{k}_3h"], f3[f"rcv_band_{k}_24h"] = 0, 0
                f3.to_parquet(cache/"v3.parquet", index=False)
                pd.DataFrame({"txn_id": row.txn_id, "ch_band": 0}).to_parquet(cache/"v5.parquet", index=False)
                np.save(cache/"groups.npy", np.full(n, "other"))
                tool.write(cache/"schema.json", {"categorical_mappings": {}})
                feature_ids.extend(row.txn_id)
                pd.DataFrame({"txn_id": feature_ids, **{f"ec{k}": 0. for k in range(39)}}).to_parquet(
                    features/"event_composition.parquet", index=False)
                if stage == "validation":
                    tool.write(source/"manifest.json", {"signature": "source"})
                    tool.write(source/"validation/baseline_offsets.json", zero_offsets())
                    baseline = np.full((48, 12), .1/11)
                    baseline[:, 0] = .9
                    tool.save_probability_artifact(source/"validation/flat_proba.npz", row.txn_id[12:], baseline,
                        provenance={"signature": "source", "stage": "validation", "class_order": tool.LABELS,
                                    "fit_ids_hash": tool.ids_hash(row.txn_id[:12])})
                    base_f1 = tool.evaluate_predictions(np.arange(12), np.zeros(12, dtype=int))["macro_f1"]
                    tool.write(source/"validation/metrics.json", {"partitions": {"assessment": {"flat_baseline": {"macro_f1": base_f1}}}})
                else:
                    (root/"data/raw").mkdir(parents=True)
                    sample = pd.DataFrame({"id": row.txn_id[12:].iloc[::-1], "label": "legit"})
                    sample.to_csv(root/"data/raw/sample_submission.csv", index=False)
                def fake_train(X_fit, y_fit, weights, X_stop, y_stop, **kwargs):
                    calls.append((len(X_fit), X_stop, y_stop, kwargs))
                    return Model(), 7
                with patch.object(tool, "train_member", side_effect=fake_train), \
                     patch.object(tool, "model_proba", side_effect=probabilities), \
                     patch.object(tool, "fit_offsets", return_value=zero_offsets()):
                    tool.train_stage(root, source, features, out, manifest, stage)
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][0], 12)
            self.assertEqual(len(calls[0][1]), 12)
            self.assertIsNone(calls[0][3]["fixed_rounds"])
            self.assertIsNone(calls[1][1])
            self.assertIsNone(calls[1][2])
            self.assertEqual(calls[1][3]["fixed_rounds"], 7)
            report = tool.read(out/"validation/comparison.json")
            self.assertTrue(report["eligible_for_refit"])
            self.assertEqual(report["candidates"]["event_features"]["assessment"]["macro_f1"], 1.)
            self.assertEqual(len(report["candidates"]["event_features"]["assessment"]["per_class"]), 12)
            result = pd.read_csv(out/"refit/submission.csv")
            self.assertEqual(result.id.tolist(), sample.id.tolist())
            self.assertEqual(result.label.tolist(), list(reversed(tool.LABELS)))
            with patch.object(tool, "train_member", side_effect=AssertionError("must not fit again")):
                tool.train_stage(root, source, features, out, manifest, "refit")

    def test_no_refit_if_gate_rejects_even_when_assessment_is_better(self):
        from test_boundary_calibration import fixture
        ctx = fixture(36)
        ctx["dates"] = pd.to_datetime(np.repeat(["2026-09-16", "2026-09-20", "2026-09-25"], 12), utc=True).to_numpy()
        rows = pd.DataFrame({"timestamp": ctx["dates"]})
        y = np.tile(np.arange(12), 3)
        baseline = probabilities(None, pd.DataFrame({"x": y}))
        candidate = baseline.copy()
        candidate[12:24] = baseline[12:24, ::-1]
        baseline[24:] = probabilities(None, pd.DataFrame({"x": np.zeros(12, dtype=int)}))
        with patch.object(tool, "fit_offsets", return_value=zero_offsets()):
            _, report = tool.compare_candidate(candidate, baseline, y, rows, np.full(36, "other"), ctx,
                                               {}, zero_offsets(), .52)
        self.assertEqual(report["selected_on_gate"], "baseline")
        self.assertFalse(report["eligible_for_refit"])


if __name__ == "__main__":
    unittest.main()
