"""Synthetic tests, with all estimator fitting replaced by mocks."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.upi.event_composition import build_event_composition
import forest_experiment as forest


class ForestTests(unittest.TestCase):
    def test_composition_leave_self_out_missingness_and_isolation(self):
        frame = pd.DataFrame({"txn_id": list("abcde"),
            "timestamp": ["2026-08-01 00:00", "2026-08-01 00:01", "2026-08-01 00:02", "2026-08-01 00:01", "2026-09-01 00:00"],
            "split": ["train", "train", "train", "test", "train"], "sender_id": "S",
            "receiver_id": "R", "receiver_type": "P2P", "device_id": "D",
            "amount": [10., 20., 30., 40., 50.], "balance_before": 100.,
            "new_payee": [0, 1, np.nan, 1, 1], "device_age_days": [20, 1, np.nan, 1, 1],
            "sim_age_hours": [100, 1, np.nan, 1, 1]})
        result = build_event_composition(frame)
        self.assertEqual(result.ec_sender2h_peers.tolist(), [2, 2, 2, 0, 0])
        self.assertEqual(result.ec_sender2h_new_payee_support.tolist(), [1, 1, 2, 0, 0])
        self.assertEqual(result.ec_sender2h_new_payee_mean.iloc[0], 1.)
        self.assertEqual(result.ec_sender2h_new_payee_mean.iloc[1], 0.)
        changed = frame.iloc[::-1].copy()
        changed["label"] = "garbage"
        changed["ip_risk_score"] = -99
        other = build_event_composition(changed).set_index("txn_id").loc[frame.txn_id].reset_index()
        pd.testing.assert_frame_equal(result, other)
        frame.loc[1, "receiver_type"] = "MERCHANT"
        self.assertEqual(build_event_composition(frame).ec_receiver1h_peers.iloc[1], 0)

    def test_fixed_onehot_and_unknown_category(self):
        X = pd.DataFrame({"cat": [-1, 0, 1], "amount": [1., np.nan, 3.]})
        encoded = forest.encode_categories(X, {"categorical_mappings": {"cat": ["a", "b"]}})
        self.assertNotIn("cat", encoded)
        np.testing.assert_array_equal(encoded.filter(like="forest_cat").to_numpy(), np.eye(3))

    def test_imputation_fits_only_supplied_training_rows(self):
        import sklearn.ensemble
        calls = []
        class Estimator:
            def __init__(self, **config):
                pass
            def fit(self, X, y, **kwargs):
                calls.append((X.copy(), y.copy(), kwargs))
        with patch.object(sklearn.ensemble, "ExtraTreesClassifier", Estimator):
            imputer, _ = forest.fit_forest(pd.DataFrame({"v": [1., 3., np.nan]}),
                                          np.array([0, 1, 0]), np.array([4., 1., 4.]), {})
        self.assertEqual(imputer.statistics_[0], 2.)
        np.testing.assert_array_equal(calls[0][2]["sample_weight"], [4, 1, 4])
        self.assertEqual(imputer.transform(pd.DataFrame({"v": [1000., np.nan]}))[1, 0], 2.)

    def test_probability_mapping_for_missing_classes(self):
        class Identity:
            def transform(self, X):
                return X
        class Model:
            classes_ = np.array([0, 4, 11])
            def predict_proba(self, X):
                return np.tile([.2, .3, .5], (len(X), 1))
        p = forest.probability(Identity(), Model(), pd.DataFrame({"x": [1, 2]}))
        np.testing.assert_allclose(p[:, [0, 4, 11]], [[.2, .3, .5]]*2)
        self.assertTrue((p[:, [1, 2, 3]] == 0).all())

    def test_mock_validation_refit_and_submission(self):
        class Identity:
            def transform(self, X):
                return X
        class Model:
            classes_ = np.arange(12)
            def predict_proba(self, X):
                p = np.full((len(X), 12), .1/11)
                p[np.arange(len(X)), X.x.to_numpy(dtype=int)] = .9
                return p
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            source, out = root/"source", root/"out"
            out.mkdir()
            ids = []
            for stage in ("validation", "refit"):
                cache = (source if stage == "validation" else out)/stage/"cache"
                cache.mkdir(parents=True)
                days = ["2026-07-01", "2026-09-02", "2026-09-16", "2026-09-20", "2026-09-25"] if stage == "validation" else ["2026-07-01", "2026-10-01"]
                y = np.tile(np.arange(12), len(days))
                n = len(y)
                row = pd.DataFrame({"txn_id": [f"{stage}{i}" for i in range(n)], "timestamp": np.repeat(days, 12),
                    "receiver_type": "P2P", "fit": np.arange(n)<12, "query": np.arange(n)>=12,
                    "y": y, "weight": 1., "early_stop": False, "calibration": False,
                    "assessment": False, "full_september": False, "tail": False})
                ids.extend(row.txn_id)
                if stage == "validation":
                    row.loc[12:23, "early_stop"] = True
                    row.loc[24:47, "calibration"] = True
                    row.loc[48:, "assessment"] = True
                    row.loc[12:, "full_september"] = True
                else:
                    (root/"data/raw").mkdir(parents=True)
                    sample = pd.DataFrame({"id": row.txn_id[12:].iloc[::-1], "label": "legit"})
                    sample.to_csv(root/"data/raw/sample_submission.csv", index=False)
                row.to_parquet(cache/"rows.parquet", index=False)
                pd.DataFrame({"x": y}).to_parquet(cache/"X.parquet", index=False)
                f3 = pd.DataFrame({"txn_id": row.txn_id, "band": 0})
                for k in range(1, 12):
                    f3[f"snd_band_{k}_3h"], f3[f"rcv_band_{k}_24h"] = 0, 0
                f3.to_parquet(cache/"v3.parquet", index=False)
                pd.DataFrame({"txn_id": row.txn_id, "ch_band": 0}).to_parquet(cache/"v5.parquet", index=False)
                np.save(cache/"groups.npy", np.full(n, "other"))
                forest.write(cache/"schema.json", {"categorical_mappings": {}})
                forest.write(source/"manifest.json", {"config": {"calibration": {"steps": [.1], "max_passes": 1, "conditional": False}}})
                pd.DataFrame({"txn_id": ids, "extra": 0.}).to_parquet(out/"event_composition.parquet", index=False)
                with patch.object(forest, "fit_forest", return_value=(Identity(), Model())), \
                     patch("joblib.dump", side_effect=lambda model, path, **kw: Path(path).write_bytes(b"fake")):
                    forest.run_stage(root, source, out, {}, "sig", stage)
            report = forest.read(out/"validation/metrics.json")
            self.assertEqual(report["partitions"]["assessment"]["macro_f1"], 1.)
            csv = pd.read_csv(out/"refit/submission.csv")
            self.assertEqual(csv.id.tolist(), sample.id.tolist())
            self.assertEqual(csv.label.tolist(), list(reversed(forest.LABELS)))


if __name__ == "__main__":
    unittest.main()
