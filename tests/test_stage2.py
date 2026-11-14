"""Run with python -I tests/test_stage2.py. Tests write only to temporary folders."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/upi"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import CATS, LABELS  # noqa: E402
from events import Events  # noqa: E402
from exp_stage2 import load_inputs, run_stage2  # noqa: E402
from stage2_neighbors import (NeighborFeatures, WINDOWS, fit_months_for, logical_folds,
                              probability_context, restore_population_prior)  # noqa: E402


class NeighborTests(unittest.TestCase):
    def assert_brute_force(self, entities, p):
        names = [f"class_{c}" for c in range(p.shape[1])]
        result = NeighborFeatures(entities).transform(p, names)
        for i, row in entities.iterrows():
            for relation, width in WINDOWS.items():
                if relation == "reverse_pair_7d":
                    same = entities.sender_id.eq(row.receiver_id) & entities.receiver_id.eq(row.sender_id)
                else:
                    key = {"receiver_24h": "receiver_id", "device_10m": "device_id",
                           "sender_3h": "sender_id"}[relation]
                    same = entities[key].eq(row[key])
                selected = same & entities.t.sub(row.t).abs().le(width)
                selected.iloc[i] = False
                total = p[selected].sum(axis=0)
                mean = total / max(int(selected.sum()), 1)
                self.assertEqual(result.loc[i, f"s1_{relation}_count"], selected.sum())
                np.testing.assert_allclose(result.loc[i, [f"s1_{relation}_sum_{c}" for c in names]],
                                           total, atol=2e-6)
                np.testing.assert_allclose(result.loc[i, [f"s1_{relation}_mean_{c}" for c in names]],
                                           mean, atol=2e-6)

    def test_boundaries_duplicates_reverse_direction_and_namespaces(self):
        rng = np.random.RandomState(41)
        time = np.array([-604801, -604800, -86401, -86400, -10801, -10800,
                         -601, -600, 0, 0, 600, 601, 10800, 10801, 86400, 86401,
                         604800, 604801], dtype=np.int64)
        n = len(time)
        entities = pd.DataFrame({"txn_id": [f"x{i}" for i in range(n)], "t": time,
            "sender_id": ["AC1", "AC2", "AC1"] * 6,
            "receiver_id": ["AC2", "AC1", "MR1"] * 6,
            "device_id": ["d0"] * n})
        entities.loc[[8, 9], ["sender_id", "receiver_id"]] = ["AC1", "AC1"]
        p = rng.dirichlet([2, 1, 1], size=n).astype(np.float32)
        self.assert_brute_force(entities, p)
        shuffled = entities.sample(frac=1, random_state=4)
        self.assert_brute_force(shuffled.reset_index(drop=True), p[shuffled.index])

    def test_long_timestamps_and_absent_reverse_pair(self):
        entities = pd.DataFrame({"txn_id": ["a", "b", "c"],
            "sender_id": ["AC1", "AC1", "AC2"], "receiver_id": ["MR1", "MR1", "MR2"],
            "device_id": ["d", "d", "e"], "t": np.array([0, 2**26, 2**26+600], dtype=np.int64)})
        self.assert_brute_force(entities, np.array([[.8,.2], [.2,.8], [.7,.3]], dtype=np.float32))

    def test_original_events_closed_bounds_masks_and_negative_keys(self):
        group = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
        time = np.array([0, 10, 10, 9, 20, 30], dtype=np.int64)
        vals = np.array([1., 2., 3., 4., 5., 6.])
        mask = np.array([True, False, True, True, True, False])
        qg = np.array([0, 0, 1, 2, -1, 7], dtype=np.int64)
        qt = np.array([0, 10, 10, 30, 10, 0], dtype=np.int64)
        for keep in [np.ones(6, dtype=bool), mask, np.zeros(6, dtype=bool)]:
            events = Events(group, time, vals, mask=keep)
            for lo, hi in [(-10, 10), (0, 0), (1, 10), (-10, -1)]:
                expected = [keep & (group == g) & (time >= t+lo) & (time <= t+hi)
                            for g, t in zip(qg, qt)]
                np.testing.assert_array_equal(events.count(qg, qt, lo, hi), [m.sum() for m in expected])
                np.testing.assert_allclose(events.sum(qg, qt, lo, hi), [vals[m].sum() for m in expected])

    def test_invalid_probabilities_and_duplicate_ids_fail(self):
        e = pd.DataFrame({"txn_id": ["a"], "sender_id": ["AC1"], "receiver_id": ["MR1"],
                          "device_id": ["d"], "t": np.array([0], dtype=np.int64)})
        with self.assertRaises(ValueError):
            NeighborFeatures(e).transform(np.array([[.3, .3]]), ["a", "b"])
        with self.assertRaises(ValueError):
            NeighborFeatures(pd.concat([e, e], ignore_index=True))

    def test_original_events_strict_previous_and_next_nonempty(self):
        group = np.array([0, 0, 0, 0, 1], dtype=np.int64)
        time = np.array([0, 10, 10, 20, 10], dtype=np.int64)
        events = Events(group, time)
        query_group = np.array([0, 0, 0, 1, -1], dtype=np.int64)
        query_time = np.array([0, 10, 20, 10, 10], dtype=np.int64)
        np.testing.assert_allclose(events.prev_dt(query_group, query_time, 10),
                                   [np.nan, 10, 10, np.nan, np.nan], equal_nan=True)
        np.testing.assert_allclose(events.next_dt(query_group, query_time, 10),
                                   [10, 10, np.nan, np.nan, np.nan], equal_nan=True)


class FoldTests(unittest.TestCase):
    def test_tails_share_september_fold(self):
        m = pd.DataFrame({"split": ["train"]*4+["test"]*2,
                          "month": ["2026-07", "2026-08", "2026-09", "2026-10", "2026-10", "2026-11"]})
        self.assertEqual(logical_folds(m).tolist(), ["2026-07", "2026-08", "2026-09", "2026-09", "", ""])

    def test_no_query_source_or_outer_month_in_any_fit(self):
        folds = np.array(["J", "A", "S", "", "S", "J"])
        labels = np.array([0, 1, 2, -1, 1, 2])
        available = ("J", "A")
        def provider_for(y):
            def predict(months):
                counts = np.bincount(y[np.isin(folds, months)], minlength=3) + 1
                return np.tile(counts / counts.sum(), (len(folds), 1))
            return predict
        changed = labels.copy()
        changed[folds == "S"] = 0
        for query in ["J", "A", "S"]:
            p, plan = probability_context(folds, available, query, provider_for(labels), 3)
            q, _ = probability_context(folds, available, query, provider_for(changed), 3)
            np.testing.assert_array_equal(p, q)
            for entry in plan:
                self.assertNotIn("S", entry["fit_months"])
                self.assertNotIn(query, entry["fit_months"])
                self.assertNotIn(entry["source_month"], entry["fit_months"])
            changed_query = labels.copy()
            changed_query[folds == query] = 2
            r, _ = probability_context(folds, available, query, provider_for(changed_query), 3)
            np.testing.assert_array_equal(p, r)
        self.assertEqual(fit_months_for(available, "J", "A"), ())
        p, _ = probability_context(folds, available, "J", provider_for(labels), 3)
        np.testing.assert_allclose(p[folds == "A"], 1/3)

    def test_sampling_prior_correction_and_decision_equivalence(self):
        population = np.array([[.96,.02,.02], [.4,.2,.4], [.1,.8,.1]])
        sampling = np.array([.25,1.,1.])
        sampled = population * sampling
        sampled /= sampled.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(restore_population_prior(sampled, .25), population, atol=1e-7)
        w = np.array([1., 2., 3.])
        w_sample = w * np.array([1., .25, .25])
        np.testing.assert_array_equal((population*w).argmax(1), (sampled*w_sample).argmax(1))

    def test_loader_rejects_target_in_extra_feature_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            processed = root / "data/processed"
            processed.mkdir(parents=True)
            b = pd.DataFrame({"txn_id": ["a", "b", "c", "d"], "split": ["train"]*3+["test"],
                "month": ["2026-07", "2026-08", "2026-09", "2026-10"], "label": ["legit"]*3+[None],
                "app_major": ["3"]*4, "ip_risk_score": [0.8]*4,
                "sender_id": ["AC1"]*4, "receiver_id": ["MR1"]*4, "device_id": ["DV1"]*4,
                "t": [0, 1, 2, 3], "amount": [100.]*4})
            for cat in CATS:
                b[cat] = "x"
            b.to_parquet(processed / "base_v0.parquet", index=False)
            pd.DataFrame({"txn_id": b.txn_id, "label": [0.,0.,0.,0.]}).to_parquet(
                processed / "bad.parquet", index=False)
            with self.assertRaisesRegex((ValueError, AssertionError), "forbidden"):
                load_inputs(root, ("bad.parquet",))


class PipelineTests(unittest.TestCase):
    def test_twelve_class_nested_pipeline_and_new_output_only(self):
        rng = np.random.RandomState(14)
        folds = np.repeat(["2026-07", "2026-08", "2026-09", ""], [96,96,96,24])
        y = np.r_[np.tile(np.repeat(np.arange(12), 8), 3), np.full(24, -1)]
        n = len(y)
        time = np.r_[np.arange(96)*150, 31*86400+np.arange(96)*150,
                     62*86400+np.arange(96)*150, 92*86400+np.arange(24)*150].astype(np.int64)
        entities = pd.DataFrame({"txn_id": [f"TX{i}" for i in range(n)], "t": time,
            "sender_id": [f"AC{i%23}" for i in range(n)],
            "receiver_id": [f"AC{(i+1)%23}" if i%3 else f"MR{i%7}" for i in range(n)],
            "device_id": [f"DV{i%19}" for i in range(n)]})
        X = pd.DataFrame({"amount": rng.uniform(1,1000,n).astype("float32"),
                          "signal": (np.maximum(y, 0)+rng.normal(0,.5,n)).astype("float32")})
        meta = pd.DataFrame({"txn_id": entities.txn_id})
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "new_run"
            scores = run_stage2(X, y, meta, entities, folds, out, stage1_trees=2,
                stage2_trees=2, legit_fraction=1, threads=1, seed=13)
            self.assertEqual(set(scores), {"2026-07", "2026-08", "2026-09", "pooled"})
            p = np.load(out / "stage2_oof_proba.npy")
            self.assertEqual(p.shape, (288, 12))
            np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-6)
            self.assertEqual(np.load(out / "test_proba.npy").shape, (24, 12))
            self.assertEqual(len(list(out.glob("s1_*_proba.npy"))), 7)
            manifest = pd.read_parquet(out / "all_rows.parquet")
            np.testing.assert_array_equal(manifest.txn_id, entities.txn_id)
            np.testing.assert_array_equal(manifest.row, np.arange(n))
            provenance = json.loads((out / "fold_provenance.json").read_text())
            for outer in ["2026-07", "2026-08", "2026-09"]:
                for split in ["train", "validation"]:
                    for entry in provenance[outer][split]:
                        self.assertNotIn(outer, entry["fit_months"])
                        self.assertNotIn(entry["query_month"], entry["fit_months"])
                        self.assertNotIn(entry["source_month"], entry["fit_months"])
            with self.assertRaises(FileExistsError):
                run_stage2(X, y, meta, entities, folds, out, stage1_trees=2, stage2_trees=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
