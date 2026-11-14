"""Synthetic v4 reproduction/selection checks; no estimator fitting."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import v4_master as v4


def artifacts(root):
    dates = np.repeat(["2026-07-01", "2026-08-01", "2026-09-01", "2026-09-16", "2026-09-24", "2026-10-01"], 12)
    y = np.tile(np.arange(12), 6)
    n = len(y)
    base = pd.DataFrame({"txn_id": [f"id{i}" for i in range(n)], "timestamp": dates,
        "receiver_type": "P2P", "split": ["train"]*(n-12)+["test"]*12,
        "label": np.array(v4.LABELS)[y]})
    processed = root / "data/processed"
    processed.mkdir(parents=True)
    (root / "data/raw").mkdir(parents=True)
    (root / "submissions").mkdir()
    base.to_parquet(processed / "base_v0.parquet", index=False)
    f3 = pd.DataFrame({"txn_id": base.txn_id, "band": 0})
    for k in range(1, 12):
        f3[f"snd_band_{k}_3h"] = 0
        f3[f"rcv_band_{k}_24h"] = 0
    f3.to_parquet(processed / "features_v3.parquet", index=False)
    pd.DataFrame({"txn_id": base.txn_id, "ch_band": 0}).to_parquet(processed / "features_v5.parquet", index=False)
    p = np.full((n, 12), .1/11)
    p[np.arange(n), y] = .9
    for member in v4.MEMBERS:
        folder = root / "models" / member
        folder.mkdir(parents=True)
        np.save(folder / "oof.npy", p[:-12].astype(np.float32))
        np.save(folder / "test_proba.npy", p[-12:].astype(np.float32))
    (root / "models/oof_raw").mkdir()
    np.save(root / "models/oof_raw/y_clean.npy", y[:-12])
    sub = pd.DataFrame({"id": base.txn_id[-12:].to_numpy(), "label": v4.LABELS}).iloc[::-1]
    sub.to_csv(root / "submissions/v4_master_ensemble.csv", index=False)
    sub.to_csv(root / "data/raw/sample_submission.csv", index=False)


class V4Tests(unittest.TestCase):
    def test_exact_five_member_blend_and_finite_weights(self):
        p = {k: np.full((2, 12), i/10.) for i, k in enumerate(v4.MEMBERS)}
        expected = sum(v4.MEMBERS[k]*p[k] for k in v4.MEMBERS)
        np.testing.assert_allclose(v4.blend(p, v4.MEMBERS), expected)
        with self.assertRaises(ValueError):
            v4.blend(p, {**v4.MEMBERS, "oof_clean": np.nan})

    def test_submission_compares_by_id_and_rejects_duplicates(self):
        frame = pd.DataFrame({"id": ["b", "a"], "label": ["legit", "mule_account"]})
        self.assertEqual(v4.compare_submission(frame, frame.iloc[::-1]), 0)
        with self.assertRaises(ValueError):
            v4.compare_submission(frame, pd.concat([frame.iloc[:1]]*2))

    def test_probability_checks_reject_nan_wrong_shape_and_zero_sum(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "p.npy"
            for p in [np.zeros((2, 12)), np.full((2, 12), np.nan), np.zeros((2, 11))]:
                np.save(file, p)
                with self.assertRaises(ValueError):
                    v4.load_probabilities(file, 2)

    def test_missing_inventory_performs_no_mutation(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            self.assertEqual(v4.main(["--artifact-root", temp, "--stage", "check"]), 2)
            self.assertEqual(list(root.iterdir()), [])

    def test_reproduce_then_tune_and_stop_on_wrong_baseline(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            artifacts(root)
            out = root / "result"
            args = ["--artifact-root", temp, "--output-dir", str(out), "--stage", "tune"]
            self.assertEqual(v4.main(args), 0)
            report = json.loads((out / "reproduction.json").read_text())
            self.assertEqual(report["label_differences_vs_submitted"], 0)
            tuning = json.loads((out / "tuning.json").read_text())
            self.assertEqual(tuning["selected"], "fixed_clean")
            self.assertEqual(tuning["assessment"]["macro_f1"], 1.)
            self.assertEqual(len(tuning["assessment"]["per_class"]), 12)
            gold = root / "submissions/v4_master_ensemble.csv"
            frame = pd.read_csv(gold)
            frame.loc[0, "label"] = "legit"
            frame.to_csv(gold, index=False)
            out2 = root / "wrong"
            args[3] = str(out2)
            self.assertEqual(v4.main(args), 2)
            self.assertFalse((out2 / "tuning.json").exists())

    def test_assessment_labels_do_not_select_policy(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            artifacts(root)
            base, mask, raw, clean, context, p, test = v4.load_inputs(root)
            dates = base.loc[mask, "timestamp"]
            context = v4.slice_rule_context(context, mask)
            first = v4.tune(p, raw, clean, dates, context)
            changed = raw.copy()
            changed[pd.to_datetime(dates).dt.day.to_numpy() >= 24] = 0
            second = v4.tune(p, changed, clean, dates, context)
            self.assertEqual(first["selected"], second["selected"])
            self.assertEqual(first["trials"], second["trials"])
            self.assertEqual(first["decision_weights"], second["decision_weights"])
            for array in list(p.values()) + list(test.values()):
                array._mmap.close()

    def test_reordered_source_features_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifacts(root)
            path = root / "data/processed/features_v3.parquet"
            pd.read_parquet(path).iloc[::-1].to_parquet(path, index=False)
            with self.assertRaisesRegex(ValueError, "positional feature order"):
                v4.load_inputs(root)


if __name__ == "__main__":
    unittest.main()
