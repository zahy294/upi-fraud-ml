"""Synthetic saved-artifact tests; no real datasets or estimator fitting."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare_saved_models as tool
from test_podium_predictions import context, zero_offsets
from make_submission import prepare_rule_context


class ComparisonTests(unittest.TestCase):
    def test_labels_are_id_aligned_and_reject_missing_duplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "labels.csv"
            pd.DataFrame({"txn_id": ["b", "a", "outside"],
                          "label": ["sim_swap", "legit", "mule_account"]}).to_csv(path, index=False)
            np.testing.assert_array_equal(tool.read_labels(path, ["a", "b"]), [0, 4])
            with self.assertRaises(ValueError):
                tool.read_labels(path, ["missing"])
            pd.DataFrame({"txn_id": ["a", "a"], "label": ["legit", "legit"]}).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                tool.read_labels(path, ["a"])

    def test_gate_keeps_original_on_ties_and_rejects_assessment_dates(self):
        meta, f3, f5 = context(6)
        ctx = prepare_rule_context(meta, f3, f5)
        p = np.full((6, 12), .1 / 11)
        p[:, 0] = .9
        with contextlib.redirect_stdout(io.StringIO()):
            chosen, trials = tool.select_policy({k: p for k in "ABC"}, np.zeros(6, dtype=int),
                np.full(6, "other"), ctx, {"steps": [.1], "max_passes": 1, "conditional": False})
        self.assertEqual(chosen["name"], "legacy")
        self.assertEqual(len(trials), 9)
        ctx["dates"][-1] = np.datetime64("2026-09-24")
        with self.assertRaises(ValueError):
            tool.select_policy({k: p for k in "ABC"}, np.zeros(6, dtype=int), np.full(6, "other"), ctx, {})

    def test_end_to_end_saved_comparison_and_test_export(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            profile = root / "run/ensemble"
            val, refit = profile / "validation", profile / "refit"
            val.mkdir(parents=True)
            refit.mkdir()
            labels = np.tile(np.arange(12), 3)
            meta, f3, f5 = context(36)
            meta["timestamp"] = np.repeat(["2026-09-16", "2026-09-20", "2026-09-24"], 12)
            good = np.full((36, 12), .2 / 11)
            good[np.arange(36), labels] = .8
            bad = np.full((36, 12), .02 / 11)
            bad[np.arange(36), (labels + 1) % 12] = .98
            config = {"members": [{"name": k, "weight": w} for k, w in zip("ABC", [.5, .3, .2])],
                      "calibration": {"conditional": False}}
            for folder, stage in ((val, "validation"), (refit, "refit")):
                provenance = {"stage": stage, "profile": "ensemble", "signature": "synthetic",
                              "fit_ids_hash": "fit", "sampled_fit_ids_hash": "sampled"}
                tool.write_json(folder / "manifest.json", {**provenance, "class_order": tool.LABELS, "config": config})
                stage_meta = meta.copy()
                if stage == "refit":
                    stage_meta["timestamp"] = "2026-10-01"
                    stage_meta["txn_id"] = "test_" + stage_meta.txn_id
                stage_meta.to_parquet(folder / "meta.parquet", index=False)
                f3.assign(txn_id=stage_meta.txn_id).to_parquet(folder / "features_v3.parquet", index=False)
                f5.assign(txn_id=stage_meta.txn_id).to_parquet(folder / "features_v5.parquet", index=False)
                np.save(folder / "groups.npy", np.full(36, "other"))
                for k, p in zip("ABC", (good, bad, bad)):
                    np.savez(folder / f"{k}_proba.npz", ids=stage_meta.txn_id.to_numpy(dtype=str),
                             proba=p, classes=tool.LABELS, provenance=json.dumps(provenance))
                original_name = "validation_proba.npz" if stage == "validation" else "test_proba.npz"
                np.savez(folder / original_name, ids=stage_meta.txn_id.to_numpy(dtype=str),
                         proba=.5 * good + .5 * bad, classes=tool.LABELS, provenance=json.dumps(provenance))
            tool.write_json(val / "offsets.json", zero_offsets())
            sample = pd.DataFrame({"id": stage_meta.txn_id.iloc[::-1].to_numpy(), "label": "legit"})
            sample.to_csv(refit / "submission.csv", index=False)
            label_path = root / "train.csv"
            pd.DataFrame({"txn_id": meta.txn_id, "label": np.asarray(tool.LABELS)[labels]}).to_csv(label_path, index=False)
            tool.write_json(profile.parent / "manifest.json", {"input_hashes": {"train.csv": tool.digest(label_path)}})
            output = root / "comparison"
            argv = ["--profile-dir", str(profile), "--labels-csv", str(label_path), "--output-dir", str(output)]
            observed_dates = []
            def fake_calibrate(p, y, g, ctx, **kwargs):
                observed_dates.extend(pd.DatetimeIndex(pd.to_datetime(ctx["dates"])).day)
                return zero_offsets()
            with patch.object(tool, "fit_offsets", side_effect=fake_calibrate), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(tool.main(argv), 0)
            self.assertTrue(all(16 <= day <= 23 for day in observed_dates))
            report = tool.read_json(output / "comparison.json")
            self.assertEqual(report["selected"], "A_broad")
            self.assertEqual(report["weights"], [1., 0., 0.])
            self.assertEqual(report["selected_assessment"]["macro_f1"], 1.)
            self.assertEqual(len(report["selected_assessment"]["per_class"]), 12)
            result = pd.read_csv(output / "submission.csv")
            self.assertEqual(result.id.tolist(), sample.id.tolist())
            self.assertEqual(result.label.tolist(), np.asarray(tool.LABELS)[labels[::-1]].tolist())
            self.assertTrue((output / "COMPARISON.txt").is_file())
            with self.assertRaises(FileExistsError):
                tool.main(argv)
            # Original label provenance is enforced before any tuning occurs.
            label_path.write_text("txn_id,label\nwrong,legit\n")
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                tool.main(argv[:-1] + [str(root / "other")])


if __name__ == "__main__":
    unittest.main()
