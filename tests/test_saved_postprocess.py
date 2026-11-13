"""Zero-training candidate tests, using tiny synthetic artifacts only."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import postprocess_saved as tool
from test_podium_predictions import context, zero_offsets
from make_submission import prepare_rule_context, predict_labels


class SavedTests(unittest.TestCase):
    def test_disagreement_directions_are_not_interpreted_as_ground_truth(self):
        a = pd.DataFrame({"id": ["a", "b", "c", "d"],
                          "label": ["legit", "mule_account", "phishing_collect", "legit"]})
        b = a.copy()
        b["label"] = ["sim_swap", "legit", "fake_cashback", "legit"]
        report = tool.compare_labels(a, b)
        self.assertEqual(report["changed_labels"], 3)
        self.assertEqual(report["changed_fraud_status"], 2)
        self.assertEqual(report["baseline_only_fraud"], 1)
        self.assertEqual(report["current_only_fraud"], 1)
        self.assertEqual(report["changed_fraud_type"], 1)
        with self.assertRaises(ValueError):
            tool.compare_labels(a, b.iloc[::-1])

    def test_align_class_and_transaction_orders(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "friend.npz"
            p = np.eye(12)[[11, 0, 4]]
            np.savez(path, ids=["a", "b", "c"], proba=p[:, ::-1], classes=np.asarray(tool.LABELS)[::-1])
            np.testing.assert_array_equal(tool.load_proba(path, ["c", "a", "b"]), p[[2, 0, 1]])
            with self.assertRaises(ValueError):
                tool.load_proba(path, ["a", "b", "unknown"])
            with self.assertRaises(ValueError):
                tool.load_proba(path, ["a", "a", "c"])
            pd.DataFrame({"id": ["a"], "label": ["legit"]}).to_csv(path.with_suffix(".csv"), index=False)
            with self.assertRaises(ValueError):
                tool.load_proba(path.with_suffix(".csv"), ["a"])

    def test_exact_k_protects_bands_and_is_order_independent(self):
        meta, f3, f5 = context(6)
        f3.loc[0, "band"] = 11
        ctx = prepare_rule_context(meta, f3, f5)
        p = np.full((6, 12), .1 / 11)
        p[:, 0] = .9
        groups = np.full(6, "other")
        pred = tool.exact_count_prediction(p, groups, zero_offsets(), ctx, 3)
        self.assertEqual((pred > 0).sum(), 3)
        self.assertEqual(pred[0], 11)
        rev = {k: v[::-1] if isinstance(v, np.ndarray) else v for k, v in ctx.items()}
        np.testing.assert_array_equal(pred, tool.exact_count_prediction(p[::-1], groups[::-1], zero_offsets(), rev, 3)[::-1])
        for k in (0, -1, 7):
            with self.assertRaises(ValueError):
                tool.exact_count_prediction(p, groups, zero_offsets(), ctx, k)
        self.assertEqual((tool.exact_count_prediction(p, groups, zero_offsets(), ctx, 6) > 0).sum(), 6)

    def test_cli_blends_and_preserves_existing_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            profile = root / "ensemble"
            refit, val = profile / "refit", profile / "validation"
            refit.mkdir(parents=True)
            val.mkdir()
            meta, f3, f5 = context(6)
            f3.loc[0, "band"] = 4
            meta.to_parquet(refit / "meta.parquet", index=False)
            f3.to_parquet(refit / "features_v3.parquet", index=False)
            f5.to_parquet(refit / "features_v5.parquet", index=False)
            groups = np.full(6, "other")
            np.save(refit / "groups.npy", groups)
            p = np.full((6, 12), .1 / 11)
            p[:, 0] = .9
            friend = np.full((6, 12), .1 / 11)
            friend[:, 11] = .9
            np.savez(refit / "test_proba.npz", ids=meta.txn_id.to_numpy(dtype=str), proba=p, classes=tool.LABELS)
            friend_path = root / "friend.npz"
            np.savez(friend_path, ids=meta.txn_id.to_numpy(dtype=str)[::-1], proba=friend[::-1], classes=tool.LABELS)
            offsets = zero_offsets()
            (val / "offsets.json").write_text(json.dumps(offsets))
            current = pd.DataFrame({"id": meta.txn_id[::-1], "label": "legit"})
            current.to_csv(refit / "submission.csv", index=False)
            original_hash = tool.digest(refit / "submission.csv")
            out = root / "blend.csv"
            argv = ["--profile-dir", str(profile), "--friend-proba", str(friend_path), "--output", str(out)]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(tool.main(argv), 0)
            expected = predict_labels(.65 * friend + .35 * p, groups, offsets, prepare_rule_context(meta, f3, f5))
            result = pd.read_csv(out)
            self.assertEqual(result.id.tolist(), current.id.tolist())
            self.assertEqual(result.label.tolist(), np.asarray(tool.LABELS)[expected[::-1]].tolist())
            self.assertEqual(tool.digest(refit / "submission.csv"), original_hash)
            self.assertEqual(json.loads(out.with_suffix(".json").read_text())["candidate"]["rows"], 6)
            with self.assertRaises(FileExistsError):
                tool.main(argv)
            baseline = root / "baseline.csv"
            current.loc[current.index[:2], "label"] = "merchant_collusion"
            current.to_csv(baseline, index=False)
            out2 = root / "matched.csv"
            with contextlib.redirect_stdout(io.StringIO()):
                tool.main(["--profile-dir", str(profile), "--baseline-csv", str(baseline),
                           "--match-baseline-count", "--output", str(out2)])
            self.assertEqual(pd.read_csv(out2).label.ne("legit").sum(), 2)
            out3 = root / "hybrid.csv"
            with contextlib.redirect_stdout(io.StringIO()):
                tool.main(["--profile-dir", str(profile), "--baseline-csv", str(baseline),
                           "--top-k", "6", "--use-baseline-types", "--output", str(out3)])
            hybrid = pd.read_csv(out3).set_index("id")
            for txn in current.loc[current.label.ne("legit"), "id"]:
                self.assertEqual(hybrid.loc[txn, "label"], "merchant_collusion")
            self.assertEqual(hybrid.label.ne("legit").sum(), 6)
            self.assertEqual(hybrid.loc["t0", "label"], "sim_swap")


if __name__ == "__main__":
    unittest.main()
