"""Synthetic decision-calibration tests: no estimator fitting or real data."""
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
from src.upi.boundary_calibration import coordinate, fit_boundary_offsets, rule_map
from src.upi.make_submission import prepare_rule_context, predict_labels, _score
from test_podium_predictions import context, zero_offsets
import compare_diverse_models as tool


def fixture(n=48):
    meta, f3, f5 = context(n)
    meta["timestamp"] = np.resize(["2026-09-16", "2026-09-20"], n)
    return prepare_rule_context(meta, f3, f5)


class BoundaryTests(unittest.TestCase):
    def test_coordinate_matches_brute_force_with_rules_and_ties(self):
        rng = np.random.default_rng(18)
        n = 72
        p = rng.dirichlet(np.ones(12), n)
        p[1:4] = p[0]  # equal margins must move as one group
        logp, y = np.log(p), rng.integers(0, 12, n)
        ctx = fixture(n)
        ctx["band"][::7] = 3
        ctx["chain_band"][1::9] = 5
        ctx["sender_band"][2::5] = 4
        mapping = rule_map(ctx, n)
        offsets = rng.uniform(-.2, .2, 12)
        offsets[0] = 0
        for k in (1, 5, 11):
            bound, penalty = 3., .002
            value, objective = coordinate(logp, offsets, k, y, mapping, bound, penalty)
            scores = logp + offsets
            scores[:, k] = -np.inf
            margin = scores.max(axis=1) - logp[:, k]
            edges = np.unique(np.r_[-bound, margin[(margin > -bound) & (margin < bound)], bound])
            candidates = np.r_[(edges[1:] + edges[:-1])/2, offsets[k], -bound, bound, 0.]
            results = []
            for candidate in candidates:
                trial = offsets.copy()
                trial[k] = candidate
                pred = mapping[np.arange(n), (logp + trial).argmax(axis=1)]
                results.append(_score(y, pred) - penalty * np.mean(trial[1:]**2))
            self.assertAlmostEqual(objective, max(results), places=12)
            self.assertTrue(-bound <= value <= bound)

    def test_rejects_assessment_dates_and_invalid_config(self):
        ctx = fixture()
        p = np.full((48, 12), 1/12)
        y = np.zeros(48, dtype=int)
        for kwargs in ({"penalty": -1}, {"bound": 7}, {"passes": 0}):
            with self.assertRaises(ValueError):
                fit_boundary_offsets(p, y, ctx, **kwargs)
        ctx["dates"][0] = np.datetime64("2026-09-24")
        with self.assertRaises(ValueError):
            fit_boundary_offsets(p, y, ctx)

    def test_permutation_invariance_and_fixed_bands(self):
        rng = np.random.default_rng(9)
        p = rng.dirichlet(np.ones(12), 48)
        y = np.tile(np.arange(12), 4)
        ctx = fixture()
        ctx["band"][::4] = 7
        a = fit_boundary_offsets(p, y, ctx)
        reverse = {k: v[::-1] if isinstance(v, np.ndarray) else v for k, v in ctx.items()}
        b = fit_boundary_offsets(p[::-1], y[::-1], reverse)
        np.testing.assert_allclose(a["global"], b["global"])
        pred = predict_labels(p, np.full(48, "other"), a, ctx)
        np.testing.assert_array_equal(pred[::4], 7)
        objectives = [item["objective"] for item in a["trace"]]
        self.assertTrue(np.all(np.diff(objectives) >= -1e-12))

    def test_recovers_missed_fraud_without_estimators(self):
        y = np.tile(np.arange(12), 4)
        p = np.full((48, 12), .001)
        p[:, 0] = .7
        p[np.arange(48), y] = .3
        p[y == 0, 0] = .9
        p /= p.sum(axis=1, keepdims=True)
        ctx = fixture()
        offsets = fit_boundary_offsets(p, y, ctx, penalty=0)
        pred = predict_labels(p, np.full(48, "other"), offsets, ctx)
        np.testing.assert_array_equal(pred, y)

    def test_selection_retains_reference_on_ties_and_rejects_assessment(self):
        ctx, p = fixture(), np.full((48, 12), 1/12)
        members = {k: p for k in ("reference", "forest", "hierarchy")}
        with patch.object(tool, "fit_boundary_offsets", return_value=zero_offsets()), contextlib.redirect_stdout(io.StringIO()):
            selected, trials = tool.select(members, np.zeros(48, dtype=int), np.full(48, "other"), ctx, zero_offsets())
        self.assertEqual(selected["name"], "reference")
        self.assertEqual(len(trials), 9)
        ctx["dates"][0] = np.datetime64("2026-09-24")
        with self.assertRaises(ValueError):
            tool.select(members, np.zeros(48, dtype=int), np.full(48, "other"), ctx, zero_offsets())

    def test_extra_artifact_rejects_changed_bytes_and_wrong_fit_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            cache = run / "validation/cache"
            cache.mkdir(parents=True)
            path = run / "validation/p.npz"
            provenance = {"signature": "same", "stage": "validation", "fit_ids_hash": "fit"}
            np.savez(path, proba=np.full((2,12), 1/12), ids=["b", "a"], classes=tool.LABELS,
                     provenance=json.dumps(provenance))
            (cache / "rows.parquet").write_bytes(b"hash-only fixture")
            tool.write_json(cache / "ready.json", {"hashes": {"rows.parquet": tool.digest(cache / "rows.parquet")}})
            tool.write_json(run / "manifest.json", {"signature": "same"})
            tool.write_json(run / "validation/complete.json", {"signature": "same", "hashes": {"p.npz": tool.digest(path)}})
            self.assertEqual(tool.load_extra(run, "p.npz", ["a", "b"], cache, "fit").shape, (2,12))
            with self.assertRaises(ValueError):
                tool.load_extra(run, "p.npz", ["a", "b"], cache, "wrong")
            with path.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaises(ValueError):
                tool.load_extra(run, "p.npz", ["a", "b"], cache, "fit")


if __name__ == "__main__":
    unittest.main()
