"""Decision tests with synthetic probabilities; no model training."""
import json
import sys
import unittest
import contextlib
import io
from unittest.mock import patch
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/upi"))
from make_submission import predict_labels, fit_offsets, prepare_rule_context, select_member_policy
from test_podium_predictions import context, zero_offsets


class GateTests(unittest.TestCase):
    def test_member_selection_can_choose_and_serialize_gate(self):
        meta, f3, f5 = context(6)
        ctx = prepare_rule_context(meta, f3, f5)
        p = np.zeros((6, 12))
        p[:, :3] = [.45, .30, .25]
        def fixed_offsets(*args, config):
            result = zero_offsets()
            if "fraud_gate_log_odds" in config:
                result["fraud_gate_log_odds"] = config["fraud_gate_log_odds"]
            return result
        with patch("make_submission.fit_offsets", side_effect=fixed_offsets), contextlib.redirect_stdout(io.StringIO()):
            offsets, chosen = select_member_policy({"A": p}, {"A": 1.}, np.ones(6, dtype=int),
                np.full(6, "other"), ctx, config={"fraud_gate_thresholds": [-.5, 0.]})
        self.assertEqual(chosen["selected"], "baseline_fraud_gate_-0.5")
        self.assertEqual(offsets["fraud_gate_log_odds"], -.5)

    def test_combined_fraud_mass_and_threshold(self):
        meta, f3, f5 = context(2)
        ctx = prepare_rule_context(meta, f3, f5)
        p = np.zeros((2, 12))
        p[:, :3] = [.45, .30, .25]
        groups = np.full(2, "other")
        zero = zero_offsets()
        np.testing.assert_array_equal(predict_labels(p, groups, zero, ctx), [0, 0])
        gate = {**zero, "fraud_gate_log_odds": 0.}
        np.testing.assert_array_equal(predict_labels(p, groups, gate, ctx), [1, 1])
        gate["fraud_gate_log_odds"] = 1.
        np.testing.assert_array_equal(predict_labels(p, groups, gate, ctx), [0, 0])
        gate["fraud_gate_log_odds"] = float("nan")
        with self.assertRaises(ValueError):
            predict_labels(p, groups, gate, ctx)

    def test_gate_does_not_override_direct_band(self):
        meta, f3, f5 = context(2)
        f3.loc[0, "band"] = 11
        p = np.zeros((2, 12))
        p[:, 0] = 1.
        out = predict_labels(p, np.full(2, "other"), {**zero_offsets(), "fraud_gate_log_odds": 2.},
                             prepare_rule_context(meta, f3, f5))
        np.testing.assert_array_equal(out, [11, 0])

    def test_calibration_persists_gate_in_all_decision_artifacts(self):
        meta, f3, f5 = context(6)
        ctx = prepare_rule_context(meta, f3, f5)
        p = np.zeros((6, 12))
        p[:, :3] = [.45, .30, .25]
        offsets = fit_offsets(p, np.ones(6, dtype=int), np.full(6, "other"), ctx,
                              config={"steps": [.1], "max_passes": 1, "fraud_gate_log_odds": 0.})
        restored = json.loads(json.dumps(offsets, allow_nan=False))
        self.assertEqual(restored["fraud_gate_log_odds"], 0.)
        self.assertEqual(restored["global_only_offsets"]["fraud_gate_log_odds"], 0.)
        np.testing.assert_array_equal(predict_labels(p, np.full(6, "other"), restored, ctx), np.ones(6, dtype=int))


if __name__ == "__main__":
    unittest.main()
