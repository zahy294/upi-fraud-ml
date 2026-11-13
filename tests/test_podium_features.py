"""Tiny synthetic checks only: no model training and no real data access."""
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/upi"))
import numpy as np
import pandas as pd
from common import LABELS
from events import Events
from features_v6 import (align_pack, build_label_free_v6, build_temporal_te,
                         assemble_features, apply_feature_schema)
from clean_labels import clean_training_labels
from bands import build_bands, legit_block_start, segment, band_scores
from build_base import build_tables


def base(n=6):
    return pd.DataFrame({
        "txn_id": [f"x{i}" for i in range(n)], "t": np.arange(n, dtype=np.int64) * 100,
        "split": ["train"] * n, "month": ["2026-07"] * n,
        "label": ["legit"] * n, "sender_id": ["AC_1"] * n,
        "receiver_id": ["AC_2"] * n, "receiver_type": ["P2P"] * n,
        "device_id": ["DV_1"] * n, "txn_type": ["PAY"] * n,
        "amount": np.arange(n, dtype=float) + 1, "balance_before": [100.] * n,
        "new_payee": [1] * n, "hour": [0] * n, "merchant_category": ["food"] * n,
    })


def packs(b):
    return {"v4": pd.DataFrame({"txn_id": b.txn_id,
                                "bait_prev_dt_h": np.full(len(b), 1.),
                                "pair_tiny_prev_dt_h": np.full(len(b), 2.),
                                "bait_cnt_before_1d": np.full(len(b), 3.)})}


def teacher(n, k=5):
    p = np.full((n, 12), 0.01 / 11)
    p[:, k] = 0.99
    return p


def cleaning(b, p, ids=None, band=None):
    bands = pd.DataFrame({"txn_id": b.txn_id, "band": np.zeros(len(b)) if band is None else band})
    sessions = pd.DataFrame({"txn_id": b.txn_id, "S_flag": np.zeros(len(b))})
    return clean_training_labels(b, p, bands, sessions,
                                 pd.Index(b.txn_id if ids is None else ids))


class EventTests(unittest.TestCase):
    def test_closed_strict_duplicate_and_missing_groups(self):
        e = Events(np.array([1, 1, 1, 2]), np.array([10, 20, 20, 30]), np.array([1., 2., 3., 4.]))
        np.testing.assert_array_equal(e.count(np.array([1, 2, -1]), np.array([20, 20, 20]), -10, 0), [3, 0, 0])
        self.assertEqual(e.sum(np.array([1]), np.array([20]), -10, 0)[0], 6)
        self.assertEqual(e.prev_dt(np.array([1]), np.array([20]), 100)[0], 10)
        self.assertTrue(np.isnan(e.next_dt(np.array([1]), np.array([20]), 100)[0]))
        self.assertTrue(np.isnan(e.previous_gap(np.array([2]), np.array([20]))[0]))

    def test_empty_and_input_errors(self):
        e = Events(np.array([], dtype=int), np.array([], dtype=int), np.array([]))
        self.assertEqual(e.sum(np.array([1]), np.array([0]), -1, 1)[0], 0)
        self.assertTrue(np.isnan(e.prev_dt(np.array([1]), np.array([0]), 10)[0]))
        self.assertEqual(len(e.next_gap(np.array([], dtype=int), np.array([], dtype=int))), 0)
        with self.assertRaises(ValueError):
            Events([1], [0]).sum([1], [0], -1, 1)
        with self.assertRaises(ValueError):
            Events([1, 2], [0])
        with self.assertRaises(ValueError):
            Events([1], [0]).count([1], [0], 2, 1)

    def test_large_groups_and_windows(self):
        e = Events(np.array([1, 2**62]), np.array([-100, 100]), values=[2., 4.])
        self.assertEqual(e.count(np.array([1]), np.array([0]), -10**12, 10**12)[0], 1)
        self.assertEqual(e.next_value(np.array([2**62]), np.array([0]), 1000)[0], 4)


class FeatureTests(unittest.TestCase):
    def test_alignment_reorders_and_rejects(self):
        b = base()
        aligned = align_pack(b, packs(b)["v4"].iloc[::-1])
        self.assertEqual(aligned.txn_id.tolist(), b.txn_id.tolist())
        with self.assertRaises(ValueError):
            align_pack(b, packs(b)["v4"].iloc[:-1])

    def test_label_free_invariant_and_ties(self):
        b = base()
        b["t"] = [0, 0, 600, 700, 86400, 86401]
        b["sender_id"] = ["A", "B", "A", "A", "A", "A"]
        f = build_label_free_v6(b, packs(b))
        changed = b.assign(label="invalid held-out label")
        pd.testing.assert_frame_equal(f, build_label_free_v6(changed, packs(b)))
        np.testing.assert_array_equal(f.v6_dev_n_10m, [0, 0, 2, 1, 0, 1])
        np.testing.assert_array_equal(f.v6_dev_unique_senders_24h, [0, 0, 2, 2, 2, 1])
        self.assertEqual(f.shape, (6, 26))
        self.assertTrue(all(dtype == np.float32 for dtype in f.drop(columns="txn_id").dtypes))

    def test_missing_bait_and_receiver_namespace(self):
        b = base(2)
        b["receiver_type"] = ["P2P", "MERCHANT"]
        p = packs(b)
        p["v4"].loc[1, "bait_prev_dt_h"] = np.nan
        f = build_label_free_v6(b, p)
        self.assertEqual(f.v6_rcv_n_24h.iloc[1], 0)
        self.assertTrue(np.isnan(f.v6_bait_delay_fraction.iloc[1]))
        self.assertEqual(f.v6_bait_prev_missing.iloc[1], 1)

    def test_rolling_matches_bruteforce(self):
        b = base(40)
        rng = np.random.default_rng(2)
        b["t"] = rng.integers(0, 100000, len(b), dtype=np.int64)
        b["device_id"] = rng.choice(["DV_1", "DV_2"], len(b))
        b["sender_id"] = rng.choice(["AC_1", "AC_2", "AC_3"], len(b))
        f = build_label_free_v6(b, packs(b))
        for i, row in b.iterrows():
            mask = (b.device_id == row.device_id) & (b.t >= row.t - 86400) & (b.t < row.t)
            self.assertEqual(f.v6_dev_unique_senders_24h.iloc[i], b.loc[mask].sender_id.nunique())

    def test_temporal_encoding_excludes_current_and_future(self):
        b = base(4)
        b["month"] = ["2026-07", "2026-08", "2026-09", "2026-10"]
        b["label"] = ["mule_account", "legit", "invalid", "invalid"]
        q = b.iloc[:2].drop(columns="label")
        f = build_temporal_te(q, b)
        self.assertAlmostEqual(f.v6_te_snd_1.iloc[0], 1 / 12)
        self.assertAlmostEqual(f.v6_te_snd_1.iloc[1], (1 + 50 * 2 / 13) / 51, places=7)
        changed = b.copy()
        changed.loc[2:, "label"] = "qr_swap"
        pd.testing.assert_frame_equal(f, build_temporal_te(q, changed))

    def test_frozen_and_null_entities(self):
        b = base(2)
        b["month"] = ["2026-07", "2026-08"]
        b.loc[0, "sender_id"] = None
        f = build_temporal_te(b.iloc[1:], b.iloc[:1], expanding_months=False)
        self.assertEqual(f.v6_te_snd_log_support.iloc[0], 0)
        with self.assertRaises(ValueError):
            build_temporal_te(b.iloc[:1], b.iloc[1:], expanding_months=False)

    def test_assembly_schema_traps_and_conflicts(self):
        b = base(2).assign(ip_risk_score=0.5, app_old=1, app_major="3")
        f, schema = assemble_features(b, {"v4": packs(b)["v4"].iloc[::-1]})
        self.assertNotIn("label", f)
        self.assertNotIn("txn_id", f)
        self.assertNotIn("ip_risk_score", f)
        self.assertNotIn("app_old", f)
        encoded = apply_feature_schema(b.assign(txn_type="UNKNOWN").join(
            packs(b)["v4"].drop(columns="txn_id")), schema)
        self.assertTrue((encoded.txn_type == -1).all())
        with self.assertRaises(ValueError):
            assemble_features(b, {"bad": b[["txn_id", "amount"]].assign(amount=999)})

    def test_base_preserves_context_and_ratio_ids(self):
        b = base(3).iloc[::-1]
        rel = b[["txn_id"]].assign(context=4.)
        built, ratios = build_tables(b, rel)
        self.assertEqual(len(built), len(b))
        self.assertEqual(ratios.index.tolist(), built.txn_id.tolist())
        self.assertEqual(built.txn_id.tolist(), ["x0", "x1", "x2"])


class CleaningTests(unittest.TestCase):
    def test_reverse_pair_votes_preserve_event_type(self):
        b = base(2)
        b["sender_id"] = ["AC_1", "AC_2"]
        b["receiver_id"] = ["AC_2", "AC_1"]
        b["device_id"] = ["DV_1", "DV_2"]
        b["label"] = ["sim_swap", "legit"]
        result = cleaning(b, teacher(2, 5))
        self.assertEqual(result.y_clean.iloc[1], 4)
        b.loc[0, "receiver_type"] = "MERCHANT"
        result = cleaning(b, teacher(2, 5))
        self.assertEqual(result.y_clean.iloc[1], 5)

    def test_sibling_label_preserved(self):
        b = base(2).assign(label="sim_swap")
        result = cleaning(b, teacher(2, 5))
        self.assertEqual(result.y_clean.tolist(), [4, 4])

    def test_event_type_and_unreported(self):
        b = base(3)
        b["label"] = ["sim_swap", "account_takeover", "legit"]
        p = teacher(3, 5)
        # One accepted neighbour is enough, with additive sender/device votes.
        result = cleaning(b, p, ids=["x0", "x2"])
        self.assertEqual(result.y_clean.tolist(), [4, 4])
        self.assertEqual(result.reason.tolist(), ["unchanged", "unreported"])
        alone = cleaning(b, p, ids=["x2"])
        self.assertEqual(alone.y_clean.tolist(), [5])

    def test_heldout_label_perturbation_and_missing_teacher(self):
        b = base(3).assign(label="sim_swap")
        p = teacher(3, 5)
        before = cleaning(b, p, ids=["x0", "x1"])
        b.loc[2, "label"] = "invalid"
        pd.testing.assert_frame_equal(before, cleaning(b, p, ids=["x0", "x1"]))
        b.loc[2, "label"] = "sim_swap"
        p[:] = np.nan
        after = cleaning(b, p, band=[0, 5, 0])
        self.assertEqual(after.y_clean.tolist(), [4, 5, 4])
        self.assertFalse(after.teacher_available.any())
        self.assertEqual(len(cleaning(b, p, ids=[])), 0)

    def test_band_override_and_probability_guard(self):
        b = base(2)
        result = cleaning(b, teacher(2), band=[4, 0])
        self.assertEqual(result.y_clean.iloc[0], 4)
        p = teacher(2)
        p[0, 1] = np.nan
        with self.assertRaises(ValueError):
            cleaning(b, p)


class BandTests(unittest.TestCase):
    def test_separate_test_block_and_label_invariance(self):
        # Shrink only the block-run detector to two ordinary devices for this
        # <=50-row fixture; production retains the 40-device requirement.
        b = base(26)
        train_devices = [423001 + 3 * k for k in range(11)] + [423040, 423041]
        test_devices = [423042 + 3 * k for k in range(11)] + [423081, 423082]
        b["device_id"] = [f"D_{i}" for i in train_devices + test_devices]
        b.loc[13:, "split"] = "test"
        b.loc[:10, "label"] = LABELS[1:]
        b.loc[13:, "label"] = "unread test label"
        p = np.vstack([teacher(1, k) for k in list(range(1, 12)) + [0, 0]] * 2)
        kinds = [0] * 26
        kinds[3] = kinds[16] = 1
        f1 = pd.DataFrame({"txn_id": b.txn_id, "snd_drain_kind": kinds})
        detector = lambda nums, lo: legit_block_start(nums, lo, run=2)
        with patch("bands.legit_block_start", side_effect=detector):
            f, d = build_bands(b, p, f1, pd.Index(b.txn_id.iloc[:13]))
            changed = b.copy()
            changed.loc[13:, "label"] = "another unread test label"
            again, diagnostic = build_bands(changed, p, f1, pd.Index(b.txn_id.iloc[:13]))
        pd.testing.assert_frame_equal(f, again)
        self.assertEqual(d, diagnostic)
        self.assertEqual(d["test"][0], 423042)
        self.assertEqual(f.band.iloc[13], 1)
        self.assertEqual(f.band.iloc[16], 4)

    def test_block_guards_and_segmentation(self):
        with self.assertRaises(ValueError):
            legit_block_start(np.array([1, 3, 4]), 1, run=3)
        with self.assertRaises(ValueError):
            segment(np.full((2, 11), np.nan))
        self.assertEqual(legit_block_start(np.array([1, 3, 4, 5]), 1, run=3), 3)
        with self.assertRaises(ValueError):
            band_scores(np.empty((0, 12)), [], np.array([], dtype=int), 1, 10, .1, .1)

    def test_band_fit_ignores_heldout_labels(self):
        # 42 rows: two fraud anchors plus a 40-device ordinary block.
        b = base(42)
        b["device_id"] = ["DV_423001", "DV_423005"] + [f"DV_{i}" for i in range(423020, 423060)]
        b.loc[0, "label"] = "mule_account"
        b.loc[1, "label"] = "sim_swap"
        p = teacher(42, 1)
        p[1] = teacher(1, 4)[0]
        f1 = pd.DataFrame({"txn_id": b.txn_id, "snd_drain_kind": [0, 1] + [0] * 40})
        f, d = build_bands(b, p, f1.iloc[::-1], pd.Index(["x0", "x1"]))
        b.loc[2:, "label"] = "invalid"
        f2, d2 = build_bands(b, p, f1, pd.Index(["x0", "x1"]))
        pd.testing.assert_frame_equal(f, f2)
        self.assertEqual(d, d2)
        self.assertEqual(f.band.iloc[0], 1)
        self.assertEqual(f.band.iloc[1], 4)
        self.assertEqual(d["unique_mule_anchors"], 1)
        self.assertEqual(d["unique_sim_anchors"], 1)


if __name__ == "__main__":
    unittest.main()
