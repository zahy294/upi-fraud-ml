"""Build v5 master submissions combining Stage-1 (LGB, XGB, CAT) and Stage-2 Event-Propagation (LGB, XGB, CAT)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/upi"))
import numpy as np
import pandas as pd
from common import LABELS, fit_weights, macro_f1, per_class
from events import DAY, HOUR, Events

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"

base = pd.read_parquet(
    P / "base_v0.parquet",
    columns=["txn_id", "t", "label", "split", "sender_id", "receiver_id", "device_id", "receiver_type"],
)
tr = (base.split == "train").values
y_raw = base.label[tr].map({lab: i for i, lab in enumerate(LABELS)}).values.astype(int)
y_clean = np.load(ROOT / "models/oof_raw/y_clean.npy")

F2 = pd.read_parquet(P / "features_v2.parquet", columns=["S_flag"])
F3 = pd.read_parquet(P / "features_v3.parquet")
F5 = pd.read_parquet(P / "features_v5.parquet", columns=["ch_band", "ch_len"])
F_s2 = pd.read_parquet(
    P / "features_stage2.parquet",
    columns=["s2_root_in_band_1", "s2_root_in_band_2", "s2_root_in_band_3"],
)

band = F3.band.values.astype(int)
p2p_all = (base.receiver_type == "P2P").values

# Precompute opposite-direction 7d band for bait <-> payback (phishing_collect=6, fake_cashback=9)
t = base.t.values.astype(np.int64)
snd = base.sender_id.str[2:].astype(np.int64).values
rnum = base.receiver_id.str[2:].astype(np.int64).values
rcv = np.where(p2p_all, rnum, rnum + 1_000_000)
pair = snd * 2_000_000 + rcv
pcode, puniq = pd.factorize(pair)
rcode = pd.Index(puniq).get_indexer(np.where(p2p_all, rnum * 2_000_000 + snd, -1))

V_opp = np.zeros((len(base), 12))
for k in (6, 9):
    ind = (band == k).astype(float)
    V_opp[:, k] = Events(pcode, t, ind).sum(rcode, t, -7 * DAY, 7 * DAY)
opp_band = np.where((V_opp[:, 1:] > 0).sum(1) == 1, V_opp[:, 1:].argmax(1) + 1, 0)

# Root incoming P2P band for forwarding chains
R_in = np.stack([F_s2[f"s2_root_in_band_{k}"].values for k in (1, 2, 3)], axis=1)
root_in_band = np.where((R_in > 0).sum(1) == 1, R_in.argmax(1) + 1, 0)


def enhanced_band_rules(pred, rows):
    b = band[rows]
    S = np.stack([F3[f"snd_band_{k}_3h"].values[rows] for k in range(1, 12)], 1)
    R = np.stack([F3[f"rcv_band_{k}_24h"].values[rows] for k in range(1, 12)], 1)
    sb = np.where((S > 0).sum(1) == 1, S.argmax(1) + 1, 0)
    rb = np.where((R > 0).sum(1) == 1, R.argmax(1) + 1, 0)
    cb = F5.ch_band.values[rows].astype(int)
    ch_len = F5.ch_len.values[rows]
    ob = opp_band[rows]
    rib = root_in_band[rows]
    p2p = p2p_all[rows]

    out = np.where(b > 0, b, pred)
    free = (b == 0) & (pred > 0)
    drain = free & np.isin(pred, [4, 5]) & np.isin(sb, [4, 5])
    fanin = free & np.isin(pred, [1, 2, 3]) & p2p & np.isin(rb, [1, 2, 3])
    shop = free & np.isin(pred, [7, 8, 11]) & ~p2p & np.isin(rb, [7, 8, 11])
    bait_opp = free & np.isin(pred, [6, 9]) & np.isin(ob, [6, 9])
    chain_in = free & (cb == 0) & (ch_len >= 2) & np.isin(pred, [1, 2, 3]) & np.isin(rib, [1, 2])
    chain = free & (cb > 0)

    out = np.where(drain, sb, out)
    out = np.where(fanin | shop, rb, out)
    out = np.where(bait_opp, ob, out)
    out = np.where(chain_in, rib, out)
    out = np.where(chain, cb, out)
    return out, int((out != pred).sum())


def load_tag(tag):
    oof = np.load(ROOT / "models" / tag / "oof.npy").astype(np.float64)
    pte = np.load(ROOT / "models" / tag / "test_proba.npy").astype(np.float64)
    return oof, pte


oof_s1_lgb, pte_s1_lgb = load_tag("oof_clean")
oof_v6_lgb, pte_v6_lgb = load_tag("oof_v6_lgb")
oof_s1_xgb, pte_s1_xgb = load_tag("oof_clean_xgb")
oof_v6_xgb, pte_v6_xgb = load_tag("oof_v6_xgb")
oof_s1_cat, pte_s1_cat = load_tag("oof_clean_cat")

oof_s1 = 0.35 * oof_s1_lgb + 0.25 * oof_v6_lgb + 0.15 * oof_s1_xgb + 0.10 * oof_v6_xgb + 0.15 * oof_s1_cat
pte_s1 = 0.35 * pte_s1_lgb + 0.25 * pte_v6_lgb + 0.15 * pte_s1_xgb + 0.10 * pte_v6_xgb + 0.15 * pte_s1_cat

oof_s2_lgb, pte_s2_lgb = load_tag("oof_s2_lgb")
oof_s2_xgb, pte_s2_xgb = load_tag("oof_s2_xgb")
if (ROOT / "models/oof_s2_cat/oof.npy").exists():
    oof_s2_cat, pte_s2_cat = load_tag("oof_s2_cat")
    oof_s2 = 0.45 * oof_s2_lgb + 0.35 * oof_s2_xgb + 0.20 * oof_s2_cat
    pte_s2 = 0.45 * pte_s2_lgb + 0.35 * pte_s2_xgb + 0.20 * pte_s2_cat
    print("Loaded 3-family Stage-2 (LGB + XGB + CAT)")
else:
    oof_s2 = 0.55 * oof_s2_lgb + 0.45 * oof_s2_xgb
    pte_s2 = 0.55 * pte_s2_lgb + 0.45 * pte_s2_xgb
    print("Loaded 2-family Stage-2 (LGB + XGB)")

print("\n=== STAGE-1 vs STAGE-2 BLEND SWEEP ===")
for alpha in (0.0, 0.25, 0.40, 0.50, 0.65, 0.80, 1.0):
    oof_b = (1.0 - alpha) * oof_s1 + alpha * oof_s2
    w = fit_weights(oof_b, y_clean)
    pred_tr, n_tr = enhanced_band_rules((oof_b * w).argmax(1), tr)
    f1_cln = macro_f1(y_clean, pred_tr)
    f1_raw = macro_f1(y_raw, pred_tr)
    print(f"alpha_s2={alpha:.2f} -> Clean OOF F1: {f1_cln:.4f} | Raw OOF F1: {f1_raw:.4f} (train band changes: {n_tr})")


def save_submission(oof_b, pte_b, name):
    w = fit_weights(oof_b, y_clean)
    pred_tr, _ = enhanced_band_rules((oof_b * w).argmax(1), tr)
    pred_te, n_te = enhanced_band_rules((pte_b * w).argmax(1), ~tr)
    print(f"\n--- {name} ---")
    print(f"Pooled OOF Macro-F1: vs clean {macro_f1(y_clean, pred_tr):.4f} | vs raw {macro_f1(y_raw, pred_tr):.4f} | test band changes: {n_te}")
    T = per_class(y_clean, pred_tr)
    T["w"] = w
    print(T.round(3).to_string())

    sub = pd.DataFrame({"id": base.txn_id.values[~tr], "label": np.array(LABELS)[pred_te]})
    sample = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
    sub = sample[["id"]].merge(sub, on="id", how="left", validate="one_to_one")
    assert len(sub) == len(sample) and sub.label.notna().all() and (sub.id.values == sample.id.values).all()
    assert set(sub.label) == set(LABELS)
    out_path = ROOT / "submissions" / f"{name}.csv"
    sub.to_csv(out_path, index=False)

    # Compare against v4_master_ensemble.csv
    v4 = pd.read_csv(ROOT / "submissions/v4_master_ensemble.csv")
    diff = (sub.label != v4.label).sum()
    n_fraud = (sub.label != "legit").sum()
    print(f"Saved {out_path} | Total test frauds: {n_fraud} | Differs from v4_master_ensemble.csv on {diff} rows")


# 1. Pure Stage-2 (100% 3-family Stage-2: Clean OOF F1 = 0.5623)
save_submission(oof_s2, pte_s2, "v5_pure_stage2")

# 2. Peak Stage-2 + Stage-1 Master Ensemble (80% Stage-2 + 20% Stage-1: Clean OOF F1 = 0.5624)
save_submission(0.20 * oof_s1 + 0.80 * oof_s2, 0.20 * pte_s1 + 0.80 * pte_s2, "v5_stage2_master")

# 3. Raw-Label Peak Ensemble (25% Stage-2 + 75% Stage-1: Raw OOF F1 = 0.5120, Clean OOF F1 = 0.5603)
save_submission(0.75 * oof_s1 + 0.25 * oof_s2, 0.75 * pte_s1 + 0.25 * pte_s2, "v5_balanced_s1_s2")

