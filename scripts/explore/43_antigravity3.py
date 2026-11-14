"""
43_antigravity3.py — Verification and execution of Search 1, Search 2, and Search 3.

Rules enforced:
- Run only with `python -I`
- Numerical threads capped at 4 before any library imports
- Modifies no existing files; writes nothing under data/, models/, or submissions/
- Prints every number used in docs/antigravity3.md
"""
import os

for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "ARROW_NUM_THREADS",
):
    os.environ[_k] = "4"

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import pyarrow as pa
from scipy import stats
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

pa.set_cpu_count(4)
pa.set_io_thread_count(4)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "upi"))
from common import CATS, DROP, LABELS, TRAPS, load_xy  # noqa: E402


def load_all_base():
    P = ROOT / "data" / "processed"
    df = pd.read_parquet(P / "base_v0.parquet")
    f1 = pd.read_parquet(P / "features_v1.parquet")
    f2 = pd.read_parquet(P / "features_v2.parquet")
    f3 = pd.read_parquet(P / "features_v3.parquet")
    f4 = pd.read_parquet(P / "features_v4.parquet")
    f5 = pd.read_parquet(P / "features_v5.parquet")
    for f in (f1, f2, f3, f4, f5):
        assert np.array_equal(f.txn_id.values, df.txn_id.values)
        new_cols = [c for c in f.columns if c != "txn_id" and c not in df.columns]
        df = pd.concat([df, f[new_cols]], axis=1)

    s1_tr = np.load(ROOT / "models" / "s1" / "oof.npy")
    s1_te = np.load(ROOT / "models" / "s1" / "test_proba.npy")
    is_tr = (df.split == "train").values
    s1_pf = np.empty(len(df), dtype=np.float64)
    s1_pf[is_tr] = 1.0 - s1_tr[:, 0]
    s1_pf[~is_tr] = 1.0 - s1_te[:, 0]
    df["s1_pf"] = s1_pf
    df["dn"] = df.device_id.str[2:].astype(np.int64)
    df["amt_over_bal"] = df.amount / np.maximum(df.balance_before, 1e-9)

    month = df.month.astype(str).values.copy()
    month[is_tr & (month > "2026-09")] = "2026-09"
    month[~is_tr] = "2026-10"
    df["m_clean"] = month
    return df


# ==============================================================================
# SEARCH 1: Is ordinary-looking fraud different from legit in October?
# ==============================================================================
def run_search_1(df):
    print("=" * 80)
    print("SEARCH 1: IS ORDINARY-LOOKING FRAUD DIFFERENT FROM LEGIT IN OCTOBER?")
    print("=" * 80)

    is_tr = (df.split == "train").values
    dn = df.dn.values
    ext_legit = df.dev_ext_legit.values.astype(int)
    s1_pf = df.s1_pf.values

    te_f_mask = (~is_tr) & (dn >= 473632) & (dn <= 474551) & (s1_pf < 0.2)
    te_l_mask = (~is_tr) & (ext_legit == 1) & (s1_pf < 0.2)
    tr_f_mask = is_tr & (dn >= 423001) & (dn <= 428454) & (s1_pf < 0.2)
    tr_l_mask = is_tr & (ext_legit == 1) & (s1_pf < 0.2)

    print(
        f"Sample counts (s1_pf < 0.2 on new devices):\n"
        f"  Test  fraud-band (473632-474551): {int(te_f_mask.sum())} | dev_ext_legit: {int(te_l_mask.sum())}\n"
        f"  Train fraud-band (423001-428454): {int(tr_f_mask.sum())} | dev_ext_legit: {int(tr_l_mask.sum())}"
    )

    for split_lbl, fm, lm in [("TEST (Oct)", te_f_mask, te_l_mask), ("TRAIN (Jul-Sep)", tr_f_mask, tr_l_mask)]:
        af, al = df.amount.values[fm], df.amount.values[lm]
        wf, wl = df.amt_is_integer.values[fm], df.amt_is_integer.values[lm]
        gmean_ratio = float(np.exp(np.log(af).mean() - np.log(al).mean()))
        mw_p = float(stats.mannwhitneyu(af, al, alternative="two-sided").pvalue)
        ks_p = float(stats.ks_2samp(af, al).pvalue)
        chi_w_p = float(
            stats.fisher_exact([[int(wf.sum()), int(len(wf) - wf.sum())], [int(wl.sum()), int(len(wl) - wl.sum())]])[1]
        )
        print(
            f"  {split_lbl}: amount median {np.median(af):.2f} vs {np.median(al):.2f} | "
            f"gmean ratio x{gmean_ratio:.4f} | MW p={mw_p:.4f} | KS p={ks_p:.4f} | "
            f"whole-rupee {wf.mean():.4f} vs {wl.mean():.4f} (Fisher p={chi_w_p:.4f})"
        )

    FEATURES_13 = [
        "amount",
        "amt_is_integer",
        "receiver_type",
        "txn_type",
        "merchant_category",
        "hour",
        "new_payee",
        "pin_retries",
        "login_to_txn_sec",
        "sim_age_hours",
        "device_age_days",
        "balance_before",
        "amt_over_bal",
    ]
    CONT_VARS = {
        "amount",
        "hour",
        "pin_retries",
        "login_to_txn_sec",
        "sim_age_hours",
        "device_age_days",
        "balance_before",
        "amt_over_bal",
    }
    LOG_VARS = {"amount", "balance_before", "amt_over_bal", "sim_age_hours", "device_age_days"}

    def run_13_and_78_permutation_suite(sub_df, y_bin, n_perm=10000, seed=42):
        rng = np.random.default_rng(seed)
        n = len(sub_df)
        n1 = int(y_bin.sum())
        n0 = n - n1

        binned_codes = {}
        n_levels = {}
        z_vecs = {}

        for col in FEATURES_13:
            s = sub_df[col]
            if col in ("receiver_type", "txn_type", "merchant_category"):
                s_str = s.astype("object").fillna("NA").astype(str)
                cats = pd.Categorical(s_str)
                binned_codes[col] = cats.codes.astype(np.int32)
                n_levels[col] = len(cats.categories)
                if col == "receiver_type":
                    v = (s_str.values == "P2P").astype(np.float64)
                elif col == "txn_type":
                    v = (s_str.values == "PAY").astype(np.float64)
                else:
                    v = (s_str.values == "ecommerce").astype(np.float64)
                z_vecs[col] = (v - v.mean()) / max(v.std(), 1e-9)
            elif col in ("amt_is_integer", "new_payee"):
                v = s.values.astype(np.float64)
                binned_codes[col] = v.astype(np.int32)
                n_levels[col] = 2
                z_vecs[col] = (v - v.mean()) / max(v.std(), 1e-9)
            elif col == "pin_retries":
                v = s.values.astype(np.float64)
                b = np.clip(v.astype(np.int32), 0, 2)
                binned_codes[col] = b
                n_levels[col] = 3
                z_vecs[col] = (v - v.mean()) / max(v.std(), 1e-9)
            elif col == "hour":
                v = s.values.astype(np.float64)
                b = (v // 6).astype(np.int32)
                binned_codes[col] = b
                n_levels[col] = 4
                rk = stats.rankdata(v)
                z_vecs[col] = (rk - rk.mean()) / max(rk.std(), 1e-9)
            else:
                v = s.values.astype(np.float64)
                rk = stats.rankdata(v)
                z_vecs[col] = (rk - rk.mean()) / max(rk.std(), 1e-9)
                q = pd.qcut(rk, q=4, labels=False, duplicates="drop").astype(np.int32)
                binned_codes[col] = q
                n_levels[col] = int(q.max() + 1)

        test_specs = []
        col_blocks = []
        col_offset = 0
        total_scalar_comparisons = 0

        for col in FEATURES_13:
            cols_this = []
            b = binned_codes[col]
            K_lv = n_levels[col]
            for lv in range(K_lv):
                ind = (b == lv).astype(np.float64)
                p = ind.mean()
                if 0 < p < 1:
                    cols_this.append((ind - p) / np.sqrt(p * (1.0 - p)))
            if col in CONT_VARS:
                v = sub_df[col].values.astype(np.float64)
                cols_this.append((v - v.mean()) / max(v.std(), 1e-9))
                cols_this.append(z_vecs[col])
                if col in LOG_VARS:
                    lv = np.log(np.maximum(v, 1e-9))
                    cols_this.append((lv - lv.mean()) / max(lv.std(), 1e-9))
            block = np.column_stack(cols_this)
            k_cols = block.shape[1]
            total_scalar_comparisons += k_cols
            test_specs.append(("marginal", col, col_offset, col_offset + k_cols))
            col_blocks.append(block)
            col_offset += k_cols

        for c1, c2 in itertools.combinations(FEATURES_13, 2):
            cols_this = []
            b1, K1 = binned_codes[c1], n_levels[c1]
            b2, K2 = binned_codes[c2], n_levels[c2]
            joint = b1 * K2 + b2
            for jv in range(K1 * K2):
                ind = (joint == jv).astype(np.float64)
                p = ind.mean()
                if p >= 5.0 / n and p < 1.0:
                    cols_this.append((ind - p) / np.sqrt(p * (1.0 - p)))
            prod = z_vecs[c1] * z_vecs[c2]
            if prod.std() > 1e-9:
                cols_this.append((prod - prod.mean()) / prod.std())
            if c1 == "amount" or c2 == "amount":
                other = c2 if c1 == "amount" else c1
                la = np.log(np.maximum(sub_df["amount"].values.astype(np.float64), 1e-9))
                la_z = (la - la.mean()) / max(la.std(), 1e-9)
                bo, Ko = binned_codes[other], n_levels[other]
                for lv in range(Ko):
                    m_lv = bo == lv
                    if m_lv.sum() >= 10:
                        vec = np.where(m_lv, la_z - la_z[m_lv].mean(), 0.0)
                        s_v = vec.std()
                        if s_v > 1e-9:
                            cols_this.append(vec / s_v)
            block = np.column_stack(cols_this)
            k_cols = block.shape[1]
            total_scalar_comparisons += k_cols
            test_specs.append(("pair", f"{c1} x {c2}", col_offset, col_offset + k_cols))
            col_blocks.append(block)
            col_offset += k_cols

        M = np.hstack(col_blocks).astype(np.float32)
        scale = float(np.sqrt(n1 * n0 / (n - 1.0)))
        z_obs = np.abs((y_bin.astype(np.float32) @ M) / scale)

        n_hyp = len(test_specs)
        stat_obs = np.empty(n_hyp, dtype=np.float32)
        for h_i, (_, _, s_idx, e_idx) in enumerate(test_specs):
            stat_obs[h_i] = z_obs[s_idx:e_idx].max()

        stat_perm = np.empty((n_perm, n_hyp), dtype=np.float32)
        max_all_scalar_perm = np.empty(n_perm, dtype=np.float32)
        chunk = 500
        for start_p in range(0, n_perm, chunk):
            b_sz = min(chunk, n_perm - start_p)
            Y_p = np.zeros((b_sz, n), dtype=np.float32)
            for r_i in range(b_sz):
                idx_1 = rng.choice(n, size=n1, replace=False)
                Y_p[r_i, idx_1] = 1.0
            Z_p = np.abs((Y_p @ M) / scale)
            max_all_scalar_perm[start_p : start_p + b_sz] = Z_p.max(axis=1)
            for h_i, (_, _, s_idx, e_idx) in enumerate(test_specs):
                stat_perm[start_p : start_p + b_sz, h_i] = Z_p[:, s_idx:e_idx].max(axis=1)

        p_raw = np.empty(n_hyp, dtype=np.float64)
        for h_i in range(n_hyp):
            p_raw[h_i] = (1.0 + (stat_perm[:, h_i] >= stat_obs[h_i] - 1e-7).sum()) / (n_perm + 1.0)

        ranks = np.argsort(np.argsort(-stat_perm, axis=0), axis=0)
        p_perm_matrix = (1.0 + ranks) / (n_perm + 1.0)
        min_p_null_91 = p_perm_matrix.min(axis=1)
        min_p_null_13 = p_perm_matrix[:, :13].min(axis=1)

        p_wy_91 = np.empty(n_hyp, dtype=np.float64)
        p_wy_13 = np.empty(n_hyp, dtype=np.float64)
        for h_i in range(n_hyp):
            p_wy_91[h_i] = (1.0 + (min_p_null_91 <= p_raw[h_i] + 1e-12).sum()) / (n_perm + 1.0)
            p_wy_13[h_i] = (1.0 + (min_p_null_13 <= p_raw[h_i] + 1e-12).sum()) / (n_perm + 1.0) if h_i < 13 else np.nan

        res = []
        for h_i, (kind, name, s_idx, e_idx) in enumerate(test_specs):
            res.append({
                "kind": kind,
                "name": name,
                "n_cols": e_idx - s_idx,
                "max_z": float(stat_obs[h_i]),
                "p_perm_raw": float(p_raw[h_i]),
                "p_bonf_13": min(1.0, float(p_raw[h_i] * 13)) if h_i < 13 else np.nan,
                "p_wy_13": float(p_wy_13[h_i]),
                "p_bonf_91": min(1.0, float(p_raw[h_i] * 91)),
                "p_wy_91": float(p_wy_91[h_i]),
            })
        return (
            pd.DataFrame(res),
            total_scalar_comparisons,
            float(z_obs.max()),
            float((1.0 + (max_all_scalar_perm >= z_obs.max() - 1e-7).sum()) / (n_perm + 1.0)),
        )

    te_sub = df.loc[te_f_mask | te_l_mask].copy()
    te_y = te_f_mask[te_f_mask | te_l_mask].astype(int)
    tr_sub = df.loc[tr_f_mask | tr_l_mask].copy()
    tr_y = tr_f_mask[tr_f_mask | tr_l_mask].astype(int)

    res_te, n_sc_te, max_z_te, p_global_te = run_13_and_78_permutation_suite(te_sub, te_y, n_perm=10000, seed=42)
    res_tr, n_sc_tr, max_z_tr, p_global_tr = run_13_and_78_permutation_suite(tr_sub, tr_y, n_perm=4000, seed=42)

    merged_1a = res_te.merge(res_tr, on=["kind", "name"], suffixes=("_te", "_tr"))
    print(f"\n--- 1a. Permutation Test Results (91 hypotheses = 13 marginals + 78 pairs; {n_sc_te} scalar cells in Test, {n_sc_tr} in Train) ---")
    print(f"Global max-|z| across all {n_sc_te} scalar cells in Test: max_z={max_z_te:.3f}, FWER p={p_global_te:.4f}")
    print(f"Global max-|z| across all {n_sc_tr} scalar cells in Train: max_z={max_z_tr:.3f}, FWER p={p_global_tr:.4f}")
    print("\nAll 13 Marginal Tests (Test vs Train):")
    print(merged_1a[merged_1a.kind == "marginal"][[
        "name", "n_cols_te", "max_z_te", "p_perm_raw_te", "p_bonf_13_te", "p_wy_13_te", "p_wy_91_te",
        "max_z_tr", "p_perm_raw_tr", "p_wy_91_tr"
    ]].round(4).to_string(index=False))

    print("\nTop 15 Pair Tests in Test sorted by raw permutation p-value (out of 78 pairs, 91 total hypotheses):")
    print(merged_1a[merged_1a.kind == "pair"].sort_values("p_perm_raw_te").head(15)[[
        "name", "n_cols_te", "max_z_te", "p_perm_raw_te", "p_bonf_91_te", "p_wy_91_te",
        "max_z_tr", "p_perm_raw_tr", "p_wy_91_tr"
    ]].round(4).to_string(index=False))

    print("\nTop 10 Pair Tests in Train sorted by raw permutation p-value:")
    print(merged_1a[merged_1a.kind == "pair"].sort_values("p_perm_raw_tr").head(10)[[
        "name", "n_cols_tr", "max_z_tr", "p_perm_raw_tr", "p_bonf_91_tr", "p_wy_91_tr",
        "max_z_te", "p_perm_raw_te", "p_wy_91_te"
    ]].round(4).to_string(index=False))


# ==============================================================================
# SEARCH 2: Does the October shift cost detections?
# ==============================================================================
def run_search_2(df):
    print("\n" + "=" * 80)
    print("SEARCH 2: DOES THE OCTOBER SHIFT COST DETECTIONS?")
    print("=" * 80)

    is_tr = (df.split == "train").values
    oof = np.load(ROOT / "models" / "oof_final_local" / "oof.npy").astype(np.float64)
    Pte = np.load(ROOT / "models" / "oof_final_local" / "test_proba.npy").astype(np.float64)
    w = np.load(ROOT / "models" / "oof_final_local" / "weights_final_local.npy").astype(np.float64)

    P_all = np.empty((len(df), len(LABELS)), dtype=np.float64)
    P_all[is_tr] = oof
    P_all[~is_tr] = Pte
    pf_all = 1.0 - P_all[:, 0]
    pred_raw = (P_all * w).argmax(axis=1)
    best_fraud_cls = (P_all[:, 1:] * w[1:]).argmax(axis=1) + 1

    # Also compute post-band_rules prediction (matching make_submission.py)
    band = df.band.values.astype(int)
    S_b = np.stack([df[f"snd_band_{k}_3h"].values for k in range(1, 12)], 1)
    R_b = np.stack([df[f"rcv_band_{k}_24h"].values for k in range(1, 12)], 1)
    sb = np.where((S_b > 0).sum(1) == 1, S_b.argmax(1) + 1, 0)
    rb = np.where((R_b > 0).sum(1) == 1, R_b.argmax(1) + 1, 0)
    cb = df.ch_band.values.astype(int)
    p2p = (df.receiver_type.values == "P2P")
    pred_band = np.where(band > 0, band, pred_raw)
    free = (band == 0) & (pred_raw > 0)
    drain = free & np.isin(pred_raw, [4, 5]) & np.isin(sb, [4, 5])
    fanin = free & np.isin(pred_raw, [1, 2, 3]) & p2p & np.isin(rb, [1, 2, 3])
    shop = free & np.isin(pred_raw, [7, 8, 11]) & ~p2p & np.isin(rb, [7, 8, 11])
    chain = free & (cb > 0)
    pred_band = np.where(drain, sb, pred_band)
    pred_band = np.where(fanin | shop, rb, pred_band)
    pred_band = np.where(chain, cb, pred_band)

    df["pf_final"] = pf_all
    df["pred_raw"] = pred_raw
    df["pred_band"] = pred_band
    df["best_fraud_cls"] = best_fraud_cls

    # Exact input columns used by the model (from common.py load_xy with features_v1..v5)
    X_model, _, _ = load_xy(
        ROOT,
        feature_files=(
            "features_v1.parquet",
            "features_v2.parquet",
            "features_v3.parquet",
            "features_v4.parquet",
            "features_v5.parquet",
        ),
    )
    num_cols = [c for c in X_model.columns if c not in CATS]
    print(f"Total model input columns: {len(X_model.columns)} ({len(num_cols)} numeric + {len(CATS)} categorical)")

    # Print monthly predicted class counts and rates per 100k (both raw weighted pred_raw and post-band pred_band)
    m_arr = df.m_clean.values
    m_list = ["2026-07", "2026-08", "2026-09", "2026-10"]
    m_names = ["Jul", "Aug", "Sep", "Oct"]
    m_Denom = {m: int((m_arr == m).sum()) for m in m_list}

    print("\n--- Monthly Flagged Row Counts & Rates per 100k by Predicted Type (pred_raw = argmax(P * w)) ---")
    pred_rate_rows = []
    for k in range(12):
        r_d = {"class_id": k, "label": LABELS[k]}
        for m_code, m_nm in zip(m_list, m_names):
            cnt = int(((m_arr == m_code) & (pred_raw == k)).sum())
            cnt_b = int(((m_arr == m_code) & (pred_band == k)).sum())
            r_d[f"{m_nm}_n"] = cnt
            r_d[f"{m_nm}_r100k"] = cnt / m_Denom[m_code] * 1e5
            r_d[f"{m_nm}_band_n"] = cnt_b
        r_d["Oct_vs_TrMean"] = r_d["Oct_r100k"] / max((r_d["Jul_r100k"] + r_d["Aug_r100k"] + r_d["Sep_r100k"]) / 3.0, 1e-9)
        pred_rate_rows.append(r_d)
    print(pd.DataFrame(pred_rate_rows).round(2).to_string(index=False))

    # 2a. Compare Jul-Sep against October separately for rows the model calls legit (k=0) and for rows it flags (k=1..11)
    # Count total comparisons = len(num_cols) * 12
    n_comp_2a = len(num_cols) * 12
    print(f"\n--- 2a. Input Column Distribution Shifts (Jul-Sep vs Oct across 12 predicted classes x {len(num_cols)} numeric cols = {n_comp_2a} comparisons) ---")

    shift_rows_legit = []
    shift_rows_flagged = []

    for k in range(12):
        m_tr_k = is_tr & (pred_raw == k)
        m_te_k = (~is_tr) & (pred_raw == k)
        m_jul_k = (m_arr == "2026-07") & (pred_raw == k)
        m_aug_k = (m_arr == "2026-08") & (pred_raw == k)
        m_sep_k = (m_arr == "2026-09") & (pred_raw == k)

        for col in num_cols:
            v_tr = X_model[col].values[m_tr_k]
            v_te = X_model[col].values[m_te_k]
            v_tr_ok = v_tr[np.isfinite(v_tr)]
            v_te_ok = v_te[np.isfinite(v_te)]
            if len(v_tr_ok) < 10 or len(v_te_ok) < 5:
                continue
            ks_res = stats.ks_2samp(v_tr_ok, v_te_ok)
            ks_stat = float(ks_res.statistic)
            ks_p = float(ks_res.pvalue)
            mw_p = float(stats.mannwhitneyu(v_tr_ok, v_te_ok).pvalue)

            tr_mean, te_mean = float(v_tr_ok.mean()), float(v_te_ok.mean())
            tr_std = float(v_tr_ok.std())
            smd = (te_mean - tr_mean) / max(tr_std, 1e-9)
            tr_med, te_med = float(np.median(v_tr_ok)), float(np.median(v_te_ok))

            # Also compute Jul, Aug, Sep means to verify if Oct is outside the 3 train months
            j_v = X_model[col].values[m_jul_k]
            a_v = X_model[col].values[m_aug_k]
            s_v = X_model[col].values[m_sep_k]
            j_m = float(np.nanmean(j_v)) if np.isfinite(j_v).any() else np.nan
            a_m = float(np.nanmean(a_v)) if np.isfinite(a_v).any() else np.nan
            s_m = float(np.nanmean(s_v)) if np.isfinite(s_v).any() else np.nan

            rec = {
                "pred_class": LABELS[k],
                "column": col,
                "n_tr": len(v_tr_ok),
                "n_oct": len(v_te_ok),
                "tr_mean": tr_mean,
                "oct_mean": te_mean,
                "tr_med": tr_med,
                "oct_med": te_med,
                "jul_mean": j_m,
                "aug_mean": a_m,
                "sep_mean": s_m,
                "smd": smd,
                "KS_stat": ks_stat,
                "KS_p": ks_p,
                "KS_p_bonf": min(1.0, ks_p * n_comp_2a),
                "MW_p": mw_p,
            }
            if k == 0:
                shift_rows_legit.append(rec)
            else:
                shift_rows_flagged.append(rec)

    df_legit_shifts = pd.DataFrame(shift_rows_legit).sort_values("KS_stat", ascending=False)
    df_flag_shifts = pd.DataFrame(shift_rows_flagged).sort_values("KS_stat", ascending=False)

    print("\nTop 25 Shifted Columns Among Rows Predicted LEGIT (pred == 0), sorted by KS_stat:")
    print(df_legit_shifts.head(25)[[
        "column", "tr_mean", "oct_mean", "tr_med", "oct_med", "smd", "KS_stat", "KS_p", "KS_p_bonf"
    ]].round(4).to_string(index=False))

    # Filter flagged shifts: KS_p_bonf < 0.05 (across 1,980 tests)
    sig_flag_shifts = df_flag_shifts[df_flag_shifts["KS_p_bonf"] < 0.05].copy()
    print(f"\nColumns with statistically significant shift among FLAGGED rows (KS_p_bonf < 0.05 across {n_comp_2a} tests): {len(sig_flag_shifts)} pairs")
    print(sig_flag_shifts.head(50)[[
        "pred_class", "column", "n_tr", "n_oct", "tr_mean", "oct_mean", "tr_med", "oct_med", "smd", "KS_stat", "KS_p", "KS_p_bonf"
    ]].round(4).to_string(index=False))

    print("\nTop 3 shifted columns per Predicted Fraud Class (1..11), regardless of significance threshold:")
    for k in range(1, 12):
        sub_k = df_flag_shifts[df_flag_shifts.pred_class == LABELS[k]].head(3)
        print(sub_k[[
            "pred_class", "column", "n_tr", "n_oct", "tr_mean", "oct_mean", "tr_med", "oct_med", "KS_stat", "KS_p", "KS_p_bonf"
        ]].round(4).to_string(index=False))

    # 2b. Near misses:
    # Part (i): groups of 3 or more rows linked by sender, receiver or device within 24h whose fraud probability is between 0.05 and 0.5
    # Part (ii): events where some rows are flagged (pred_raw > 0) and others are not (pred_raw == 0).
    print("\n--- 2b. Near-Miss Groups & Partially Flagged Events Across Jul, Aug, Sep, Oct ---")

    t_sec = df.t.values.astype(np.int64)
    snd_num = df.sender_id.str[2:].astype(np.int64).values
    r_num = df.receiver_id.str[2:].astype(np.int64).values
    rcv_ns = np.where(p2p, r_num, r_num + 1_000_000)
    dev_num = df.dn.values
    W24 = 86400

    def build_24h_components(mask_rows, include_all_merch_rcv=False):
        """Build 24h connected components among rows where mask_rows is True."""
        idx = np.flatnonzero(mask_rows)
        if len(idx) == 0:
            return np.full(len(df), -1, dtype=np.int64), {}
        edges_l, edges_r = [], []
        # Links: sender_id (24h), device_id (24h), receiver_id (24h: either P2P | S_flag==1, or all receivers)
        rcv_elig = np.ones(len(df), dtype=bool) if include_all_merch_rcv else (p2p | (df.S_flag.values == 1) | (df.Q_flag.values == 1) | (df.R_flag.values == 1))
        for key_arr, elig_arr in [(snd_num, np.ones(len(df), bool)), (dev_num, np.ones(len(df), bool)), (rcv_ns, rcv_elig)]:
            sub_i = idx[elig_arr[idx]]
            if len(sub_i) < 2:
                continue
            ord_i = sub_i[np.lexsort((t_sec[sub_i], key_arr[sub_i]))]
            same_k = (key_arr[ord_i[1:]] == key_arr[ord_i[:-1]]) & ((t_sec[ord_i[1:]] - t_sec[ord_i[:-1]]) <= W24)
            edges_l.append(ord_i[:-1][same_k])
            edges_r.append(ord_i[1:][same_k])
        if edges_l:
            L_e = np.concatenate(edges_l)
            R_e = np.concatenate(edges_r)
            G = coo_matrix((np.ones(len(L_e), dtype=np.int8), (L_e, R_e)), shape=(len(df), len(df)))
            _, comp_labels = connected_components(G, directed=False)
        else:
            comp_labels = np.arange(len(df), dtype=np.int64)

        comp_sub = comp_labels[idx]
        s_ser = pd.Series(idx).groupby(comp_sub, sort=False)
        groups_ge3 = {int(cid): grp.values for cid, grp in s_ser if len(grp) >= 3}
        row_comp = np.full(len(df), -1, dtype=np.int64)
        for cid, r_idxs in groups_ge3.items():
            row_comp[r_idxs] = cid
        return row_comp, groups_ge3

    # Test both receiver link variants for Part (i):
    # Variant A: P2P receiver or recipe-eligible merchant (S_flag | Q_flag | R_flag) within 24h (avoids linking 50 random customers at a big grocery store)
    # Variant B: Sender or Device within 24h, or P2P receiver within 24h (pure entity links without busy merchant background)
    # Variant C: Any receiver including busy merchants within 24h
    nm_mask = (pf_all >= 0.05) & (pf_all <= 0.50)
    nm_mask_15 = (pf_all >= 0.15) & (pf_all <= 0.50)
    s1_all = df.s1_pf.values
    nm_s1_15 = (s1_all >= 0.15) & (s1_all <= 0.50)
    print(f"Total rows with 0.05 <= pf <= 0.50: Jul={int((nm_mask & (m_arr=='2026-07')).sum())} ({((nm_mask & (m_arr=='2026-07')).sum()/m_Denom['2026-07']*1e5):.1f}/100k), "
          f"Aug={int((nm_mask & (m_arr=='2026-08')).sum())} ({((nm_mask & (m_arr=='2026-08')).sum()/m_Denom['2026-08']*1e5):.1f}/100k), "
          f"Sep={int((nm_mask & (m_arr=='2026-09')).sum())} ({((nm_mask & (m_arr=='2026-09')).sum()/m_Denom['2026-09']*1e5):.1f}/100k), "
          f"Oct={int((nm_mask & (m_arr=='2026-10')).sum())} ({((nm_mask & (m_arr=='2026-10')).sum()/m_Denom['2026-10']*1e5):.1f}/100k)")
    print(f"Total rows with 0.15 <= pf <= 0.50 (above ~0.08 ordinary baseline): Jul={int((nm_mask_15 & (m_arr=='2026-07')).sum())} ({((nm_mask_15 & (m_arr=='2026-07')).sum()/m_Denom['2026-07']*1e5):.2f}/100k), "
          f"Aug={int((nm_mask_15 & (m_arr=='2026-08')).sum())} ({((nm_mask_15 & (m_arr=='2026-08')).sum()/m_Denom['2026-08']*1e5):.2f}/100k), "
          f"Sep={int((nm_mask_15 & (m_arr=='2026-09')).sum())} ({((nm_mask_15 & (m_arr=='2026-09')).sum()/m_Denom['2026-09']*1e5):.2f}/100k), "
          f"Oct={int((nm_mask_15 & (m_arr=='2026-10')).sum())} ({((nm_mask_15 & (m_arr=='2026-10')).sum()/m_Denom['2026-10']*1e5):.2f}/100k)")
    print(f"Total rows with 0.15 <= s1_pf <= 0.50 (stage-1 above baseline): Jul={int((nm_s1_15 & (m_arr=='2026-07')).sum())} ({((nm_s1_15 & (m_arr=='2026-07')).sum()/m_Denom['2026-07']*1e5):.2f}/100k), "
          f"Aug={int((nm_s1_15 & (m_arr=='2026-08')).sum())} ({((nm_s1_15 & (m_arr=='2026-08')).sum()/m_Denom['2026-08']*1e5):.2f}/100k), "
          f"Sep={int((nm_s1_15 & (m_arr=='2026-09')).sum())} ({((nm_s1_15 & (m_arr=='2026-09')).sum()/m_Denom['2026-09']*1e5):.2f}/100k), "
          f"Oct={int((nm_s1_15 & (m_arr=='2026-10')).sum())} ({((nm_s1_15 & (m_arr=='2026-10')).sum()/m_Denom['2026-10']*1e5):.2f}/100k)")

    for var_name, mask_use, inc_merch in [
        ("2b-i (A: 24h snd/dev/P2P_rcv/recipe_merch, pf in [0.05, 0.50], size>=3)", nm_mask, False),
        ("2b-i (B: 24h snd/dev/ALL_rcv incl. background merchants, pf in [0.05, 0.50], size>=3)", nm_mask, True),
        ("2b-i (C: 24h snd/dev/ALL_rcv, pf in [0.15, 0.50] above baseline, size>=3)", nm_mask_15, True),
        ("2b-i (D: 24h snd/dev/ALL_rcv, s1_pf in [0.15, 0.50] above baseline, size>=3)", nm_s1_15, True),
    ]:
        row_c, grps_c = build_24h_components(mask_use, include_all_merch_rcv=inc_merch)
        in_grp = row_c >= 0
        # Assign each group's dominant fraud type (by summed weighted fraud probability across the group)
        grp_type = np.zeros(len(df), dtype=int)
        for cid, r_idxs in grps_c.items():
            w_probs = (P_all[r_idxs, 1:] * w[1:]).sum(axis=0)
            dom_c = int(w_probs.argmax()) + 1
            grp_type[r_idxs] = dom_c

        print(f"\n{var_name}:")
        rows_tbl = []
        exceed_types = []
        for k in range(1, 12):
            mk = in_grp & (grp_type == k)
            r_d = {"class_id": k, "label": LABELS[k]}
            for m_code, m_nm in zip(m_list, m_names):
                cnt = int((mk & (m_arr == m_code)).sum())
                n_ev = len(set(row_c[mk & (m_arr == m_code)]))
                r_d[f"{m_nm}_rows"] = cnt
                r_d[f"{m_nm}_ev"] = n_ev
                r_d[f"{m_nm}_r100k"] = cnt / m_Denom[m_code] * 1e5
            max_tr_r = max(r_d["Jul_r100k"], r_d["Aug_r100k"], r_d["Sep_r100k"])
            r_d["Oct_vs_MaxTr"] = r_d["Oct_r100k"] / max(max_tr_r, 1e-9) if r_d["Oct_rows"] > 0 else 0.0
            rows_tbl.append(r_d)
            if r_d["Oct_r100k"] > 2.0 * max_tr_r and r_d["Oct_rows"] > 0:
                exceed_types.append((k, mk & (m_arr == "2026-10")))
        # Also add TOTAL row
        r_tot = {"class_id": 0, "label": "TOTAL_ALL_TYPES"}
        for m_code, m_nm in zip(m_list, m_names):
            cnt = int((in_grp & (m_arr == m_code)).sum())
            n_ev = len(set(row_c[in_grp & (m_arr == m_code)]))
            r_tot[f"{m_nm}_rows"] = cnt
            r_tot[f"{m_nm}_ev"] = n_ev
            r_tot[f"{m_nm}_r100k"] = cnt / m_Denom[m_code] * 1e5
        max_tr_tot = max(r_tot["Jul_r100k"], r_tot["Aug_r100k"], r_tot["Sep_r100k"])
        r_tot["Oct_vs_MaxTr"] = r_tot["Oct_r100k"] / max(max_tr_tot, 1e-9)
        rows_tbl.append(r_tot)
        print(pd.DataFrame(rows_tbl).round(2).to_string(index=False))

        for k, oct_mask in exceed_types:
            print(f"  >>> Printing October rows for {LABELS[k]} (Oct_r100k > 2x max train month):")
            for r_i in np.flatnonzero(oct_mask):
                r = df.iloc[r_i]
                print(
                    f"      {r.txn_id} | {r.timestamp} | comp={row_c[r_i]} | snd={r.sender_id} rcv={r.receiver_id}({r.receiver_type}:{r.merchant_category}) "
                    f"dev={r.device_id} | {r.txn_type} amt={r.amount:.2f} bal={r.balance_before:.2f} pf={r.pf_final:.3f} best_cls={LABELS[int(r.best_fraud_cls)]}"
                )

    # Part (ii): Events where some rows are flagged (pred_raw > 0) and others are not (pred_raw == 0)
    # Test both:
    # (1) 24h entity-linked groups among (pred_raw > 0 | pf >= 0.05) where the component has BOTH pred_raw > 0 and pred_raw == 0
    # (2) Physical recipe sessions (Ds_n>=2, Vs_n>=3, Qs_n>=3, S_n>=2, ch_len>=2, bait/payback, R_snd_cnt_14d>=2, L_snd_cnt_7d>=2)
    #     where the session has BOTH pred_raw > 0 and pred_raw == 0.
    cand_mix_mask = (pred_raw > 0) | (pf_all >= 0.05)
    row_cm, grps_cm = build_24h_components(cand_mix_mask, include_all_merch_rcv=False)
    # Filter to components (of size >= 2) that contain at least one flagged row (pred_raw > 0) and at least one unflagged row (pred_raw == 0)
    idx_cand = np.flatnonzero(cand_mix_mask)
    edges_l, edges_r = [], []
    rcv_elig = p2p | (df.S_flag.values == 1) | (df.Q_flag.values == 1) | (df.R_flag.values == 1)
    for key_arr, elig_arr in [(snd_num, np.ones(len(df), bool)), (dev_num, np.ones(len(df), bool)), (rcv_ns, rcv_elig)]:
        sub_i = idx_cand[elig_arr[idx_cand]]
        ord_i = sub_i[np.lexsort((t_sec[sub_i], key_arr[sub_i]))]
        same_k = (key_arr[ord_i[1:]] == key_arr[ord_i[:-1]]) & ((t_sec[ord_i[1:]] - t_sec[ord_i[:-1]]) <= W24)
        edges_l.append(ord_i[:-1][same_k])
        edges_r.append(ord_i[1:][same_k])
    L_e = np.concatenate(edges_l)
    R_e = np.concatenate(edges_r)
    G_m = coo_matrix((np.ones(len(L_e), dtype=np.int8), (L_e, R_e)), shape=(len(df), len(df)))
    _, comp_m = connected_components(G_m, directed=False)

    # Identify mixed components (containing >=1 flagged row and >=1 unflagged companion with pf>=0.05)
    comp_ids_cand = comp_m[idx_cand]
    is_flg_cand = (pred_raw[idx_cand] > 0).astype(int)
    df_c = pd.DataFrame({"idx": idx_cand, "cid": comp_ids_cand, "flg": is_flg_cand})
    agg_c = df_c.groupby("cid")["flg"].agg(["count", "sum"])
    mixed_cids = set(agg_c[(agg_c["sum"] >= 1) & (agg_c["sum"] < agg_c["count"])].index)

    unflagged_in_mixed = np.zeros(len(df), dtype=bool)
    all_in_mixed = np.zeros(len(df), dtype=bool)
    mixed_type = np.zeros(len(df), dtype=int)
    for cid, grp in df_c[df_c["cid"].isin(mixed_cids)].groupby("cid"):
        r_idxs = grp["idx"].values
        flg_idxs = r_idxs[pred_raw[r_idxs] > 0]
        unflg_idxs = r_idxs[pred_raw[r_idxs] == 0]
        # Dominant flagged type
        dom_t = int(pd.Series(pred_raw[flg_idxs]).mode().iloc[0])
        all_in_mixed[r_idxs] = True
        unflagged_in_mixed[unflg_idxs] = True
        mixed_type[r_idxs] = dom_t

    print("\n2b-ii: Mixed Events (some rows flagged pred>0, companion rows unflagged with 0.05<=pf<=0.50 within 24h):")
    mix_tbl = []
    exceed_mix = []
    for k in range(1, 12):
        mk_u = unflagged_in_mixed & (mixed_type == k)
        mk_a = all_in_mixed & (mixed_type == k)
        r_d = {"class_id": k, "label": LABELS[k]}
        for m_code, m_nm in zip(m_list, m_names):
            u_cnt = int((mk_u & (m_arr == m_code)).sum())
            ev_cnt = len(set(comp_m[mk_a & (m_arr == m_code)]))
            r_d[f"{m_nm}_unflg_rows"] = u_cnt
            r_d[f"{m_nm}_ev"] = ev_cnt
            r_d[f"{m_nm}_r100k"] = u_cnt / m_Denom[m_code] * 1e5
        max_tr_r = max(r_d["Jul_r100k"], r_d["Aug_r100k"], r_d["Sep_r100k"])
        r_d["Oct_vs_MaxTr"] = r_d["Oct_r100k"] / max(max_tr_r, 1e-9) if r_d["Oct_unflg_rows"] > 0 else 0.0
        mix_tbl.append(r_d)
        if r_d["Oct_r100k"] > 2.0 * max_tr_r and r_d["Oct_unflg_rows"] > 0:
            exceed_mix.append((k, mk_a & (m_arr == "2026-10")))

    r_tot_m = {"class_id": 0, "label": "TOTAL_ALL_TYPES"}
    for m_code, m_nm in zip(m_list, m_names):
        u_cnt = int((unflagged_in_mixed & (m_arr == m_code)).sum())
        ev_cnt = len(set(comp_m[all_in_mixed & (m_arr == m_code)]))
        r_tot_m[f"{m_nm}_unflg_rows"] = u_cnt
        r_tot_m[f"{m_nm}_ev"] = ev_cnt
        r_tot_m[f"{m_nm}_r100k"] = u_cnt / m_Denom[m_code] * 1e5
    max_tr_m = max(r_tot_m["Jul_r100k"], r_tot_m["Aug_r100k"], r_tot_m["Sep_r100k"])
    r_tot_m["Oct_vs_MaxTr"] = r_tot_m["Oct_r100k"] / max(max_tr_m, 1e-9)
    mix_tbl.append(r_tot_m)
    print(pd.DataFrame(mix_tbl).round(2).to_string(index=False))

    for k, oct_mask in exceed_mix:
        print(f"  >>> Printing October mixed-event rows for {LABELS[k]} (Oct_r100k > 2x max train month):")
        for r_i in np.flatnonzero(oct_mask):
            r = df.iloc[r_i]
            print(
                f"      {r.txn_id} | {r.timestamp} | comp={comp_m[r_i]} | pred={LABELS[int(r.pred_raw)]} pf={r.pf_final:.3f} | "
                f"snd={r.sender_id} rcv={r.receiver_id}({r.receiver_type}:{r.merchant_category}) dev={r.device_id} | {r.txn_type} amt={r.amount:.2f} bal={r.balance_before:.2f}"
            )

    # Also evaluate 2b-ii with companion rows requiring pf >= 0.15 (above ~0.08 ordinary baseline)
    cand_mix_15 = (pred_raw > 0) | (pf_all >= 0.15)
    idx_cand_15 = np.flatnonzero(cand_mix_15)
    edges_l15, edges_r15 = [], []
    for key_arr, elig_arr in [(snd_num, np.ones(len(df), bool)), (dev_num, np.ones(len(df), bool)), (rcv_ns, rcv_elig)]:
        sub_i = idx_cand_15[elig_arr[idx_cand_15]]
        ord_i = sub_i[np.lexsort((t_sec[sub_i], key_arr[sub_i]))]
        same_k = (key_arr[ord_i[1:]] == key_arr[ord_i[:-1]]) & ((t_sec[ord_i[1:]] - t_sec[ord_i[:-1]]) <= W24)
        edges_l15.append(ord_i[:-1][same_k])
        edges_r15.append(ord_i[1:][same_k])
    G_m15 = coo_matrix((np.ones(len(np.concatenate(edges_l15)), dtype=np.int8), (np.concatenate(edges_l15), np.concatenate(edges_r15))), shape=(len(df), len(df)))
    _, comp_m15 = connected_components(G_m15, directed=False)
    df_c15 = pd.DataFrame({"idx": idx_cand_15, "cid": comp_m15[idx_cand_15], "flg": (pred_raw[idx_cand_15] > 0).astype(int)})
    agg_c15 = df_c15.groupby("cid")["flg"].agg(["count", "sum"])
    mixed_cids_15 = set(agg_c15[(agg_c15["sum"] >= 1) & (agg_c15["sum"] < agg_c15["count"])].index)
    unflg_15 = np.zeros(len(df), dtype=bool)
    all_15 = np.zeros(len(df), dtype=bool)
    mtype_15 = np.zeros(len(df), dtype=int)
    for cid, grp in df_c15[df_c15["cid"].isin(mixed_cids_15)].groupby("cid"):
        r_idxs = grp["idx"].values
        flg_idxs = r_idxs[pred_raw[r_idxs] > 0]
        unflg_idxs = r_idxs[pred_raw[r_idxs] == 0]
        dom_t = int(pd.Series(pred_raw[flg_idxs]).mode().iloc[0])
        all_15[r_idxs] = True
        unflg_15[unflg_idxs] = True
        mtype_15[r_idxs] = dom_t

    print("\n2b-ii (B: Mixed Events where companion unflagged rows have 0.15<=pf<=0.50 above baseline within 24h):")
    mix_tbl_15 = []
    exceed_mix_15 = []
    for k in range(1, 12):
        mk_u = unflg_15 & (mtype_15 == k)
        mk_a = all_15 & (mtype_15 == k)
        r_d = {"class_id": k, "label": LABELS[k]}
        for m_code, m_nm in zip(m_list, m_names):
            u_cnt = int((mk_u & (m_arr == m_code)).sum())
            ev_cnt = len(set(comp_m15[mk_a & (m_arr == m_code)]))
            r_d[f"{m_nm}_unflg_rows"] = u_cnt
            r_d[f"{m_nm}_ev"] = ev_cnt
            r_d[f"{m_nm}_r100k"] = u_cnt / m_Denom[m_code] * 1e5
        max_tr_r = max(r_d["Jul_r100k"], r_d["Aug_r100k"], r_d["Sep_r100k"])
        r_d["Oct_vs_MaxTr"] = r_d["Oct_r100k"] / max(max_tr_r, 1e-9) if r_d["Oct_unflg_rows"] > 0 else 0.0
        mix_tbl_15.append(r_d)
        if r_d["Oct_r100k"] > 2.0 * max_tr_r and r_d["Oct_unflg_rows"] > 0:
            exceed_mix_15.append((k, mk_a & (m_arr == "2026-10")))
    r_tot_15 = {"class_id": 0, "label": "TOTAL_ALL_TYPES"}
    for m_code, m_nm in zip(m_list, m_names):
        u_cnt = int((unflg_15 & (m_arr == m_code)).sum())
        ev_cnt = len(set(comp_m15[all_15 & (m_arr == m_code)]))
        r_tot_15[f"{m_nm}_unflg_rows"] = u_cnt
        r_tot_15[f"{m_nm}_ev"] = ev_cnt
        r_tot_15[f"{m_nm}_r100k"] = u_cnt / m_Denom[m_code] * 1e5
    max_tr_15 = max(r_tot_15["Jul_r100k"], r_tot_15["Aug_r100k"], r_tot_15["Sep_r100k"])
    r_tot_15["Oct_vs_MaxTr"] = r_tot_15["Oct_r100k"] / max(max_tr_15, 1e-9)
    mix_tbl_15.append(r_tot_15)
    print(pd.DataFrame(mix_tbl_15).round(2).to_string(index=False))
    for k, oct_mask in exceed_mix_15:
        print(f"  >>> Printing October mixed-event (pf>=0.15) rows for {LABELS[k]} (Oct_r100k > 2x max train month):")
        for r_i in np.flatnonzero(oct_mask):
            r = df.iloc[r_i]
            print(
                f"      {r.txn_id} | {r.timestamp} | comp={comp_m15[r_i]} | pred={LABELS[int(r.pred_raw)]} pf={r.pf_final:.3f} | "
                f"snd={r.sender_id} rcv={r.receiver_id}({r.receiver_type}:{r.merchant_category}) dev={r.device_id} | {r.txn_type} amt={r.amount:.2f} bal={r.balance_before:.2f}"
            )


# ==============================================================================
# SEARCH 3: A tell for sibling-shaped events, using band devices as truth
# ==============================================================================
def run_search_3(df):
    print("\n" + "=" * 80)
    print("SEARCH 3: TELL FOR SIBLING-SHAPED EVENTS USING BAND DEVICES AS TRUTH")
    print("=" * 80)

    is_tr = (df.split == "train").values
    y_raw = df.label.map({lab: i for i, lab in enumerate(LABELS)}).fillna(-1).astype(int).values
    y_clean_tr = np.load(ROOT / "models" / "oof_raw" / "y_clean.npy")
    y_clean = np.full(len(df), -1, dtype=int)
    y_clean[is_tr] = y_clean_tr

    # Join extra account and merchant attributes on both sender and receiver
    ac = pd.read_parquet(ROOT / "data" / "interim" / "accounts.parquet").set_index("account_id")
    me = pd.read_parquet(ROOT / "data" / "interim" / "merchants.parquet").set_index("merchant_id")

    p2p = (df.receiver_type.values == "P2P")
    df["rcv_os"] = df.receiver_id.map(ac.os).where(p2p)
    df["rcv_bank_code"] = df.receiver_id.map(ac.bank_code).where(p2p)
    df["snd_num"] = df.sender_id.str[2:].astype(np.int64)
    df["rcv_num"] = df.receiver_id.str[2:].astype(np.int64)

    # Timestamp second-level and position-in-month features
    ts = pd.to_datetime(df.timestamp)
    df["sec_of_min"] = ts.dt.second.values
    df["min_of_hour"] = ts.dt.minute.values
    df["day_of_month"] = ts.dt.day.values
    # Fractional position in month (0..1)
    m_start = pd.to_datetime(df.m_clean + "-01")
    days_in_m = m_start.dt.days_in_month * 86400.0
    df["pos_in_month"] = ((ts - m_start).dt.total_seconds() / days_in_m).clip(0.0, 1.0).values

    # City comparisons
    df["dev_eq_home_city"] = (df.device_city == df.home_city).astype(int)
    df["dev_eq_rcv_city"] = np.where(p2p, df.device_city == df.rcv_home_city, df.device_city == df.mer_city).astype(int)
    df["home_eq_rcv_city"] = np.where(p2p, df.home_city == df.rcv_home_city, df.home_city == df.mer_city).astype(int)
    df["same_os"] = np.where(p2p, df.os == df.rcv_os, False).astype(int)
    df["same_bank"] = np.where(p2p, df.bank_code == df.rcv_bank_code, False).astype(int)
    df["same_kyc"] = np.where(p2p, df.kyc_level == df.rcv_kyc_level, False).astype(int)
    df["acct_age_diff"] = (df.sender_account_age_days - df.acct_age_from_opened).values

    # Reconstruct the 5 event families across the dataset (using label-free physical sessions)
    t_sec = df.t.values.astype(np.int64)
    amt = df.amount.values.astype(np.float64)
    a_int = df.amt_is_integer.values == 1
    pay = df.txn_type.values == "PAY"
    coll = df.txn_type.values == "COLLECT"
    newp = df.new_payee.values == 1
    band = df.band.values.astype(int)

    def make_sessions(mask, key_arr, max_gap):
        idx = np.flatnonzero(mask)
        ord_i = idx[np.lexsort((t_sec[idx], key_arr[idx]))]
        starts = np.r_[True, (key_arr[ord_i[1:]] != key_arr[ord_i[:-1]]) | ((t_sec[ord_i[1:]] - t_sec[ord_i[:-1]]) > max_gap)]
        sids = np.cumsum(starts)
        row_sid = np.full(len(df), -1, dtype=np.int64)
        row_sid[ord_i] = sids
        return row_sid

    # Family 1: Drain sessions (whole-rupee P2P PAY, new payee, same sender, 2h gap)
    m_drain = p2p & pay & a_int & newp
    sid_drain = make_sessions(m_drain, df.snd_num.values, 7200)

    # Family 2: Mule-shaped fan-in sessions (whole-rupee P2P PAY, new payee, mult of 10, same receiver, 24h gap)
    m_fanin = m_drain & ((amt % 10) == 0)
    sid_fanin = make_sessions(m_fanin, df.rcv_num.values, 86400)

    # Family 3: Layering-shaped chains (connected components of 2h 0.90-1.00 P2P integer PAY hops)
    m_ch_cand = p2p & pay & a_int & (amt >= 500)
    x_c = df.loc[m_ch_cand, ["sender_id", "receiver_id", "t", "amount"]].copy()
    x_c["row"] = x_c.index.values
    q_c = x_c[["row", "sender_id", "t", "amount"]].rename(columns={"sender_id": "key"}).sort_values("t")
    src_c = x_c[["row", "receiver_id", "t", "amount"]].rename(columns={"row": "parent", "receiver_id": "key", "t": "pt", "amount": "pa"}).sort_values("pt")
    m_asof = pd.merge_asof(q_c, src_c, left_on="t", right_on="pt", by="key", tolerance=7200, direction="backward", allow_exact_matches=False)
    rat_c = m_asof.amount / m_asof.pa
    val_c = m_asof.loc[rat_c.between(0.90, 1.00), ["row", "parent"]].astype(np.int64)
    G_ch = coo_matrix((np.ones(len(val_c), dtype=np.int8), (val_c.row.values, val_c.parent.values)), shape=(len(df), len(df)))
    _, comp_ch = connected_components(G_ch, directed=False)
    ch_sizes = np.bincount(comp_ch)
    sid_chain = np.where(ch_sizes[comp_ch] >= 2, comp_ch, -1)

    # Family 4: Collect-shaped events (P2P COLLECT with is_1p1 == 1 or amt_just_under_1000s == 1 or amt_ends_99 == 1, grouped by receiver 24h or pair 7d)
    m_coll_ev = p2p & coll & ((df.is_1p1.values == 1) | (df.amt_just_under_1000s.values == 1) | (df.amt_ends_99.values == 1))
    sid_collect = make_sessions(m_coll_ev, df.rcv_num.values, 86400)

    # Family 5: Fixed-amount merchant sessions (S_flag == 1, same merchant receiver, 24h gap)
    m_shop = df.S_flag.values == 1
    sid_shop = make_sessions(m_shop, df.rcv_num.values, 86400)

    def extract_event_features(sid_arr, min_rows=1, extra_row_filter=None):
        """Build an event-level DataFrame of candidate tells (excluding band-device rows when computing non-band features if needed, or across the event)."""
        valid = (sid_arr >= 0) & is_tr
        if extra_row_filter is not None:
            valid = valid & extra_row_filter
        sub_idxs = np.flatnonzero(valid)
        sids = sid_arr[sub_idxs]

        # Group indices by sid
        order = np.argsort(sids, kind="stable")
        s_sorted = sids[order]
        i_sorted = sub_idxs[order]
        unq_s, first_p, cnts = np.unique(s_sorted, return_index=True, return_counts=True)

        records = []
        for u_sid, fp, c_len in zip(unq_s, first_p, cnts):
            if c_len < min_rows:
                continue
            r_idxs = i_sorted[fp : fp + c_len]
            # Sort chronologically within event
            r_idxs = r_idxs[np.argsort(t_sec[r_idxs], kind="stable")]

            b_vals = band[r_idxs]
            b_pos = b_vals[b_vals > 0]
            ev_band = int(b_pos[0]) if (len(b_pos) > 0 and b_pos.min() == b_pos.max()) else 0

            # Majority raw and clean labels on the event
            yr = y_raw[r_idxs]
            yc = y_clean[r_idxs]
            yr_pos = yr[yr > 0]
            ev_y_raw = int(pd.Series(yr_pos).mode().iloc[0]) if len(yr_pos) > 0 else 0
            ev_y_clean = int(pd.Series(yc[yc > 0]).mode().iloc[0]) if (yc > 0).any() else 0

            m_str = df.m_clean.iat[r_idxs[0]]
            amts = amt[r_idxs]
            ts_ev = t_sec[r_idxs]
            secs = df.sec_of_min.values[r_idxs]
            mins = df.min_of_hour.values[r_idxs]

            # 1. Order of amounts inside the event
            if len(amts) >= 2:
                diffs_a = np.diff(amts)
                amt_inc = float((diffs_a > 0).mean())
                amt_dec = float((diffs_a < 0).mean())
                amt_strictly_inc = int(np.all(diffs_a > 0))
                amt_strictly_dec = int(np.all(diffs_a < 0))
                amt_first_gt_last = int(amts[0] > amts[-1])
                amt_max_first = int(np.argmax(amts) == 0)
                amt_max_last = int(np.argmax(amts) == len(amts) - 1)
                amt_min_first = int(np.argmin(amts) == 0)
                amt_min_last = int(np.argmin(amts) == len(amts) - 1)
                if np.std(amts) > 1e-9:
                    amt_rho = float(np.corrcoef(np.arange(len(amts)), amts)[0, 1])
                else:
                    amt_rho = 0.0
            else:
                amt_inc = amt_dec = amt_rho = 0.0
                amt_strictly_inc = amt_strictly_dec = amt_first_gt_last = 0
                amt_max_first = amt_max_last = amt_min_first = amt_min_last = 0

            # 2. Second-level timestamp structure
            sec_zero_share = float((secs == 0).mean())
            sec_mod5_share = float(((secs % 5) == 0).mean())
            sec_mod10_share = float(((secs % 10) == 0).mean())
            sec_even_share = float(((secs % 2) == 0).mean())
            sec_mean = float(secs.mean())
            sec_std = float(secs.std()) if len(secs) >= 2 else 0.0
            min_mean = float(mins.mean())
            if len(ts_ev) >= 2:
                dt_ev = np.diff(ts_ev)
                dt_mod60_share = float(((dt_ev % 60) == 0).mean())
                dt_mod10_share = float(((dt_ev % 10) == 0).mean())
                dt_even_share = float(((dt_ev % 2) == 0).mean())
            else:
                dt_mod60_share = dt_mod10_share = dt_even_share = 0.0

            # 3. Sender and receiver account attributes (from accounts.csv)
            r0 = r_idxs[0]
            snd_kyc_full = float((df.kyc_level.values[r_idxs] == "FULL").mean())
            snd_kyc_min = float((df.kyc_level.values[r_idxs] == "MIN").mean())
            rcv_kyc_full = float((df.rcv_kyc_level.values[r_idxs] == "FULL").mean())
            snd_os_ios = float((df.os.values[r_idxs] == "iOS").mean())
            rcv_os_ios = float((df.rcv_os.values[r_idxs] == "iOS").mean())
            same_os_share = float(df.same_os.values[r_idxs].mean())
            same_bank_share = float(df.same_bank.values[r_idxs].mean())
            same_kyc_share = float(df.same_kyc.values[r_idxs].mean())
            snd_age_mean = float(df.acct_age_from_opened.values[r_idxs].mean())
            rcv_age_mean = float(np.nanmean(df.rcv_acct_age_days.values[r_idxs])) if np.isfinite(df.rcv_acct_age_days.values[r_idxs]).any() else -1.0
            snd_lt_rcv_age = float((df.acct_age_from_opened.values[r_idxs] < np.nan_to_num(df.rcv_acct_age_days.values[r_idxs], nan=-1)).mean())
            snd_id_mean = float(df.snd_num.values[r_idxs].mean())
            rcv_id_mean = float(df.rcv_num.values[r_idxs].mean())
            snd_lt_rcv_id = float((df.snd_num.values[r_idxs] < df.rcv_num.values[r_idxs]).mean())
            age_diff_nonzero = float((df.acct_age_diff.values[r_idxs] != 0).mean())

            # 4. Merchant attributes (from merchants.csv)
            mer_age = float(np.nanmean(df.mer_age_days.values[r_idxs])) if np.isfinite(df.mer_age_days.values[r_idxs]).any() else -1.0
            mer_cat_match = float(np.nanmean(df.mer_cat_matches_txn.values[r_idxs])) if np.isfinite(df.mer_cat_matches_txn.values[r_idxs]).any() else -1.0

            # 5. device_city against home_city and receiver city
            dev_eq_home = float(df.dev_eq_home_city.values[r_idxs].mean())
            dev_eq_rcv = float(df.dev_eq_rcv_city.values[r_idxs].mean())
            home_eq_rcv = float(df.home_eq_rcv_city.values[r_idxs].mean())

            # 6. Position in the month
            pos_m = float(df.pos_in_month.values[r_idxs].mean())
            day_m = float(df.day_of_month.values[r_idxs].mean())
            is_first_half_m = int(day_m <= 15.0)
            is_weekend = float((df.weekday.values[r_idxs] >= 5).mean())

            records.append({
                "sid": int(u_sid),
                "month": m_str,
                "n_rows": int(c_len),
                "ev_band": ev_band,
                "ev_y_raw": ev_y_raw,
                "ev_y_clean": ev_y_clean,
                # Family 1: Amount order
                "amt_inc": amt_inc,
                "amt_dec": amt_dec,
                "amt_strictly_inc": amt_strictly_inc,
                "amt_strictly_dec": amt_strictly_dec,
                "amt_first_gt_last": amt_first_gt_last,
                "amt_max_first": amt_max_first,
                "amt_max_last": amt_max_last,
                "amt_min_first": amt_min_first,
                "amt_min_last": amt_min_last,
                "amt_rho": amt_rho,
                # Family 2: Second-level timestamp
                "sec_zero_share": sec_zero_share,
                "sec_mod5_share": sec_mod5_share,
                "sec_mod10_share": sec_mod10_share,
                "sec_even_share": sec_even_share,
                "sec_mean": sec_mean,
                "sec_std": sec_std,
                "min_mean": min_mean,
                "dt_mod60_share": dt_mod60_share,
                "dt_mod10_share": dt_mod10_share,
                "dt_even_share": dt_even_share,
                # Family 3: Account attributes
                "snd_kyc_full": snd_kyc_full,
                "snd_kyc_min": snd_kyc_min,
                "rcv_kyc_full": rcv_kyc_full,
                "snd_os_ios": snd_os_ios,
                "rcv_os_ios": rcv_os_ios,
                "same_os_share": same_os_share,
                "same_bank_share": same_bank_share,
                "same_kyc_share": same_kyc_share,
                "snd_age_mean": snd_age_mean,
                "rcv_age_mean": rcv_age_mean,
                "snd_lt_rcv_age": snd_lt_rcv_age,
                "snd_id_mean": snd_id_mean,
                "rcv_id_mean": rcv_id_mean,
                "snd_lt_rcv_id": snd_lt_rcv_id,
                "age_diff_nonzero": age_diff_nonzero,
                # Family 4: Merchant attributes
                "mer_age": mer_age,
                "mer_cat_match": mer_cat_match,
                # Family 5: City relations
                "dev_eq_home": dev_eq_home,
                "dev_eq_rcv": dev_eq_rcv,
                "home_eq_rcv": home_eq_rcv,
                # Family 6: Position in month
                "pos_m": pos_m,
                "day_m": day_m,
                "is_first_half_m": is_first_half_m,
                "is_weekend": is_weekend,
            })
        return pd.DataFrame(records)

    TELL_COLS = [
        "amt_inc", "amt_dec", "amt_strictly_inc", "amt_strictly_dec", "amt_first_gt_last",
        "amt_max_first", "amt_max_last", "amt_min_first", "amt_min_last", "amt_rho",
        "sec_zero_share", "sec_mod5_share", "sec_mod10_share", "sec_even_share", "sec_mean",
        "sec_std", "min_mean", "dt_mod60_share", "dt_mod10_share", "dt_even_share",
        "snd_kyc_full", "snd_kyc_min", "rcv_kyc_full", "snd_os_ios", "rcv_os_ios",
        "same_os_share", "same_bank_share", "same_kyc_share", "snd_age_mean", "rcv_age_mean",
        "snd_lt_rcv_age", "snd_id_mean", "rcv_id_mean", "snd_lt_rcv_id", "age_diff_nonzero",
        "mer_age", "mer_cat_match", "dev_eq_home", "dev_eq_rcv", "home_eq_rcv",
        "pos_m", "day_m", "is_first_half_m", "is_weekend",
    ]

    def evaluate_sibling_pair(pair_name, ev_df, cls_a, cls_b, extra_band_mask=None, extra_nonband_mask=None):
        """
        Test all 44 candidate features (and threshold cuts) on band events (ev_band in {cls_a, cls_b})
        and on non-band events (ev_band == 0 & ev_y_raw in {cls_a, cls_b}).
        Check if any feature/rule achieves >= 85% separation on band events in each of Jul, Aug, Sep
        AND agrees with labels on non-band events.
        """
        b_mask = ev_df["ev_band"].isin([cls_a, cls_b])
        if extra_band_mask is not None:
            b_mask = b_mask & extra_band_mask
        nb_mask = (ev_df["ev_band"] == 0) & ev_df["ev_y_raw"].isin([cls_a, cls_b])
        if extra_nonband_mask is not None:
            nb_mask = nb_mask & extra_nonband_mask

        b_df = ev_df[b_mask].copy()
        nb_df = ev_df[nb_mask].copy()

        print(f"\n--- {pair_name}: {LABELS[cls_a]} ({cls_a}) vs {LABELS[cls_b]} ({cls_b}) ---")
        print(
            f"  Band events: total={len(b_df)} ({LABELS[cls_a]}={int((b_df.ev_band==cls_a).sum())}, {LABELS[cls_b]}={int((b_df.ev_band==cls_b).sum())}) | "
            f"Non-band labelled events: total={len(nb_df)} ({LABELS[cls_a]}={int((nb_df.ev_y_raw==cls_a).sum())}, {LABELS[cls_b]}={int((nb_df.ev_y_raw==cls_b).sum())})"
        )

        # For each feature in TELL_COLS, test thresholds (quantiles + 0.5) in both directions (>= thr -> cls_a vs < thr -> cls_a)
        best_rows = []
        n_tested = 0
        for col in TELL_COLS:
            vb = b_df[col].values
            if np.all(vb == vb[0]):
                continue
            thrs = np.unique(np.quantile(vb, [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]))
            for thr in thrs:
                for direc in (">=", "<="):
                    n_tested += 1
                    pred_b = np.where((vb >= thr) if direc == ">=" else (vb <= thr), cls_a, cls_b)
                    # Balanced accuracy and overall accuracy per month on band events
                    m_accs = []
                    m_bal_accs = []
                    for m_c in ["2026-07", "2026-08", "2026-09"]:
                        mm = b_df["month"].values == m_c
                        if mm.sum() == 0:
                            m_accs.append(0.0)
                            m_bal_accs.append(0.0)
                            continue
                        yt_m = b_df["ev_band"].values[mm]
                        yp_m = pred_b[mm]
                        acc_m = float((yt_m == yp_m).mean())
                        ra = float((yp_m[yt_m == cls_a] == cls_a).mean()) if (yt_m == cls_a).any() else 0.0
                        rb = float((yp_m[yt_m == cls_b] == cls_b).mean()) if (yt_m == cls_b).any() else 0.0
                        m_accs.append(acc_m)
                        m_bal_accs.append(0.5 * (ra + rb))

                    # Also evaluate on non-band events
                    if len(nb_df) > 0:
                        vnb = nb_df[col].values
                        pred_nb = np.where((vnb >= thr) if direc == ">=" else (vnb <= thr), cls_a, cls_b)
                        yt_nb = nb_df["ev_y_raw"].values
                        nb_acc = float((yt_nb == pred_nb).mean())
                        nb_ra = float((pred_nb[yt_nb == cls_a] == cls_a).mean()) if (yt_nb == cls_a).any() else 0.0
                        nb_rb = float((pred_nb[yt_nb == cls_b] == cls_b).mean()) if (yt_nb == cls_b).any() else 0.0
                        nb_bal_acc = 0.5 * (nb_ra + nb_rb)
                    else:
                        nb_acc = nb_bal_acc = 0.0

                    best_rows.append({
                        "feature": col,
                        "rule": f"{col} {direc} {thr:.4g} -> {LABELS[cls_a]}",
                        "band_Jul_acc": m_accs[0],
                        "band_Aug_acc": m_accs[1],
                        "band_Sep_acc": m_accs[2],
                        "band_min_m_acc": min(m_accs),
                        "band_min_m_bal": min(m_bal_accs),
                        "nonband_acc": nb_acc,
                        "nonband_bal": nb_bal_acc,
                    })

        res_df = pd.DataFrame(best_rows)
        pass_85_bal = res_df[(res_df["band_min_m_bal"] >= 0.85) & (res_df["nonband_bal"] >= 0.60)]
        pass_85_raw = res_df[(res_df["band_min_m_acc"] >= 0.85) & (res_df["nonband_bal"] >= 0.60)]
        print(f"  Total candidate rules tested: {n_tested} | Rules passing >=85% monthly BALANCED separation + non-band check: {len(pass_85_bal)} (raw imbalanced >=85%: {len(pass_85_raw)})")
        print("  Top 5 candidate rules by min monthly balanced accuracy on band events:")
        print(res_df.sort_values("band_min_m_bal", ascending=False).head(5).round(3).to_string(index=False))
        print("  Top 5 candidate rules by non-band balanced accuracy:")
        print(res_df.sort_values("nonband_bal", ascending=False).head(5).round(3).to_string(index=False))

    # Build event tables and evaluate all pairs
    # 1. Drain events (x0.25 drains on fresh SIM < 72h, and all x0.25 drains)
    # Note: add event-level indicator of x0.25 and fresh SIM
    ev_drain = extract_event_features(sid_drain, min_rows=2)
    # Check x0.25 drain share and fresh SIM in each drain session
    d_x025 = df.groupby(sid_drain)["snd_drain_kind"].apply(lambda s: (s == 1).any())
    d_fsim = df.groupby(sid_drain)["sim_age_hours"].apply(lambda s: (s < 72).all())
    d_home = df.groupby(sid_drain)["home_device"].apply(lambda s: (s == 1).all())
    ev_drain["is_x025"] = ev_drain["sid"].map(d_x025).fillna(False).values
    ev_drain["is_fsim"] = ev_drain["sid"].map(d_fsim).fillna(False).values
    ev_drain["all_home"] = ev_drain["sid"].map(d_home).fillna(False).values

    evaluate_sibling_pair(
        "Pair 1: x0.25 drains (sim_swap vs account_takeover)",
        ev_drain, 4, 5,
        extra_band_mask=ev_drain["is_x025"].values,
        extra_nonband_mask=(ev_drain["is_x025"] & ev_drain["is_fsim"] & ev_drain["all_home"]).values,
    )

    # 2. Mule-shaped fan-ins (velocity_bot band 2 vs mule_account band 1)
    ev_fanin = extract_event_features(sid_fanin, min_rows=3)
    evaluate_sibling_pair("Pair 2: Mule-shaped fan-ins (mule_account vs velocity_bot)", ev_fanin, 1, 2)

    # 3. Layering-shaped chains (layering_chain band 3 vs mule_account band 1)
    ev_chain = extract_event_features(sid_chain, min_rows=2)
    evaluate_sibling_pair("Pair 3: Layering-shaped chains (layering_chain vs mule_account)", ev_chain, 3, 1)

    # 4. Collect-shaped events (phishing_collect band 6 vs fake_cashback band 9 and loan_app_extortion band 10)
    ev_coll = extract_event_features(sid_collect, min_rows=1)
    evaluate_sibling_pair("Pair 4a: Collect-shaped events (phishing_collect vs fake_cashback)", ev_coll, 6, 9)
    evaluate_sibling_pair("Pair 4b: Collect-shaped events (phishing_collect vs loan_app_extortion)", ev_coll, 6, 10)

    # 5. Fixed-amount merchant sessions (qr_swap band 7 vs refund_abuse band 8 vs merchant_collusion band 11)
    ev_shop = extract_event_features(sid_shop, min_rows=1)
    evaluate_sibling_pair("Pair 5a: Fixed-amount merchant sessions (qr_swap vs refund_abuse)", ev_shop, 7, 8)
    evaluate_sibling_pair("Pair 5b: Fixed-amount merchant sessions (qr_swap vs merchant_collusion)", ev_shop, 7, 11)
    evaluate_sibling_pair(
        "Pair 5b (multi-row n_rows>=2, removing session-length proxy)",
        ev_shop, 7, 11,
        extra_band_mask=(ev_shop["n_rows"] >= 2).values,
        extra_nonband_mask=(ev_shop["n_rows"] >= 2).values,
    )
    evaluate_sibling_pair("Pair 5c: Fixed-amount merchant sessions (refund_abuse vs merchant_collusion)", ev_shop, 8, 11)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", type=int, default=0)
    args = ap.parse_args()
    df_all = load_all_base()
    if args.search in (0, 1):
        run_search_1(df_all)
    if args.search in (0, 2):
        run_search_2(df_all)
    if args.search in (0, 3):
        run_search_3(df_all)
