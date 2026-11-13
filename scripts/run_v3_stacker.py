from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from upi.artifacts import write_json, write_submission
from upi.constants import LABELS, LABEL_TO_ID
from upi.metrics import apply_weights, classification_report_dict, fit_class_weights, macro_f1_ids
from upi.stage2 import CHANNEL_NAMES, compute_neighbour_channels


CONTEXT_COLS = [
    "receiver_cnt_1h", "receiver_cnt_24h", "receiver_cnt_7d",
    "receiver_unique_senders_day", "receiver_new_payee_share_day",
    "device_cnt_5m", "device_cnt_15m", "device_cnt_1h", "device_cnt_24h",
    "device_unique_accounts_total", "device_unique_accounts_day",
    "device_int_amt_cnt_1h", "device_int_amt_share_1h",
    "sender_cnt_1h", "sender_cnt_24h", "sender_drain_ratio_1h", "sender_drain_ratio_24h",
    "sender_500_cnt_24h", "sender_pin_retry_sum_24h", "sender_new_payee_cnt_24h",
    "pair_cnt_24h", "pair_cnt_7d", "pair_cnt_30d", "pair_round500_cnt_30d", "pair_round500_cnt_total",
    "evening_pair_cnt_7d", "evening_pair_cnt_30d", "pair_evening_share_30d", "snd_lending_cnt_30d",
    "reverse_pair_cnt_24h", "reverse_pair_cnt_7d", "rev_pair_cnt_total",
    "rev_payback_ratio_24h", "rev_bait_ratio_24h",
    "snd_refund_cnt_30d", "snd_refund_cnt_total", "snd_refund_share_30d", "is_fractional_refund",
    "flow_forward_compatible", "flow_previous_compatible",
    "forward_amount_ratio_30m", "previous_in_amount_ratio_30m",
    "sim_age_hours", "device_age_days", "home_device", "away_from_home",
    "new_payee", "pin_retries", "login_to_txn_sec", "amt_to_balance", "residual_balance_frac",
    "amt_is_integer", "amt_round_500", "amt_round_1000", "amt_just_under_1000", "amt_tiny_le10",
    "qr_vpa_mismatch", "qr_p2p_flag", "is_late_evening", "is_night",
    "sig_hit_count", "rule_hit_count",
    "rule_refund_abuse", "rule_layering", "rule_fake_cashback", "rule_merchant_collusion",
    "rule_mule", "rule_phishing", "rule_qr_swap", "rule_velocity_bot",
    "rule_sim_swap", "rule_ato", "rule_loan_extortion",
]


def build_meta_features(
    prob_list: list[np.ndarray],
    model_names: list[str],
    channels: np.ndarray,
    context_df: pd.DataFrame,
) -> pd.DataFrame:
    p_stack = np.stack(prob_list, axis=0)  # (M, N, 12)
    p_mean = p_stack.mean(axis=0)
    p_max = p_stack.max(axis=0)
    p_min = p_stack.min(axis=0)
    p_std = p_stack.std(axis=0)

    feats: dict[str, np.ndarray] = {}
    for m_idx, m_name in enumerate(model_names):
        pm = prob_list[m_idx]
        for c, lab in enumerate(LABELS):
            feats[f"p_{m_name}_{lab}"] = pm[:, c].astype(np.float32)

    for c, lab in enumerate(LABELS):
        feats[f"p_mean_{lab}"] = p_mean[:, c].astype(np.float32)
        feats[f"p_max_{lab}"] = p_max[:, c].astype(np.float32)
        feats[f"p_min_{lab}"] = p_min[:, c].astype(np.float32)
        feats[f"p_std_{lab}"] = p_std[:, c].astype(np.float32)

    fraud_probs = p_mean[:, 1:]
    feats["p_fraud_sum"] = (1.0 - p_mean[:, 0]).astype(np.float32)
    feats["p_fraud_max"] = fraud_probs.max(axis=1).astype(np.float32)
    part = np.partition(fraud_probs, -2, axis=1)
    feats["p_fraud_second"] = part[:, -2].astype(np.float32)
    feats["p_fraud_margin"] = (feats["p_fraud_max"] - feats["p_fraud_second"]).astype(np.float32)
    feats["p_fraud_ratio"] = (feats["p_fraud_max"] / np.maximum(feats["p_fraud_second"], 1e-6)).astype(np.float32)
    clipped = np.clip(p_mean, 1e-7, 1.0)
    feats["p_entropy"] = (-np.sum(clipped * np.log(clipped), axis=1)).astype(np.float32)

    pairs = [
        ("sim_swap", "account_takeover"),
        ("mule_account", "layering_chain"),
        ("phishing_collect", "fake_cashback"),
        ("qr_swap", "refund_abuse"),
        ("qr_swap", "merchant_collusion"),
        ("mule_account", "velocity_bot"),
        ("merchant_collusion", "refund_abuse"),
    ]
    for a, b in pairs:
        ia, ib = LABEL_TO_ID[a], LABEL_TO_ID[b]
        pa, pb = p_mean[:, ia], p_mean[:, ib]
        feats[f"logodds_{a}_vs_{b}"] = np.log((pa + 1e-5) / (pb + 1e-5)).astype(np.float32)
        feats[f"sum_{a}_{b}"] = (pa + pb).astype(np.float32)
        feats[f"share_{a}_in_{b}"] = (pa / np.maximum(pa + pb, 1e-6)).astype(np.float32)

    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        ch_p = channels[:, ch_idx, :]
        feats[f"nb_{ch_name}_anyfraud"] = ch_p[:, 1:].sum(axis=1).astype(np.float32)
        feats[f"nb_{ch_name}_maxfraud"] = ch_p[:, 1:].max(axis=1).astype(np.float32)
        for c in range(1, len(LABELS)):
            lab = LABELS[c]
            feats[f"nb_{ch_name}_{lab}"] = ch_p[:, c].astype(np.float32)
            feats[f"ep_{ch_name}_{lab}"] = (0.5 * p_mean[:, c] + 0.5 * ch_p[:, c]).astype(np.float32)

    for col in CONTEXT_COLS:
        if col in context_df.columns:
            feats[f"ctx_{col}"] = context_df[col].to_numpy(dtype=np.float32)

    if "pair_round500_cnt_30d" in context_df.columns and "pair_cnt_30d" in context_df.columns:
        c500 = context_df["pair_round500_cnt_30d"].to_numpy(dtype=np.float32)
        c30 = np.maximum(context_df["pair_cnt_30d"].to_numpy(dtype=np.float32), 1.0)
        feats["ctx_pair_round500_share_30d"] = (c500 / c30).astype(np.float32)

    return pd.DataFrame(feats)


def align_probabilities(runs: list[Path]) -> tuple[pd.DataFrame, list[np.ndarray], pd.DataFrame, list[np.ndarray]]:
    ref_val = pd.read_parquet(runs[0] / "validation_predictions.parquet")
    ref_val_ids = ref_val.txn_id.tolist()
    val_blocks = []
    for r in runs:
        rows = pd.read_parquet(r / "validation_predictions.parquet")
        probs = np.load(r / "validation_probabilities.npy")
        pos = pd.Series(np.arange(len(rows)), index=rows.txn_id).loc[ref_val_ids].to_numpy()
        val_blocks.append(probs[pos])

    ref_test = pd.read_parquet(runs[0] / "test_ids.parquet")
    ref_test_ids = ref_test.txn_id.tolist()
    test_blocks = []
    for r in runs:
        rows = pd.read_parquet(r / "test_ids.parquet")
        probs = np.load(r / "test_probabilities.npy")
        pos = pd.Series(np.arange(len(rows)), index=rows.txn_id).loc[ref_test_ids].to_numpy()
        test_blocks.append(probs[pos])

    return ref_val, val_blocks, ref_test, test_blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", default="submissions/v3_stacker_blend.csv")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    run_paths = [Path(r) for r in args.runs]
    model_names = [f"m{i}_{p.name.split('-')[0]}" for i, p in enumerate(run_paths)]
    print("Stacking runs:", [str(p) for p in run_paths])

    ref_val, val_blocks, ref_test, test_blocks = align_probabilities(run_paths)

    table = pd.read_parquet("data/processed/features_engineered_v2.parquet")
    sep = table[table.split.eq("train") & table.timestamp.ge("2026-09-01") & table.timestamp.lt("2026-10-01")].copy()
    val_context = pd.concat([
        sep.loc[sep.timestamp.lt("2026-09-24")],
        sep.loc[sep.timestamp.ge("2026-09-24")],
    ], axis=0).reset_index(drop=True)
    # Reorder val_context to match ref_val
    val_pos = pd.Series(np.arange(len(val_context)), index=val_context.txn_id).loc[ref_val.txn_id.tolist()].to_numpy()
    val_context = val_context.iloc[val_pos].reset_index(drop=True)

    test_context = table[table.split.eq("test")].copy().reset_index(drop=True)
    test_pos = pd.Series(np.arange(len(test_context)), index=test_context.txn_id).loc[ref_test.txn_id.tolist()].to_numpy()
    test_context = test_context.iloc[test_pos].reset_index(drop=True)

    y_val = ref_val.label.map(LABEL_TO_ID).to_numpy(dtype=np.int32)
    cal = pd.to_datetime(ref_val.timestamp).lt("2026-09-24").to_numpy()
    ass = ~cal

    p_val_s1 = sum(val_blocks) / len(val_blocks)
    p_test_s1 = sum(test_blocks) / len(test_blocks)

    fine_grid = np.geomspace(0.25, 35.0, 65)
    w_s1 = fit_class_weights(p_val_s1[cal], y_val[cal], grid=fine_grid, sweeps=5)
    print(f"Stage-1 {len(run_paths)}-Model Equal Blend | Cal F1: {macro_f1_ids(y_val[cal], apply_weights(p_val_s1[cal], w_s1)):.4f} | Assess F1: {macro_f1_ids(y_val[ass], apply_weights(p_val_s1[ass], w_s1)):.4f}")

    print("Computing 7-channel neighbour probabilities (val & test)...")
    val_channels = compute_neighbour_channels(val_context, p_val_s1)
    test_channels = compute_neighbour_channels(test_context, p_test_s1)

    print("Building Stage-2 meta-feature matrices...")
    X_val_meta = build_meta_features(val_blocks, model_names, val_channels, val_context)
    X_test_meta = build_meta_features(test_blocks, model_names, test_channels, test_context)
    print("Meta shape:", X_val_meta.shape, X_test_meta.shape)

    cal_idx = np.where(cal)[0]
    ass_idx = np.where(ass)[0]
    senders_cal = pd.factorize(val_context.loc[cal, "sender_id"])[0]
    folds = senders_cal % args.folds

    oof_lgb = np.zeros((len(cal_idx), 12), dtype=np.float64)
    ass_lgb = np.zeros((len(ass_idx), 12), dtype=np.float64)
    test_lgb = np.zeros((len(X_test_meta), 12), dtype=np.float64)

    oof_xgb = np.zeros((len(cal_idx), 12), dtype=np.float64)
    ass_xgb = np.zeros((len(ass_idx), 12), dtype=np.float64)
    test_xgb = np.zeros((len(X_test_meta), 12), dtype=np.float64)

    for fold in range(args.folds):
        tr_m = folds != fold
        va_m = folds == fold

        m_lgb = LGBMClassifier(
            objective="multiclass", num_class=12, n_estimators=500,
            learning_rate=0.03, num_leaves=31, min_child_samples=100,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.55,
            reg_lambda=12.0, reg_alpha=2.0, n_jobs=-1,
            random_state=202611 + fold, verbose=-1,
        )
        m_lgb.fit(
            X_val_meta.iloc[cal_idx[tr_m]], y_val[cal_idx[tr_m]],
            eval_set=[(X_val_meta.iloc[cal_idx[va_m]], y_val[cal_idx[va_m]])],
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )
        oof_lgb[va_m] = m_lgb.predict_proba(X_val_meta.iloc[cal_idx[va_m]])
        ass_lgb += m_lgb.predict_proba(X_val_meta.iloc[ass_idx]) / args.folds
        test_lgb += m_lgb.predict_proba(X_test_meta) / args.folds

        m_xgb = XGBClassifier(
            objective="multi:softprob", tree_method="hist", device="cuda",
            n_estimators=500, learning_rate=0.03, max_depth=5,
            min_child_weight=30, subsample=0.8, colsample_bytree=0.55,
            reg_lambda=15.0, reg_alpha=2.0, random_state=202611 + fold,
            eval_metric="mlogloss", early_stopping_rounds=40,
        )
        m_xgb.fit(
            X_val_meta.iloc[cal_idx[tr_m]], y_val[cal_idx[tr_m]],
            eval_set=[(X_val_meta.iloc[cal_idx[va_m]], y_val[cal_idx[va_m]])],
            verbose=False,
        )
        oof_xgb[va_m] = m_xgb.predict_proba(X_val_meta.iloc[cal_idx[va_m]])
        ass_xgb += m_xgb.predict_proba(X_val_meta.iloc[ass_idx]) / args.folds
        test_xgb += m_xgb.predict_proba(X_test_meta) / args.folds
        print(f"  Fold {fold}: lgb_iter={m_lgb.best_iteration_}, xgb_iter={m_xgb.best_iteration}")

    oof_s2 = 0.5 * oof_lgb + 0.5 * oof_xgb
    ass_s2 = 0.5 * ass_lgb + 0.5 * ass_xgb
    test_s2 = 0.5 * test_lgb + 0.5 * test_xgb

    best_alpha, best_cal_f1, best_w = 0.0, -1.0, w_s1
    for alpha in (0.0, 0.15, 0.25, 0.35, 0.50):
        c_cal = (1.0 - alpha) * p_val_s1[cal] + alpha * oof_s2
        c_ass = (1.0 - alpha) * p_val_s1[ass] + alpha * ass_s2
        w_c = fit_class_weights(c_cal, y_val[cal], grid=fine_grid, sweeps=5)
        f1_c = macro_f1_ids(y_val[cal], apply_weights(c_cal, w_c))
        f1_a = macro_f1_ids(y_val[ass], apply_weights(c_ass, w_c))
        print(f"  alpha_s2={alpha:.2f} | Cal F1: {f1_c:.4f} | Assess F1: {f1_a:.4f}")
        if f1_c > best_cal_f1:
            best_cal_f1 = f1_c
            best_alpha = alpha
            best_w = w_c

    final_val_p = np.zeros_like(p_val_s1)
    final_val_p[cal] = (1.0 - best_alpha) * p_val_s1[cal] + best_alpha * oof_s2
    final_val_p[ass] = (1.0 - best_alpha) * p_val_s1[ass] + best_alpha * ass_s2
    final_test_p = (1.0 - best_alpha) * p_test_s1 + best_alpha * test_s2

    val_pred = apply_weights(final_val_p, best_w)
    test_pred = apply_weights(final_test_p, best_w)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path("runs") / f"v3_stacker-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "validation_probabilities.npy", final_val_p)
    np.save(run_dir / "test_probabilities.npy", final_test_p)

    sample = pd.read_csv("data/raw/sample_submission.csv")
    sub_info = write_submission(Path(args.output), ref_test.txn_id.to_numpy(), test_pred, sample)
    reports = {
        "calibration": classification_report_dict(y_val[cal], val_pred[cal]),
        "assessment": classification_report_dict(y_val[ass], val_pred[ass]),
        "full_september": classification_report_dict(y_val, val_pred),
    }
    manifest = {
        "kind": "v3_stage2_stacker",
        "members": [str(p) for p in run_paths],
        "best_alpha_s2": best_alpha,
        "weights": best_w.tolist(),
        "reports": reports,
        "submission": sub_info,
    }
    write_json(run_dir / "manifest.json", manifest)
    print(f"DONE! Selected alpha_s2={best_alpha:.2f}")
    print(f"  Cal F1: {reports['calibration']['macro_f1']:.4f}")
    print(f"  Assess F1 (Sep 24-30): {reports['assessment']['macro_f1']:.4f}")
    print(f"  Full Sep F1: {reports['full_september']['macro_f1']:.4f}")
    print("  Submission:", sub_info["path"], "rows:", sub_info["rows"])


if __name__ == "__main__":
    main()
