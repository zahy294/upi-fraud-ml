from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
import lightgbm as lgb

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
    p_xgb: np.ndarray,
    p_lgb: np.ndarray,
    p_cat: np.ndarray,
    channels: np.ndarray,
    context_df: pd.DataFrame,
) -> pd.DataFrame:
    n = len(p_xgb)
    feats: dict[str, np.ndarray] = {}

    # 1. Individual model probabilities & ensemble mean/max/std
    p_stack = np.stack([p_xgb, p_lgb, p_cat], axis=0)  # (3, N, 12)
    p_mean = p_stack.mean(axis=0)
    p_max = p_stack.max(axis=0)
    p_min = p_stack.min(axis=0)
    p_std = p_stack.std(axis=0)

    for c, lab in enumerate(LABELS):
        feats[f"p_xgb_{lab}"] = p_xgb[:, c].astype(np.float32)
        feats[f"p_lgb_{lab}"] = p_lgb[:, c].astype(np.float32)
        feats[f"p_cat_{lab}"] = p_cat[:, c].astype(np.float32)
        feats[f"p_mean_{lab}"] = p_mean[:, c].astype(np.float32)
        feats[f"p_max_{lab}"] = p_max[:, c].astype(np.float32)
        feats[f"p_min_{lab}"] = p_min[:, c].astype(np.float32)
        feats[f"p_std_{lab}"] = p_std[:, c].astype(np.float32)

    # 2. Global fraud summary statistics
    fraud_probs = p_mean[:, 1:]
    feats["p_fraud_sum"] = (1.0 - p_mean[:, 0]).astype(np.float32)
    feats["p_fraud_max"] = fraud_probs.max(axis=1).astype(np.float32)
    part = np.partition(fraud_probs, -2, axis=1)
    feats["p_fraud_second"] = part[:, -2].astype(np.float32)
    feats["p_fraud_margin"] = (feats["p_fraud_max"] - feats["p_fraud_second"]).astype(np.float32)
    feats["p_fraud_ratio"] = (feats["p_fraud_max"] / np.maximum(feats["p_fraud_second"], 1e-6)).astype(np.float32)
    feats["p_entropy"] = (-np.sum(np.clip(p_mean, 1e-7, 1.0) * np.log(np.clip(p_mean, 1e-7, 1.0)), axis=1)).astype(np.float32)

    # 3. Pairwise confusion log-odds & sums for confused pairs
    pairs = [
        ("sim_swap", "account_takeover"),
        ("mule_account", "layering_chain"),
        ("phishing_collect", "fake_cashback"),
        ("qr_swap", "refund_abuse"),
        ("qr_swap", "merchant_collusion"),
        ("mule_account", "velocity_bot"),
    ]
    for a, b in pairs:
        ia, ib = LABEL_TO_ID[a], LABEL_TO_ID[b]
        pa, pb = p_mean[:, ia], p_mean[:, ib]
        feats[f"logodds_{a}_vs_{b}"] = np.log((pa + 1e-5) / (pb + 1e-5)).astype(np.float32)
        feats[f"sum_{a}_{b}"] = (pa + pb).astype(np.float32)
        feats[f"share_{a}_in_{b}"] = (pa / np.maximum(pa + pb, 1e-6)).astype(np.float32)

    # 4. Neighbour channel probabilities & ratios
    # channels has shape (N, 7, 12)
    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        ch_p = channels[:, ch_idx, :]
        feats[f"nb_{ch_name}_anyfraud"] = ch_p[:, 1:].sum(axis=1).astype(np.float32)
        feats[f"nb_{ch_name}_maxfraud"] = ch_p[:, 1:].max(axis=1).astype(np.float32)
        for c in range(1, len(LABELS)):
            lab = LABELS[c]
            feats[f"nb_{ch_name}_{lab}"] = ch_p[:, c].astype(np.float32)
            # Combined episode probability (self + neighbour)
            feats[f"ep_{ch_name}_{lab}"] = (0.5 * p_mean[:, c] + 0.5 * ch_p[:, c]).astype(np.float32)

    # 5. Context columns & density-normalized shares
    for col in CONTEXT_COLS:
        if col in context_df.columns:
            feats[f"ctx_{col}"] = context_df[col].to_numpy(dtype=np.float32)

    if "pair_round500_cnt_30d" in context_df.columns and "pair_cnt_30d" in context_df.columns:
        c500 = context_df["pair_round500_cnt_30d"].to_numpy(dtype=np.float32)
        c30 = np.maximum(context_df["pair_cnt_30d"].to_numpy(dtype=np.float32), 1.0)
        feats["ctx_pair_round500_share_30d"] = (c500 / c30).astype(np.float32)

    return pd.DataFrame(feats)


def main():
    runs = [
        Path("runs/xgb_v2-20261113-201458"),
        Path("runs/lgb_v2-20261113-202052"),
        Path("runs/catboost_v2-20261113-202410"),
    ]
    print("Loading feature table and Stage-1 validation predictions...")
    table = pd.read_parquet("data/processed/features_engineered_v2.parquet")
    train_sep = table[table.split.eq("train") & table.timestamp.ge("2026-09-01") & table.timestamp.lt("2026-10-01")].copy()
    cal_mask_tbl = train_sep.timestamp.lt("2026-09-24")
    ass_mask_tbl = train_sep.timestamp.ge("2026-09-24")
    val_context = pd.concat([train_sep.loc[cal_mask_tbl], train_sep.loc[ass_mask_tbl]], axis=0).reset_index(drop=True)

    ref = pd.read_parquet(runs[0] / "validation_predictions.parquet")
    assert (val_context.txn_id.to_numpy() == ref.txn_id.to_numpy()).all(), "Row order mismatch!"

    y = ref.label.map(LABEL_TO_ID).to_numpy(dtype=np.int32)
    cal = (pd.to_datetime(ref.timestamp) < "2026-09-24").to_numpy()
    ass = ~cal

    p_xgb = np.load(runs[0] / "validation_probabilities.npy")
    p_lgb = np.load(runs[1] / "validation_probabilities.npy")
    p_cat = np.load(runs[2] / "validation_probabilities.npy")

    # Baseline Stage-1 equal blend
    p_eq = (p_xgb + p_lgb + p_cat) / 3.0
    w_eq = fit_class_weights(p_eq[cal], y[cal])
    print(f"Baseline Equal Blend | Cal F1: {macro_f1_ids(y[cal], apply_weights(p_eq[cal], w_eq)):.4f} | Assess F1: {macro_f1_ids(y[ass], apply_weights(p_eq[ass], w_eq)):.4f}")

    # Test 1: Class-specific model blending weights (12 x 3)
    print("Fitting per-class model blend weights...")
    best_model_w = np.full((12, 3), 1.0 / 3.0, dtype=np.float64)
    p_blend = p_eq.copy()
    w_curr = w_eq.copy()
    best_cal_f1 = macro_f1_ids(y[cal], apply_weights(p_blend[cal], w_curr))
    candidates = [
        (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
        (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5),
        (0.5, 0.25, 0.25), (0.25, 0.5, 0.25), (0.25, 0.25, 0.5),
        (0.4, 0.4, 0.2), (0.4, 0.2, 0.4), (0.2, 0.4, 0.4),
        (1/3, 1/3, 1/3),
    ]
    for c in range(1, 12):
        for wx, wl, wc in candidates:
            trial_col = wx * p_xgb[cal, c] + wl * p_lgb[cal, c] + wc * p_cat[cal, c]
            old_col = p_blend[cal, c].copy()
            p_blend[cal, c] = trial_col
            # Also quick line search on w_curr[c]
            for mult in (0.85, 0.92, 1.0, 1.08, 1.18):
                tw = w_curr.copy()
                tw[c] = w_curr[c] * mult
                score = macro_f1_ids(y[cal], apply_weights(p_blend[cal], tw))
                if score > best_cal_f1 + 1e-6:
                    best_cal_f1 = score
                    best_model_w[c] = (wx, wl, wc)
                    w_curr[c] = tw[c]
            p_blend[:, c] = best_model_w[c, 0] * p_xgb[:, c] + best_model_w[c, 1] * p_lgb[:, c] + best_model_w[c, 2] * p_cat[:, c]
    w_cw = fit_class_weights(p_blend[cal], y[cal])
    print(f"Per-Class Model Blend | Cal F1: {macro_f1_ids(y[cal], apply_weights(p_blend[cal], w_cw)):.4f} | Assess F1: {macro_f1_ids(y[ass], apply_weights(p_blend[ass], w_cw)):.4f}")

    # Compute neighbour channels on p_blend
    print("Computing 7-channel neighbour probabilities...")
    channels = compute_neighbour_channels(val_context, p_blend)

    print("Building Stage-2 meta-feature matrix...")
    X_meta = build_meta_features(p_xgb, p_lgb, p_cat, channels, val_context)
    print("Meta-feature matrix shape:", X_meta.shape)

    # 3-fold out-of-fold CV on calibration (grouped by sender_id)
    cal_idx = np.where(cal)[0]
    ass_idx = np.where(ass)[0]
    senders_cal = pd.factorize(val_context.loc[cal, "sender_id"])[0]
    folds = senders_cal % 3

    oof_cal = np.zeros((len(cal_idx), 12), dtype=np.float64)
    pred_ass = np.zeros((len(ass_idx), 12), dtype=np.float64)

    for fold in range(3):
        tr_m = folds != fold
        va_m = folds == fold
        model = LGBMClassifier(
            objective="multiclass",
            num_class=12,
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=100,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_lambda=10.0,
            reg_alpha=2.0,
            n_jobs=-1,
            random_state=2026 + fold,
            verbose=-1,
        )
        model.fit(
            X_meta.iloc[cal_idx[tr_m]], y[cal_idx[tr_m]],
            eval_set=[(X_meta.iloc[cal_idx[va_m]], y[cal_idx[va_m]])],
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )
        print(f"  Fold {fold} best_iteration: {model.best_iteration_}")
        oof_cal[va_m] = model.predict_proba(X_meta.iloc[cal_idx[va_m]])
        pred_ass += model.predict_proba(X_meta.iloc[ass_idx]) / 3.0

    w_meta = fit_class_weights(oof_cal, y[cal])
    print(f"Stage-2 GBDT Only | OOF Cal F1: {macro_f1_ids(y[cal], apply_weights(oof_cal, w_meta)):.4f} | Assess F1: {macro_f1_ids(y[ass], apply_weights(pred_ass, w_meta)):.4f}")

    # Blend Stage-1 p_blend + Stage-2 GBDT
    for alpha_s2 in (0.15, 0.25, 0.35, 0.50, 0.65, 0.80):
        comb_cal = (1.0 - alpha_s2) * p_blend[cal] + alpha_s2 * oof_cal
        comb_ass = (1.0 - alpha_s2) * p_blend[ass] + alpha_s2 * pred_ass
        w_c = fit_class_weights(comb_cal, y[cal])
        f1_c = macro_f1_ids(y[cal], apply_weights(comb_cal, w_c))
        f1_a = macro_f1_ids(y[ass], apply_weights(comb_ass, w_c))
        print(f"Blend (alpha_s2={alpha_s2:.2f}) | Cal F1: {f1_c:.4f} | Assess F1: {f1_a:.4f}")


if __name__ == "__main__":
    main()
