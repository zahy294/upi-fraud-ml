"""Fixed, label-free candidate signals; training-month diagnostics only.

Run from project root: python scripts/research/probe_conditional_signals.py
This is research, not a training or submission pipeline. Original CSVs are read only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "upi-fraud-rings" / "train.csv"
OUT = ROOT / "reports" / "research"
LABELS = ["legit", "mule_account", "velocity_bot", "layering_chain", "sim_swap",
          "account_takeover", "phishing_collect", "qr_swap", "refund_abuse",
          "fake_cashback", "loan_app_extortion", "merchant_collusion"]
MONTHS = ["2026-07", "2026-08", "2026-09"]


def main():
    cols = ["txn_id", "timestamp", "sender_id", "receiver_id", "receiver_type",
            "device_id", "amount", "txn_type", "merchant_category", "sim_age_hours",
            "device_age_days", "new_payee", "pin_retries", "login_to_txn_sec",
            "balance_before", "qr_vpa_mismatch", "label"]
    df = pd.read_csv(DATA, usecols=cols)
    df["t"] = pd.to_datetime(df.timestamp)
    df = df.sort_values(["t", "txn_id"], kind="stable").reset_index(drop=True)
    month = df.timestamp.str[:7]
    integer = np.isclose(df.amount, np.rint(df.amount), rtol=0, atol=1e-7)
    p2p = df.receiver_type.eq("P2P")
    ratio = df.amount.div(df.balance_before.where(df.balance_before.gt(0)))
    new = df.new_payee.eq(1)
    sim = df.sim_age_hours.lt(168)
    device = df.device_age_days.lt(7)
    small500 = df.amount.eq(500)
    drain = df.amount.gt(500) & ratio.ge(.5)
    # Shift after forward/backward filling so the query transaction is excluded.
    # Strict positive time gap leaves same-second order unresolved.
    prev500 = df.t.where(small500).groupby(df.sender_id, sort=False).ffill()
    prev500 = prev500.groupby(df.sender_id, sort=False).shift(1)
    nextdrain = df.t.where(drain).groupby(df.sender_id, sort=False).bfill()
    nextdrain = nextdrain.groupby(df.sender_id, sort=False).shift(-1)
    before_gap = (df.t-prev500).dt.total_seconds()
    after_gap = (nextdrain-df.t).dt.total_seconds()
    next_same_month = nextdrain.dt.strftime("%Y-%m").eq(month)
    episode500 = (drain & before_gap.gt(0) & before_gap.le(3600)) | (small500 & after_gap.gt(0) & after_gap.le(3600) & next_same_month)
    hour = df.t.dt.hour
    home_phone = df.device_id.eq("DV0" + df.sender_id.str[2:])
    # Month-local symmetric device counts reproduce the baseline's 5-minute
    # burst condition, while preventing later-month context entering a probe.
    group = pd.factorize(month + "/" + df.device_id, sort=False)[0].astype(np.int64)
    seconds = df.t.to_numpy(dtype="datetime64[s]").astype(np.int64)
    span = int(seconds.max()-seconds.min()+2*3600+1)
    keys = group*span + seconds-seconds.min()+3600
    sorted_keys = np.sort(keys)
    count5 = np.searchsorted(sorted_keys,keys+300,side="right")-np.searchsorted(sorted_keys,keys-300,side="left")
    count60 = np.searchsorted(sorted_keys,keys+3600,side="right")-np.searchsorted(sorted_keys,keys-3600,side="left")
    old_bot_rule = integer & ~p2p & ~home_phone & ((count5>=3) | df.login_to_txn_sec.le(5))
    features = {
        "amount_above_balance": df.amount.gt(df.balance_before),
        "above_balance_integer_new_p2p": df.amount.gt(df.balance_before) & integer & new & p2p,
        "drain_integer_new_p2p": drain & integer & new & p2p,
        "drain_integer_new_p2p_fresh_sim": drain & integer & new & p2p & sim,
        "drain_integer_new_p2p_new_device": drain & integer & new & p2p & device,
        "500_then_drain_episode_1h": episode500,
        "500_then_drain_episode_1h_fresh_sim": episode500 & sim,
        "500_then_drain_episode_1h_new_device": episode500 & device,
        "500_then_drain_episode_1h_integer_new_p2p": episode500 & integer & new & p2p,
        "500_then_drain_episode_1h_outside_sim_row_rule": episode500 & ~(integer & p2p & sim & new),
        "500_then_drain_episode_1h_outside_ato_row_rule": episode500 & ~(integer & p2p & device & new),
        "night_drain_new_p2p_sim_7_to_30d": drain & new & p2p & hour.lt(6) & df.sim_age_hours.between(168,720,inclusive="left"),
        "drain_new_p2p_device_7_to_30d": drain & new & p2p & df.device_age_days.between(7,30,inclusive="left"),
        "integer_new_p2p_fast_login": integer & new & p2p & df.login_to_txn_sec.le(5),
        "integer_nonhome_merchant_slow_login": integer & ~home_phone & ~p2p & df.login_to_txn_sec.between(6,20),
        "integer_nonhome_merchant_slow_login_outside_bot_rule": integer & ~home_phone & ~p2p & df.login_to_txn_sec.between(6,20) & ~old_bot_rule,
        "device_hour_burst_outside_bot_rule": integer & ~home_phone & ~p2p & (count60>=3) & ~old_bot_rule,
        "fractional_refund": ~integer & df.txn_type.eq("REFUND"),
        "integer_refund": integer & df.txn_type.eq("REFUND"),
        "round500_refund": df.amount.mod(500).abs().lt(1e-7) & df.txn_type.eq("REFUND"),
        "round500_merchant_pay": df.amount.mod(500).abs().lt(1e-7) & ~p2p & df.txn_type.eq("PAY"),
        "round500_sim_new_p2p": df.amount.mod(500).abs().lt(1e-7) & sim & new & p2p,
        "round500_device_new_p2p": df.amount.mod(500).abs().lt(1e-7) & device & new & p2p,
    }
    risk_definitions = [
        ("receiver_prior_qr_current_qr", "receiver_id", "qr_swap", df.txn_type.eq("QR"), 1),
        ("receiver_prior_qr_current_qr_outside_qr_rule", "receiver_id", "qr_swap", df.txn_type.eq("QR") & ~(df.qr_vpa_mismatch.eq(1) | (df.txn_type.eq("QR") & p2p)), 1),
        ("receiver_prior_qr_current_p2p", "receiver_id", "qr_swap", p2p, 1),
        ("receiver_prior_phishing_current_collect", "receiver_id", "phishing_collect", df.txn_type.eq("COLLECT"), 1),
        ("sender_prior_refund_current_refund", "sender_id", "refund_abuse", df.txn_type.eq("REFUND"), 1),
        ("sender_prior_2refund_current_refund", "sender_id", "refund_abuse", df.txn_type.eq("REFUND"), 2),
        ("device_prior_bot_current_merchant", "device_id", "velocity_bot", ~p2p, 1),
        ("device_prior_bot_current_merchant_outside_bot_rule", "device_id", "velocity_bot", ~p2p & ~old_bot_rule, 1),
        ("receiver_prior_anyfraud_current_p2p", "receiver_id", "anyfraud", p2p, 1),
        ("receiver_prior_2fraud_current_p2p", "receiver_id", "anyfraud", p2p, 2),
    ]
    for name, role, target, context, minimum in risk_definitions:
        mask=pd.Series(False,index=df.index)
        for m in MONTHS[1:]:
            historical=month.lt(m)
            positive=df.label.ne("legit") if target=="anyfraud" else df.label.eq(target)
            counts=df.loc[historical & positive,role].value_counts()
            mask |= month.eq(m) & context & df[role].map(counts).fillna(0).ge(minimum)
        features[name]=mask
    gap=np.ceil(df.amount/1000)*1000-df.amount
    just_under=(gap>0) & (gap<=50) & (df.amount>=900)
    ends99=np.floor(df.amount).mod(100).eq(99)
    old_phish_rule=df.amount.eq(np.floor(df.amount)) & df.txn_type.eq("COLLECT") & (just_under|ends99)
    features["receiver_prior_phishing_current_collect_outside_phish_rule"]=features["receiver_prior_phishing_current_collect"] & ~old_phish_rule
    features["receiver_prior_phishing_current_collect_p2p_outside_phish_rule"]=features["receiver_prior_phishing_current_collect_outside_phish_rule"] & p2p
    features["receiver_prior_qr_current_p2p_outside_qr_rule"]=features["receiver_prior_qr_current_p2p"] & ~(df.qr_vpa_mismatch.eq(1) | (df.txn_type.eq("QR") & p2p))
    refund_counts=np.zeros(len(df),dtype=np.int32)
    sender_code=pd.factorize(df.sender_id,sort=False)[0].astype(np.int64)
    half_window=15*86400
    sender_span=int(seconds.max()-seconds.min()+2*half_window+1)
    sender_keys=sender_code*sender_span+seconds-seconds.min()+half_window
    for m in MONTHS[1:]:
        available=month.le(m) & df.txn_type.eq("REFUND")
        events=np.sort(sender_keys[available.to_numpy()])
        query=month.eq(m).to_numpy()
        query_keys=sender_keys[query]
        refund_counts[query]=np.searchsorted(events,query_keys+half_window,side="right")-np.searchsorted(events,query_keys-half_window,side="left")
    features["sender_prior_refund_current_refund_outside_refund_rule"]=features["sender_prior_refund_current_refund"] & (refund_counts<3)
    features["sender_prior_2refund_current_refund_outside_refund_rule"]=features["sender_prior_2refund_current_refund"] & (refund_counts<3)
    results=[]
    for m in MONTHS:
        mm=month.eq(m)
        support=df.loc[mm,"label"].value_counts().reindex(LABELS,fill_value=0)
        for name, mask in features.items():
            hit=mm & mask
            hit_counts=df.loc[hit,"label"].value_counts().reindex(LABELS,fill_value=0)
            n=int(hit.sum())
            metrics={label:{"hits":int(hit_counts[label]), "support":int(support[label]),
                            "precision":float(hit_counts[label]/n) if n else 0.,
                            "recall":float(hit_counts[label]/support[label]) if support[label] else 0.}
                     for label in LABELS}
            results.append({"month":m,"signal":name,"n_flagged":n,"metrics":metrics})
    digest=hashlib.sha256()
    with DATA.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            digest.update(chunk)
    document={"method":"38 fixed masks: initial 21 row/episode masks, 2 follow-up burst-overlap checks, 10 prior-month label-history masks, and 5 follow-up refund/phishing/QR overlap or P2P-stratum checks. History masks use strictly earlier months' labels, with July unavailable for history-based evaluation. No model fits, threshold search, test-row inspection, or test labels. All scored months are exploratory diagnostics once inspected; no untouched evaluation is claimed.",
              "data_path":str(DATA.relative_to(ROOT)),"sha256":digest.hexdigest(),
              "rows":len(df),"excluded_outside_Jul_Sep":int((~month.isin(MONTHS)).sum()),
              "versions":{"python":platform.python_version(),"pandas":pd.__version__,"numpy":np.__version__},
              "same_second_policy":"Exclude zero-time 500-to-drain edges; txn_id is only a deterministic tie breaker.",
              "context_policy":"Past 500 payment may be in an earlier month. Following drain and device burst context are restricted to the query month; no future-month rows support a query. Device burst overlap check is month-local, so it may differ slightly from original full-dataset rules at month boundaries. Refund-rule overlap counts REFUND rows per sender within +/-15days using historical plus current-month rows only; this likewise differs from the EDA's full-data context at the right boundary.",
              "history_policy":"For August history masks, use July labels only; for September, use July+August labels only. July history masks are unavailable (all false), not meaningful zero-performance estimates. Counts come from original labels with no relabelling.",
              "definitions":{"drain":"amount > 500 and amount / positive balance_before >= 0.5", "episode500":"same sender has exact 500 payment followed in (0,3600] seconds by a drain; flag both ends", "fresh_sim":"sim_age_hours < 168", "new_device":"device_age_days < 7", "new":"new_payee == 1"},
              "results":results}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"conditional_signal_probe.json").write_text(json.dumps(document,indent=2),encoding="utf-8")
    lines=["# Conditional signal probe", "", document["method"], "", "Training data SHA256: `"+digest.hexdigest()+"`.", "", "No additional signal here has yet demonstrated an improvement in macro-F1. Precision below refers to each mask as a one-vs-rest prediction, without resolving overlaps.", ""]
    for name in features:
        lines.extend(["## "+name, ""])
        for m in MONTHS:
            r=next(r for r in results if r["signal"]==name and r["month"]==m)
            ordered=sorted(((lab,v) for lab,v in r["metrics"].items() if lab!="legit"),key=lambda x:x[1]["hits"],reverse=True)[:3]
            stats="; ".join(f"{lab}: {v['hits']} positives, precision {v['precision']:.1%}, recall {v['recall']:.1%}" for lab,v in ordered)
            lines.append(f"- {m}: {r['n_flagged']} flagged. {stats}.")
        lines.append("")
    (OUT/"conditional_signal_probe.md").write_text("\n".join(lines),encoding="utf-8")
    for r in results:
        if r["month"]=="2026-09":
            best=max(((lab,v) for lab,v in r["metrics"].items() if lab!="legit"),key=lambda x:x[1]["hits"])
            print(r["signal"],r["n_flagged"],best)


if __name__ == "__main__":
    main()

