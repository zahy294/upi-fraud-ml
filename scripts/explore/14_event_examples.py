"""Print compact raw examples of patterned events for a class: anchor = a trap-identified row, context = same sender/receiver/device +-1 day."""
import sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 500)
lab, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
df = pd.read_parquet("data/processed/base_v0.parquet")
df["ts"] = df.timestamp.dt.strftime("%m-%d %H:%M:%S"); df["dev"] = df.device_id.str[-5:]
df["cat"] = df.merchant_category.astype("object").fillna("-").str[:6]; df["lb"] = df.label.astype("object").fillna("TEST").str[:9]
df["snd"] = df.sender_id.str[2:]; df["rcv"] = df.receiver_id.str[:1] + df.receiver_id.str[2:]
df["o"] = (df.app_major == "3").astype(int)
cols = ["ts", "snd", "rcv", "dev", "amount", "txn_type", "cat", "new_payee", "balance_before", "login_to_txn_sec", "sim_age_hours", "device_age_days", "qr_vpa_mismatch", "o", "ip_risk_score", "lb"]
tr = df[df.split == "train"]
anchors = tr[(tr.label == lab) & (tr.app_major == "3") & (tr.ip_risk_score > 0.5)]
rng = np.random.RandomState(seed)
for i in rng.choice(len(anchors), n, replace=False):
    a = anchors.iloc[i]
    m = (df.t.between(a.t - 86400, a.t + 86400)) & ((df.sender_id == a.sender_id) | (df.receiver_id == a.receiver_id) | (df.device_id == a.device_id)
                                                     | (df.sender_id == a.receiver_id) | (df.receiver_id == a.sender_id))
    d = df[m].sort_values("t")
    if len(d) > 28:
        d = d[(d.t - a.t).abs() <= 6 * 3600]
    if len(d) > 28:
        d = d[(d.sender_id == a.sender_id) | (d.device_id == a.device_id) | (d.label != "legit")].head(28)
    print(f"--- anchor {a.ts} snd {a.snd} rcv {a.rcv} amt {a.amount}")
    print(d[cols].to_string(index=False, header=(i == i)))
    print()
