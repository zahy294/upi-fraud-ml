"""Use device-number bands as ground truth for fraud type and ask: on band devices, does an event's SHAPE fix its type?

If shape and band agree almost always while raw labels disagree, the sibling-type confusion in train is label noise
(test labels are clean) and events should be typed by shape. Usage: python -I scripts/explore/31_band_truth.py <root>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)
ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src/upi"))
from common import LABELS  # noqa: E402

P = ROOT / "data/processed"
b = pd.read_parquet(P / "base_v0.parquet", columns=["txn_id", "split", "label", "device_id", "sender_id", "receiver_id", "receiver_type", "t",
                                                    "amount", "balance_before", "txn_type", "sim_age_hours", "device_age_days", "home_device",
                                                    "away_from_home", "new_payee", "hour", "app_major", "ip_risk_score", "month"])
f1 = pd.read_parquet(P / "features_v1.parquet", columns=["snd_drain_kind", "dev_drain_kind", "is_1p1", "is_1p02", "snd_drainshape_cnt_3h"])
f3 = pd.read_parquet(P / "features_v3.parquet", columns=["band", "dev_ext_legit"])
f2 = pd.read_parquet(P / "features_v2.parquet", columns=["S_flag", "L_flag", "R_flag", "Q_flag", "qr_native"])
f4 = pd.read_parquet(P / "features_v4.parquet", columns=["bait_cnt_before_7d", "is_tiny_p2p_pay"])
d = pd.concat([b, f1, f3, f2, f4], axis=1)
d["dn"] = d.device_id.str[2:].astype(np.int64)
d["band"] = d.band.astype(int)
d["lab"] = d.label.fillna("TEST")
print(json.dumps(json.load(open(P / "band_edges.json")), indent=None)[:1500])

tr = d[d.split == "train"]
te = d[d.split == "test"]
print("\n=== train: band x raw label (rows on band devices) ===")
print(pd.crosstab(tr[tr.band > 0].band.map(lambda k: f"{k:02d} {LABELS[k]}"), tr[tr.band > 0].lab))

for name, g in (("train", tr), ("test", te)):
    print(f"\n=== {name}: band x sender drain kind (0 none, 1 = x0.25, 2 = x0.35) ===")
    x = g[g.band.isin([4, 5])]
    print(pd.crosstab(x.band.map(lambda k: LABELS[k]), x.snd_drain_kind))
    print(f"--- {name}: band x is_1p1 (balance = 1.1 x amount), bands phishing(6) cashback(9) loan(10) ---")
    x = g[g.band.isin([6, 9, 10])]
    print(pd.crosstab(x.band.map(lambda k: LABELS[k]), [x.is_1p1, x.txn_type]))
    print(f"--- {name}: band x bait count before (7d), paybacks only ---")
    x = g[g.band.isin([6, 9, 10]) & (g.is_1p1 == 1)]
    print(pd.crosstab(x.band.map(lambda k: LABELS[k]), x.bait_cnt_before_7d.clip(upper=3)))
    print(f"--- {name}: band x is_1p02 / fan-in flags, bands mule(1) bot(2) layering(3) ---")
    x = g[g.band.isin([1, 2, 3])]
    print(pd.crosstab(x.band.map(lambda k: LABELS[k]), [x.is_1p02, x.receiver_type]))
    print(f"--- {name}: bands qr(7) refund(8) collusion(11) x txn_type ---")
    x = g[g.band.isin([7, 8, 11])]
    print(pd.crosstab(x.band.map(lambda k: LABELS[k]), [x.txn_type, x.S_flag]))

print("\n=== train: in bands 4/5, raw label x drain kind ===")
x = tr[tr.band.isin([4, 5])]
print(pd.crosstab([x.band.map(lambda k: LABELS[k]), x.lab], x.snd_drain_kind))
print("\n=== train: ALL rows with a drain kind (any device): raw label x kind x on-band-device ===")
x = tr[tr.snd_drain_kind > 0]
print(pd.crosstab([x.snd_drain_kind, (x.band > 0).map({True: "band dev", False: "other dev"})], x.lab))
print("\n=== train drain rows NOT on band devices: kind x label x home_device / fresh sim (<72h) / new device (<2d) ===")
x = tr[(tr.snd_drain_kind > 0) & (tr.band == 0)]
print(pd.crosstab([x.snd_drain_kind, x.home_device, x.sim_age_hours < 72, x.device_age_days < 2], x.lab))
