"""Clean the training labels the way the test labels are clean.

Inside a fraud event about 12% of rows are labelled legit (unreported) and about 3% carry some other fraud type.
The event's own type is reliable in the raw labels (rows on device-band devices carry the band's label 92-98% of the
time), including events that one generator emits in a sibling type's shape. So a row is corrected to the type of
ITS EVENT, taken as the majority raw label among the event's other detected rows, never to the type its shape suggests.

  band device                       -> band type
  labelled legit, model sure (>=tau)-> event type (falls back to the model's type when the event has no labelled row)
  labelled fraud, differs from its event's majority -> event type

Evidence used: leave-one-month-out probabilities from a model trained on the raw labels, device bands, raw labels of
neighbouring rows. The trap columns decide nothing; they are printed as an independent purity check.

Usage: python -I src/upi/clean_labels.py <project_root> <oof_tag> [--tau-unrep 0.85]
Writes models/<oof_tag>/y_clean.npy (train rows in base order).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import LABELS, macro_f1, fit_weights  # noqa: E402
from events import Events, MIN, HOUR, DAY  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("tag")
ap.add_argument("--tau-unrep", type=float, default=0.85)
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)
ROOT = Path(a.root)
OUT = ROOT / "models" / a.tag
K = len(LABELS)

base = pd.read_parquet(ROOT / "data/processed/base_v0.parquet",
                       columns=["txn_id", "t", "label", "split", "month", "app_major", "ip_risk_score", "sender_id", "receiver_id", "device_id", "receiver_type"])
tr = (base.split == "train").values
b = base[tr]
y = b.label.map({lab: i for i, lab in enumerate(LABELS)}).values.astype(int)
old = (b.app_major == "3").values
band = pd.read_parquet(ROOT / "data/processed/features_v3.parquet", columns=["band"]).band.values[tr].astype(int)
s_flag = pd.read_parquet(ROOT / "data/processed/features_v2.parquet", columns=["S_flag"]).S_flag.values[tr] == 1
P = np.load(OUT / "oof.npy").astype(float)
pf = 1 - P[:, 0]
top = P[:, 1:].argmax(1) + 1

# ---- the type of each row's event: votes from the raw labels of other detected rows linked to it ----
t = b.t.values.astype(np.int64)
snd = b.sender_id.str[2:].astype(np.int64).values
is_p2p = (b.receiver_type == "P2P").values
rnum = b.receiver_id.str[2:].astype(np.int64).values
rcv = np.where(is_p2p, rnum, rnum + 1_000_000)
dev = b.device_id.str[2:].astype(np.int64).values
pair = snd * 2_000_000 + rcv
pcode, puniq = pd.factorize(pair)
rcode = pd.Index(puniq).get_indexer(np.where(is_p2p, rnum * 2_000_000 + snd, -1))
rcv_evt = np.where(is_p2p | s_flag, rcv, -1)            # receiver link only for person accounts and fixed-amount merchant sessions
votes = np.zeros((len(y), K))
detected = (pf >= 0.5) | (band > 0)
for k in range(1, K):
    ind = ((y == k) & detected).astype(float)
    if ind.sum() == 0:
        continue
    v = (Events(snd, t, ind).sum(snd, t, -7 * DAY, 7 * DAY) - ind                      # same sender (bursts, repeats over days)
         + Events(rcv_evt, t, ind, mask=rcv_evt >= 0).sum(rcv_evt, t, -DAY, DAY) - ind * (rcv_evt >= 0)   # payments into the same account
         + Events(dev, t, ind).sum(dev, t, -10 * MIN, 10 * MIN) - ind               # same device within minutes
         + Events(pcode, t, ind).sum(rcode, t, -7 * DAY, 7 * DAY))                   # opposite direction of the pair (bait and payback)
    votes[:, k] = np.maximum(v, 0)
tot = votes.sum(1)
evt = votes.argmax(1)
evt_ok = (votes.max(1) >= 1) & (votes.max(1) / np.maximum(tot, 1e-9) >= 0.6)
evt = np.where(evt_ok, evt, 0)


def purity(mask):
    return (old[mask].mean() - 0.1506) / (0.625 - 0.1506) if mask.any() else np.nan


yc = y.copy()
r_unrep = (band == 0) & (y == 0) & (pf >= a.tau_unrep)
r_type = (band == 0) & (y > 0) & detected & (evt > 0) & (evt != y)
r_band = (band > 0) & (y != band)
yc[r_unrep] = np.where(evt[r_unrep] > 0, evt[r_unrep], top[r_unrep])
yc[r_type] = evt[r_type]
yc[r_band] = band[r_band]
print(f"unreported -> fraud: {r_unrep.sum()} rows (patterned~{purity(r_unrep):.2f}); event type known for {(evt[r_unrep] > 0).mean():.2f} of them, "
      f"and it differs from the model's type in {(evt[r_unrep] != top[r_unrep])[evt[r_unrep] > 0].mean():.3f}")
print(f"row disagrees with its event -> event type: {r_type.sum()} rows (patterned~{purity(r_type):.2f})")
print(f"band rule: {r_band.sum()} rows")
keep_sibling = (band == 0) & (y > 0) & detected & (top != y) & ~r_type
print(f"fraud rows whose label differs from the model's type but agrees with their event (kept as labelled): {keep_sibling.sum()}")
print(pd.crosstab(pd.Series(np.array(LABELS)[y[keep_sibling]], name="label kept"), pd.Series(np.array(LABELS)[top[keep_sibling]], name="model type")).to_string())
ct = pd.DataFrame({"raw": np.bincount(y, minlength=K), "clean": np.bincount(yc, minlength=K)}, index=LABELS)
ct["change_%"] = ((ct.clean / ct.raw - 1) * 100).round(1)
print(ct.to_string())
np.save(OUT / "y_clean.npy", yc)
w = fit_weights(P, yc)
pred = np.where(band > 0, band, (P * w).argmax(1))
print(f"\nthe raw-label model scored against cleaned labels: {macro_f1(yc, pred):.4f} | against raw labels: {macro_f1(y, pred):.4f}")
