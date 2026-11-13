"""Turn a model's out-of-fold and test probabilities into a submission.

Class weights are fitted on the pooled leave-one-month-out probabilities against the given labels, then applied to the
test probabilities; rows on fraud-band devices take the band type.

Usage: python -I src/upi/make_submission.py <project_root> <tag> [--labels models/x/y_clean.npy] [--name out_name]
           [--blend tagA,tagB]   (average probabilities of several tags that share the same row order)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import LABELS, macro_f1, per_class, fit_weights  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("tag")
ap.add_argument("--labels", default="raw")
ap.add_argument("--name", default=None)
ap.add_argument("--blend", default=None)
ap.add_argument("--no-band", action="store_true")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)
ROOT = Path(a.root)
tags = a.blend.split(",") if a.blend else [a.tag]
oof = sum(np.load(ROOT / "models" / t / "oof.npy").astype(float) for t in tags) / len(tags)
have_test = all((ROOT / "models" / t / "test_proba.npy").exists() for t in tags)
Pte = sum(np.load(ROOT / "models" / t / "test_proba.npy").astype(float) for t in tags) / len(tags) if have_test else None

base = pd.read_parquet(ROOT / "data/processed/base_v0.parquet", columns=["txn_id", "label", "split"])
tr = (base.split == "train").values
y_raw = base.label[tr].map({lab: i for i, lab in enumerate(LABELS)}).values.astype(int)
y = y_raw if a.labels == "raw" else np.load(ROOT / a.labels)
band = pd.read_parquet(ROOT / "data/processed/features_v3.parquet", columns=["band"]).band.values.astype(int)

F3 = pd.read_parquet(ROOT / "data/processed/features_v3.parquet")
F5 = pd.read_parquet(ROOT / "data/processed/features_v5.parquet", columns=["ch_band"])
p2p_all = pd.read_parquet(ROOT / "data/processed/base_v0.parquet", columns=["receiver_type"]).receiver_type.values == "P2P"


def band_rules(pred, rows):
    """A device band on one row of an event gives the type of the whole event (see bands.py)."""
    b = band[rows]
    S = np.stack([F3[f"snd_band_{k}_3h"].values[rows] for k in range(1, 12)], 1)
    R = np.stack([F3[f"rcv_band_{k}_24h"].values[rows] for k in range(1, 12)], 1)
    sb = np.where((S > 0).sum(1) == 1, S.argmax(1) + 1, 0)           # the one band type among the sender's rows within 3 h
    rb = np.where((R > 0).sum(1) == 1, R.argmax(1) + 1, 0)           # ... among payments into the receiver within 24 h
    cb = F5.ch_band.values[rows].astype(int)                         # ... among the rows of the forwarding chain
    p2p = p2p_all[rows]
    out = np.where(b > 0, b, pred)
    free = (b == 0) & (pred > 0)
    drain = free & np.isin(pred, [4, 5]) & np.isin(sb, [4, 5])
    fanin = free & np.isin(pred, [1, 2, 3]) & p2p & np.isin(rb, [1, 2, 3])
    shop = free & np.isin(pred, [7, 8, 11]) & ~p2p & np.isin(rb, [7, 8, 11])
    chain = free & (cb > 0)
    out = np.where(drain, sb, out)
    out = np.where(fanin | shop, rb, out)
    out = np.where(chain, cb, out)
    return out, int((out != pred).sum())


w = fit_weights(oof, y)
pred_tr = (oof * w).argmax(1)
if not a.no_band:
    pred_tr, _ = band_rules(pred_tr, tr)
print(f"pooled out-of-fold macro-F1: vs given labels {macro_f1(y, pred_tr):.4f} | vs raw labels {macro_f1(y_raw, pred_tr):.4f}")
T = per_class(y, pred_tr)
T["w"] = w
print(T.round(3).to_string())

if Pte is None:
    print("no test probabilities for this tag: local score only")
    sys.exit(0)
pred = (Pte * w).argmax(1)
if not a.no_band:
    pred, n_changed = band_rules(pred, ~tr)
    print("band rules on test: rows on band devices", int((band[~tr] > 0).sum()), "| predictions changed", n_changed)
sub = pd.DataFrame({"id": base.txn_id.values[~tr], "label": np.array(LABELS)[pred]})
sample = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
sub = sample[["id"]].merge(sub, on="id", how="left", validate="one_to_one")
assert len(sub) == len(sample) and sub.label.notna().all() and (sub.id.values == sample.id.values).all()
assert set(sub.label) == set(LABELS), "every label must be predicted at least once"
name = a.name or a.tag
sub.to_csv(ROOT / "submissions" / f"{name}.csv", index=False)
share_te = sub.label.value_counts() / len(sub) * 100
share_tr = pd.Series(np.bincount(pred_tr, minlength=12), index=LABELS) / len(pred_tr) * 100
print(f"\nwrote submissions/{name}.csv | predicted % of rows, test vs train out-of-fold:")
print(pd.DataFrame({"test": share_te, "train_oof": share_tr}).reindex(LABELS).round(3).T.to_string())
np.save(ROOT / "models" / tags[0] / f"weights_{name}.npy", w)
