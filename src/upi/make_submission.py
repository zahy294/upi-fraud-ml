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
Pte = sum(np.load(ROOT / "models" / t / "test_proba.npy").astype(float) for t in tags) / len(tags)

base = pd.read_parquet(ROOT / "data/processed/base_v0.parquet", columns=["txn_id", "label", "split"])
tr = (base.split == "train").values
y_raw = base.label[tr].map({lab: i for i, lab in enumerate(LABELS)}).values.astype(int)
y = y_raw if a.labels == "raw" else np.load(ROOT / a.labels)
band = pd.read_parquet(ROOT / "data/processed/features_v3.parquet", columns=["band"]).band.values.astype(int)

w = fit_weights(oof, y)
pred_tr = (oof * w).argmax(1)
if not a.no_band:
    pred_tr = np.where(band[tr] > 0, band[tr], pred_tr)
print(f"pooled out-of-fold macro-F1: vs given labels {macro_f1(y, pred_tr):.4f} | vs raw labels {macro_f1(y_raw, pred_tr):.4f}")
T = per_class(y, pred_tr)
T["w"] = w
print(T.round(3).to_string())

pred = (Pte * w).argmax(1)
if not a.no_band:
    bt = band[~tr]
    print("band rule on test: rows", int((bt > 0).sum()), "changed", int(((bt > 0) & (pred != bt)).sum()))
    pred = np.where(bt > 0, bt, pred)
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
