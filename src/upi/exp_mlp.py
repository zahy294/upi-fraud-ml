"""A different style: a small neural network on the same feature table (a second model family for the blend).

Numeric columns are squashed (signed log), standardised with statistics from all rows, and missing values become 0
plus a missing flag where a column is missing in more than 1% of rows. Categorical columns are one-hot encoded.
Like run_oof.py it writes leave-one-month-out probabilities for the train rows and, with --test, test probabilities.

Usage: python -I src/upi/exp_mlp.py <project_root> <tag> --features a.parquet,b.parquet [--labels models/x/y_clean.npy]
           [--device cpu|cuda] [--epochs 8] [--hidden 768] [--seeds 2] [--test]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from common import CATS, LABELS, load_xy  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("tag")
ap.add_argument("--features", required=True)
ap.add_argument("--labels", default="raw")
ap.add_argument("--device", default="cpu")
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--hidden", type=int, default=768)
ap.add_argument("--batch", type=int, default=4096)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--legit-frac", type=float, default=0.25)
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--jobs", type=int, default=8)
ap.add_argument("--test", action="store_true")
a = ap.parse_args()
ROOT = Path(a.root)
OUT = ROOT / "models" / a.tag
OUT.mkdir(parents=True, exist_ok=True)
K = len(LABELS)
torch.set_num_threads(a.jobs)
dev = torch.device(a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu")

X, y_raw, meta = load_xy(ROOT, tuple(a.features.split(",")))
tr = (meta.split == "train").values
y = y_raw.copy()
if a.labels != "raw":
    y[tr] = np.load(ROOT / a.labels)
month = meta.month.values
n = len(y)
onehot = pd.get_dummies(X[CATS], dtype="float32").values
num_cols = [c for c in X.columns if c not in CATS]
A = np.empty((n, len(num_cols)), dtype="float32")
flags = []
for j, c in enumerate(num_cols):                                   # column by column, to keep memory flat
    v = X[c].values.astype("float32")
    v[~np.isfinite(v)] = np.nan
    miss = np.isnan(v)
    v = np.sign(v) * np.log1p(np.abs(v))
    mu, sd = np.nanmean(v), np.nanstd(v)
    v = (v - mu) / (sd if sd > 0 else 1.0)
    v[miss] = 0.0
    A[:, j] = np.clip(v, -8, 8)
    if miss.mean() > 0.01:
        flags.append(miss.astype("float32"))
del X
A = np.hstack([A, onehot] + ([np.stack(flags, 1)] if flags else []))
print(f"inputs: {A.shape[1]} columns ({len(num_cols)} numeric, {onehot.shape[1]} one-hot, {len(flags)} missing flags) | device {dev}", flush=True)


def net(d):
    h = a.hidden
    return nn.Sequential(nn.Linear(d, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(0.1),
                         nn.Linear(h, h // 2), nn.BatchNorm1d(h // 2), nn.SiLU(), nn.Dropout(0.1),
                         nn.Linear(h // 2, h // 4), nn.BatchNorm1d(h // 4), nn.SiLU(),
                         nn.Linear(h // 4, K))


def fit_predict(fit_mask, pred_mask, seed):
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    keep = np.where(fit_mask & ((y != 0) | (rng.rand(n) < a.legit_frac)))[0]
    xt = torch.from_numpy(A[keep]).to(dev)
    yt = torch.from_numpy(y[keep].astype(np.int64)).to(dev)
    m = net(A.shape[1]).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(keep) // a.batch + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(a.epochs):
        m.train()
        perm = torch.randperm(len(keep), device=dev)
        for i in range(0, len(keep), a.batch):
            idx = perm[i:i + a.batch]
            if len(idx) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            loss_fn(m(xt[idx]), yt[idx]).backward()
            opt.step()
            sched.step()
    m.eval()
    rows = np.where(pred_mask)[0]
    out = np.empty((len(rows), K), dtype="float32")
    with torch.no_grad():
        for i in range(0, len(rows), 65536):
            out[i:i + 65536] = torch.softmax(m(torch.from_numpy(A[rows[i:i + 65536]]).to(dev)), 1).cpu().numpy()
    del xt, yt, m
    return out


t0 = time.time()
oof = np.zeros((int(tr.sum()), K), dtype="float32")
tr_idx = np.where(tr)[0]
for mth in ("2026-07", "2026-08", "2026-09"):
    hold = tr & (month == mth)
    oof[np.searchsorted(tr_idx, np.where(hold)[0])] = sum(fit_predict(tr & ~hold, hold, s) for s in range(a.seeds)) / a.seeds
    print(f"fold {mth} done ({time.time() - t0:.0f}s)", flush=True)
np.save(OUT / "oof.npy", oof)
if a.test:
    k = max(a.seeds, 3)
    np.save(OUT / "test_proba.npy", (sum(fit_predict(tr, ~tr, s) for s in range(k)) / k).astype("float32"))
    print(f"test done ({time.time() - t0:.0f}s)")
