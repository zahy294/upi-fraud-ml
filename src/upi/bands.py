"""Device-number bands.

Devices created by the data generator for fraud rows get consecutive ID numbers, one contiguous band per fraud
type, in label order; ordinary new devices follow in a fully occupied block. Train and test were generated in
separate runs, so each split has its own block. Bot bursts use a shared pool of device numbers (420001-423000).

This script finds the band edges (train: from labels; test: from the train proportions, refined with stage-1
probabilities) and writes data/processed/features_v3.parquet:
  band            0 = none, else the label index of the band the device number falls in
  dev_ext_legit   1 if the device is an ordinary new device (not a fraud band, not any account's home device)
  snd_band_<k>_3h / rcv_band_<k>_24h   rows of the same sender / receiver nearby that sit on band-k devices

Usage: python -I src/upi/bands.py <project_root> <stage1_tag_with_test_proba>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import LABELS  # noqa: E402
from events import Events, HOUR, DAY  # noqa: E402

ROOT, TAG = Path(sys.argv[1]), sys.argv[2]
P = ROOT / "data/processed"
N_ACCOUNTS = 420000
BOT_POOL = (420001, 423000)
ORDER = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]            # label indices in band order (same as LABELS order)
K = len(ORDER)
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)

df = pd.read_parquet(P / "base_v0.parquet", columns=["txn_id", "t", "sender_id", "receiver_id", "device_id", "receiver_type", "label", "split"])
dn = df.device_id.str[2:].astype(np.int64).values
is_tr = (df.split == "train").values
y = df.label.map({lab: i for i, lab in enumerate(LABELS)}).fillna(-1).astype(int).values


def legit_block_start(nums, lo, run=40):
    """First device number >= lo from which `run` consecutive numbers are all present."""
    present = np.zeros(nums.max() - lo + 2, bool)
    present[nums[nums >= lo] - lo] = True
    c = np.convolve(present.astype(int), np.ones(run, int), "valid")
    return lo + int(np.argmax(c == run))


def segment(score, prior=None, sd=None):
    """Split positions 0..n-1 into K ordered segments maximising summed score[pos, k] minus a Gaussian penalty on
    the K-1 cut points around `prior`. Returns cut points c_1..c_{K-1}; segment k covers [c_k, c_{k+1})."""
    n = score.shape[0]
    cs = np.vstack([np.zeros(K), np.cumsum(score, 0)])                 # cs[j, k] = sum of score[:j, k]
    pos = np.arange(n + 1)
    best = cs[:, 0].copy()                                             # best total with segment 0 ending at j
    back = []
    for k in range(1, K):
        pen = -((pos - prior[k - 1]) ** 2) / (2 * sd ** 2) if prior is not None else np.zeros(n + 1)
        cand = best - cs[:, k] + pen                                   # choose cut i <= j
        idx = np.zeros(n + 1, int)
        m = np.empty(n + 1)
        cur = 0
        for j in range(n + 1):
            if cand[j] > cand[cur]:
                cur = j
            idx[j] = cur
            m[j] = cand[cur]
        best = m + cs[:, k]
        back.append(idx)
    cuts = [n]
    for idx in reversed(back):
        cuts.append(int(idx[cuts[-1]]))
    return cuts[::-1][:-1]


# ---------------- train block: edges from labels ----------------
tr_ext = is_tr & (dn > BOT_POOL[1])
start_tr = BOT_POOL[1] + 1
end_tr = legit_block_start(dn[tr_ext], start_tr) - 1                   # last fraud-band number
width_tr = end_tr - start_tr + 1
sc = np.zeros((width_tr, K))
m = tr_ext & (dn <= end_tr) & (y > 0)
np.add.at(sc, (dn[m] - start_tr, y[m] - 1), 1.0)                        # +1 where a row label matches the segment type
cuts_tr = segment(sc)
edges_tr = [start_tr] + [start_tr + c for c in cuts_tr] + [end_tr + 1]
rel = [(e - start_tr) / width_tr for e in edges_tr]
print("train fraud block:", start_tr, "-", end_tr, "width", width_tr)

# ---------------- test block: proportional prior, refined with stage-1 probabilities ----------------
te_ext = (~is_tr) & (dn > BOT_POOL[1])
start_te = int(dn[is_tr].max()) + 1
assert dn[te_ext].min() >= start_te, "test devices outside the bot pool should start after the last train device"
end_te = legit_block_start(dn[te_ext], start_te) - 1
width_te = end_te - start_te + 1
prior = np.array([r * width_te for r in rel[1:-1]])
Pte = np.load(ROOT / "models" / TAG / "test_proba.npy").astype(float)   # rows of the test split, in base order
assert len(Pte) == (~is_tr).sum()
te_idx = np.where(~is_tr)[0]
inblk = (dn[te_idx] >= start_te) & (dn[te_idx] <= end_te)
Q = Pte[inblk][:, 1:].copy()
drain = Q[:, 3] + Q[:, 4]                                               # sim_swap + account_takeover look alike to the model
Q[:, 3] = drain
Q[:, 4] = drain
sc = np.zeros((width_te, K))
np.add.at(sc, dn[te_idx][inblk] - start_te, np.log(np.maximum(Q, 0.05)))
cuts_te = segment(sc, prior=prior, sd=0.02 * width_te)
edges_te = [start_te] + [start_te + c for c in cuts_te] + [end_te + 1]
print("test fraud block:", start_te, "-", end_te, "width", width_te, "(train/test width ratio %.2f)" % (width_tr / width_te))
T = pd.DataFrame({"type": [LABELS[k] for k in ORDER], "train_start": edges_tr[:-1], "train_width": np.diff(edges_tr),
                  "test_start": edges_te[:-1], "test_width": np.diff(edges_te),
                  "test_start_prior": [start_te] + [int(round(start_te + p)) for p in prior]})
print(T.to_string(index=False))

# ---------------- band id per row ----------------
band = np.zeros(len(df), np.int8)
for edges, mask in ((edges_tr, is_tr), (edges_te, ~is_tr)):
    for k in range(K):
        band[mask & (dn >= edges[k]) & (dn < edges[k + 1])] = ORDER[k]
band[(dn >= BOT_POOL[0]) & (dn <= BOT_POOL[1])] = 2                     # velocity_bot pool, shared by both runs
ext_legit = ((dn > N_ACCOUNTS) & (band == 0)).astype(np.int8)

# diagnostics
trb = pd.crosstab(pd.Series(band[is_tr], name="band"), pd.Series(np.array(LABELS)[y[is_tr]], name="label"))
own = np.array([trb.loc[b, LABELS[b]] if b > 0 and LABELS[b] in trb.columns else 0 for b in trb.index])
print("\ntrain rows per band: n, own-label share, legit-label share")
print(pd.DataFrame({"n": trb.sum(1), "own": (own / trb.sum(1)).round(3), "legit": (trb["legit"] / trb.sum(1)).round(3)}).iloc[1:].to_string())
pred = Pte.argmax(1)
teb = pd.crosstab(pd.Series(band[~is_tr], name="band"), pd.Series(np.array(LABELS)[pred], name="stage1_argmax"))
print("\ntest rows per band vs stage-1 argmax (rows):")
print(teb.iloc[1:].to_string())
print("ordinary new devices: train rows %d (fraud-labelled %.4f) | test rows %d" % (
    (ext_legit[is_tr] == 1).sum(), (y[is_tr][ext_legit[is_tr] == 1] > 0).mean(), (ext_legit[~is_tr] == 1).sum()))

# ---------------- features ----------------
t = df.t.values.astype(np.int64)
snd = df.sender_id.str[2:].astype(np.int64).values
is_p2p = (df.receiver_type == "P2P").values
rnum = df.receiver_id.str[2:].astype(np.int64).values
rcv = np.where(is_p2p, rnum, rnum + 1_000_000)
F = pd.DataFrame({"txn_id": df.txn_id.values, "band": band.astype("float32"), "dev_ext_legit": ext_legit.astype("float32")})
anyb = band > 0
F["snd_band_any_3h"] = Events(snd, t, mask=anyb).count(snd, t, -3 * HOUR, 3 * HOUR).astype("float32")
F["rcv_band_any_24h"] = Events(rcv, t, mask=anyb).count(rcv, t, -DAY, DAY).astype("float32")
for k in ORDER:
    mk = band == k
    F[f"snd_band_{k}_3h"] = Events(snd, t, mask=mk).count(snd, t, -3 * HOUR, 3 * HOUR).astype("float32")
    F[f"rcv_band_{k}_24h"] = Events(rcv, t, mask=mk).count(rcv, t, -DAY, DAY).astype("float32")
F.to_parquet(P / "features_v3.parquet", index=False)
(P / "band_edges.json").write_text(json.dumps({"train": [int(e) for e in edges_tr], "test": [int(e) for e in edges_te],
                                                 "order": [LABELS[k] for k in ORDER], "bot_pool": BOT_POOL}, indent=1))
print("\nfeatures_v3:", F.shape)
