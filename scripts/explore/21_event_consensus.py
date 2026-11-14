"""How much of the type confusion is disagreement INSIDE an event (fixable by a vote) and how much is a whole event
predicted as a sibling type (not fixable by a vote)?

Events = connected components over detected rows linked by: same sender within 3 h, same receiver (person account or
fixed-amount merchant session) within 24 h, same device within 10 min, opposite direction of the pair within 7 d,
forwarding parent (features_v5 root).
"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
TAG = sys.argv[2] if len(sys.argv) > 2 else "oof_c3"
WNAME = sys.argv[3] if len(sys.argv) > 3 else "weights_s5_c3.npy"
sys.path.insert(0, str(ROOT / "src/upi"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from common import LABELS, macro_f1, per_class  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
P = ROOT / "data/processed"
base = pd.read_parquet(P / "base_v0.parquet", columns=["txn_id", "t", "label", "split", "sender_id", "receiver_id", "device_id", "receiver_type"])
tr = (base.split == "train").values
b = base[tr].reset_index(drop=True)
y_raw = b.label.map({lab: i for i, lab in enumerate(LABELS)}).values.astype(int)
y = np.load(ROOT / "models/oof_raw/y_clean.npy")
oof = np.load(ROOT / "models" / TAG / "oof.npy").astype(float)
w = np.load(ROOT / "models" / TAG / WNAME)
F3 = pd.read_parquet(P / "features_v3.parquet")
F5 = pd.read_parquet(P / "features_v5.parquet", columns=["ch_band"])
s_flag = pd.read_parquet(P / "features_v2.parquet", columns=["S_flag"]).S_flag.values[tr] == 1
band_all = F3.band.values.astype(int)
p2p_all = (base.receiver_type == "P2P").values


def band_rules(pred, rows):
    bb = band_all[rows]
    S = np.stack([F3[f"snd_band_{k}_3h"].values[rows] for k in range(1, 12)], 1)
    R = np.stack([F3[f"rcv_band_{k}_24h"].values[rows] for k in range(1, 12)], 1)
    sb = np.where((S > 0).sum(1) == 1, S.argmax(1) + 1, 0)
    rb = np.where((R > 0).sum(1) == 1, R.argmax(1) + 1, 0)
    cb = F5.ch_band.values[rows].astype(int)
    p2p = p2p_all[rows]
    out = np.where(bb > 0, bb, pred)
    free = (bb == 0) & (pred > 0)
    drain = free & np.isin(pred, [4, 5]) & np.isin(sb, [4, 5])
    fanin = free & np.isin(pred, [1, 2, 3]) & p2p & np.isin(rb, [1, 2, 3])
    shop = free & np.isin(pred, [7, 8, 11]) & ~p2p & np.isin(rb, [7, 8, 11])
    chain = free & (cb > 0)
    out = np.where(drain, sb, out)
    out = np.where(fanin | shop, rb, out)
    out = np.where(chain, cb, out)
    return out


pred = band_rules((oof * w).argmax(1), tr)
print(f"start: macro-F1 vs cleaned {macro_f1(y, pred):.4f} | vs raw {macro_f1(y_raw, pred):.4f}")

t = b.t.values.astype(np.int64)
snd = b.sender_id.str[2:].astype(np.int64).values
is_p2p = (b.receiver_type == "P2P").values
rnum = b.receiver_id.str[2:].astype(np.int64).values
rcv = np.where(is_p2p, rnum, rnum + 1_000_000)
dev = b.device_id.str[2:].astype(np.int64).values
det = np.where(pred > 0)[0]
print("detected rows:", len(det))


def links(key, window, ok=None):
    """Pairs (i, j) of detected rows with the same key and |t_i - t_j| <= window (consecutive in time is enough)."""
    idx = det if ok is None else det[ok[det]]
    o = idx[np.lexsort((t[idx], key[idx]))]
    same = (key[o][1:] == key[o][:-1]) & (t[o][1:] - t[o][:-1] <= window)
    return o[:-1][same], o[1:][same]


def components(spec):
    I, J = [], []
    for key, window, ok in spec:
        i, j = links(key, window, ok)
        I.append(i)
        J.append(j)
    I, J = np.concatenate(I), np.concatenate(J)
    n = len(b)
    g = coo_matrix((np.ones(len(I)), (I, J)), shape=(n, n))
    _, comp = connected_components(g, directed=False)
    return comp


HOUR, DAY = 3600, 86400
rcv_ok = is_p2p | s_flag
pairkey = np.minimum(snd, rcv) * 2_000_000 + np.maximum(snd, rcv)          # unordered pair (person-to-person only)
# an account as a node: a person account that both receives and sends links fan-in rows with its onward transfers
specs = {
    "sender 3h": [(snd, 3 * HOUR, None)],
    "receiver 24h": [(rcv, DAY, rcv_ok)],
    "device 10m": [(dev, 600, None)],
    "pair 7d (either direction)": [(pairkey, 7 * DAY, is_p2p)],
    "sender 7d": [(snd, 7 * DAY, None)],
    "sender3h+receiver24h+device10m+pair7d": [(snd, 3 * HOUR, None), (rcv, DAY, rcv_ok), (dev, 600, None), (pairkey, 7 * DAY, is_p2p)],
    "sender7d+receiver24h+device10m+pair7d": [(snd, 7 * DAY, None), (rcv, DAY, rcv_ok), (dev, 600, None), (pairkey, 7 * DAY, is_p2p)],
}
Pw = oof * w
for name, spec in specs.items():
    comp = components(spec)
    c = comp[det]
    df = pd.DataFrame({"c": c, "pred": pred[det], "y": y[det], "yr": y_raw[det]})
    size = df.groupby("c").c.transform("size").values
    # majority of the hard predictions inside the event
    maj = df.groupby(["c", "pred"]).size().reset_index(name="n").sort_values(["c", "n"], ascending=[True, False]).drop_duplicates("c").set_index("c").pred
    vote = maj.reindex(c).values
    # soft: sum of log weighted probabilities over the fraud classes
    L = np.log(np.maximum(Pw[det][:, 1:], 1e-6))
    S = pd.DataFrame(L).groupby(c).transform("sum").values
    soft = S.argmax(1) + 1
    on_band = band_all[tr][det] > 0
    out_v, out_s = pred.copy(), pred.copy()
    out_v[det] = np.where(on_band, pred[det], vote)
    out_s[det] = np.where(on_band, pred[det], soft)
    mixed = df.groupby("c").pred.transform("nunique").values > 1
    wrong = (df.y > 0) & (df.pred != df.y)
    print(f"\n[{name}] events {df.c.nunique()}, multi-row share of rows {np.mean(size > 1):.3f}, rows in events with mixed predictions {mixed.sum()}"
          f" | type-confused rows {int(wrong.sum())}, of them in mixed events {int((wrong & mixed).sum())}")
    print(f"   hard vote: vs cleaned {macro_f1(y, out_v):.4f} | vs raw {macro_f1(y_raw, out_v):.4f} (changed {int((out_v != pred).sum())})"
          f"   soft vote: vs cleaned {macro_f1(y, out_s):.4f} | vs raw {macro_f1(y_raw, out_s):.4f} (changed {int((out_s != pred).sum())})")
    # oracle: every event takes the majority TRUE label of its rows -> what a perfect event-level typer would score
    majy = df[df.y > 0].groupby(["c", "y"]).size().reset_index(name="n").sort_values(["c", "n"], ascending=[True, False]).drop_duplicates("c").set_index("c").y
    orc = pred.copy()
    orc[det] = np.where(np.isnan(majy.reindex(c).values), pred[det], majy.reindex(c).fillna(0).values).astype(int)
    print(f"   oracle (event's majority cleaned label): {macro_f1(y, orc):.4f}; label purity of multi-row events: "
          f"{(df[df.y > 0].groupby('c').y.agg(lambda s: s.value_counts().iloc[0] / len(s))[lambda s: s.index.isin(df.c[size > 1])]).mean():.3f}")
