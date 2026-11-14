"""Stage-2 event-neighbour probability propagation features (features_stage2.parquet).

Computes leave-one-out event-neighbour sums and means of Stage-1 probabilities across:
  1. Same sender within ±3h (drain bursts: sim_swap, account_takeover)
  2. Same sender within ±7d (repeat sender campaigns: loan_app_extortion, refund_abuse)
  3. Same event receiver (P2P receiver or S_flag merchant session) within ±24h (fan-in / merchant session)
  4. Same device within ±10m and ±24h (velocity_bot bursts, shared device rings)
  5. Opposite-direction (receiver, sender) pair within ±7d (bait <-> payback: phishing_collect, fake_cashback)
  6. Same-direction (sender, receiver) pair within ±7d (repeat baits, repeat extortion/refunds)
  7. Forwarding chain peers (same chain root) + incoming P2P payments to chain root sender within ±48h (mule_account vs layering_chain)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from common import LABELS
from events import Events, MIN, HOUR, DAY

t0 = time.time()
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
P = ROOT / "data/processed"

df = pd.read_parquet(
    P / "base_v0.parquet",
    columns=["txn_id", "t", "sender_id", "receiver_id", "device_id", "receiver_type", "amount", "split"],
)
f2 = pd.read_parquet(P / "features_v2.parquet", columns=["S_flag"])
n = len(df)
tr = (df.split == "train").values

# Load 5-model Stage-1 probability blend (leave-one-month-out on train, full fit on test)
oof = (
    0.35 * np.load(ROOT / "models/oof_clean/oof.npy")
    + 0.25 * np.load(ROOT / "models/oof_v6_lgb/oof.npy")
    + 0.15 * np.load(ROOT / "models/oof_clean_xgb/oof.npy")
    + 0.10 * np.load(ROOT / "models/oof_v6_xgb/oof.npy")
    + 0.15 * np.load(ROOT / "models/oof_clean_cat/oof.npy")
).astype(np.float64)

Pte = (
    0.35 * np.load(ROOT / "models/oof_clean/test_proba.npy")
    + 0.25 * np.load(ROOT / "models/oof_v6_lgb/test_proba.npy")
    + 0.15 * np.load(ROOT / "models/oof_clean_xgb/test_proba.npy")
    + 0.10 * np.load(ROOT / "models/oof_v6_xgb/test_proba.npy")
    + 0.15 * np.load(ROOT / "models/oof_clean_cat/test_proba.npy")
).astype(np.float64)

P1 = np.zeros((n, 12), dtype=np.float64)
P1[tr] = oof
P1[~tr] = Pte
pf = 1.0 - P1[:, 0]

t = df.t.values.astype(np.int64)
snd = df.sender_id.str[2:].astype(np.int64).values
is_p2p = (df.receiver_type == "P2P").values
rnum = df.receiver_id.str[2:].astype(np.int64).values
rcv = np.where(is_p2p, rnum, rnum + 1_000_000)
s_flag = f2.S_flag.values == 1
rcv_evt = np.where(is_p2p | s_flag, rcv, -1)
rcv_p2p = np.where(is_p2p, rnum, -1)
dev = df.device_id.str[2:].astype(np.int64).values

pair = snd * 2_000_000 + rcv
pcode, puniq = pd.factorize(pair)
rev = np.where(is_p2p, rnum * 2_000_000 + snd, -1)
rcode = pd.Index(puniq).get_indexer(rev)

out = {"txn_id": df.txn_id.values}


def add_window_channel(prefix, g_key, g_query, win, self_exclude=True, mask=None):
    ev_cnt = Events(g_key, t, mask=mask)
    cnt = ev_cnt.count(g_query, t, -win, win).astype(np.float64)
    if self_exclude:
        self_in = np.ones(n, dtype=np.float64) if mask is None else mask.astype(np.float64)
        cnt = np.maximum(cnt - self_in, 0.0)
    out[f"s2_{prefix}_cnt"] = cnt.astype(np.float32)
    denom = np.maximum(cnt, 1.0)

    ev_pf = Events(g_key, t, pf, mask=mask)
    s_pf = ev_pf.sum(g_query, t, -win, win)
    if self_exclude:
        s_pf = np.maximum(s_pf - pf * self_in, 0.0)
    out[f"s2_{prefix}_pf_sum"] = s_pf.astype(np.float32)
    out[f"s2_{prefix}_pf_mean"] = np.where(cnt > 0, s_pf / denom, 0.0).astype(np.float32)

    mean_mat = np.zeros((n, 11), dtype=np.float32)
    for k in range(1, 12):
        pk = P1[:, k]
        ev_k = Events(g_key, t, pk, mask=mask)
        sk = ev_k.sum(g_query, t, -win, win)
        if self_exclude:
            sk = np.maximum(sk - pk * self_in, 0.0)
        mk = np.where(cnt > 0, sk / denom, 0.0).astype(np.float32)
        out[f"s2_{prefix}_p{k}_mean"] = mk
        out[f"s2_{prefix}_p{k}_sum"] = sk.astype(np.float32)
        mean_mat[:, k - 1] = mk
    out[f"s2_{prefix}_top_prob"] = mean_mat.max(axis=1)
    out[f"s2_{prefix}_top_cls"] = np.where(cnt > 0, mean_mat.argmax(axis=1) + 1, 0).astype(np.float32)


# 1. Sender ±3h (drain bursts)
add_window_channel("snd3h", snd, snd, 3 * HOUR, self_exclude=True)
# 2. Sender ±7d (multi-day sender campaigns)
add_window_channel("snd7d", snd, snd, 7 * DAY, self_exclude=True)
# 3. Event receiver (P2P or S_flag merchant session) ±24h
add_window_channel("rcvevt24h", rcv_evt, rcv_evt, DAY, self_exclude=True, mask=(rcv_evt >= 0))
# 4. Device ±10m and ±24h
add_window_channel("dev10m", dev, dev, 10 * MIN, self_exclude=True)
add_window_channel("dev24h", dev, dev, DAY, self_exclude=True)
# 5. Opposite-direction pair (rcode) ±7d (no self_exclude needed since opposite direction)
add_window_channel("opp7d", pcode, rcode, 7 * DAY, self_exclude=False)
# 6. Same-direction pair (pcode) ±7d
add_window_channel("pair7d", pcode, pcode, 7 * DAY, self_exclude=True)

# 7. Forwarding chain reconstruction (same logic as features_v5.py) + incoming P2P to root sender
df_row = np.arange(n)
W_ch, LO, HI = 7200, 0.90, 1.0
q = pd.DataFrame({"row": df_row, "t": t, "k": df.sender_id.values, "amount": df.amount.values}).sort_values("t")
inn = pd.DataFrame({
    "p_row": df_row[is_p2p],
    "p_t": t[is_p2p],
    "k": df.receiver_id.values[is_p2p],
    "p_amt": df.amount.values[is_p2p],
}).sort_values("p_t")
m = pd.merge_asof(q, inn, left_on="t", right_on="p_t", by="k", direction="backward", allow_exact_matches=False, tolerance=W_ch).set_index("row")
ratio = (m.amount / m.p_amt).reindex(df_row).values
par = np.where((ratio >= LO) & (ratio <= HI), m.p_row.reindex(df_row).fillna(-1).values, -1).astype(np.int64)

q2 = pd.DataFrame({"row": df_row[is_p2p], "t": t[is_p2p], "k": df.receiver_id.values[is_p2p], "amount": df.amount.values[is_p2p]}).sort_values("t")
out_df = pd.DataFrame({"c_row": df_row, "c_t": t, "k": df.sender_id.values, "c_amt": df.amount.values}).sort_values("c_t")
m2 = pd.merge_asof(q2, out_df, left_on="t", right_on="c_t", by="k", direction="forward", allow_exact_matches=False, tolerance=W_ch).set_index("row")
r2 = (m2.c_amt / m2.amount).reindex(df_row).values
chi = np.where((r2 >= LO) & (r2 <= HI), m2.c_row.reindex(df_row).fillna(-1).values, -1).astype(np.int64)


def walk(ptr, hops=8):
    depth = np.zeros(n, np.int16)
    end = np.arange(n)
    cur = ptr.copy()
    for _ in range(hops):
        has = cur >= 0
        depth += has
        end = np.where(has, cur, end)
        cur = np.where(has, ptr[np.maximum(cur, 0)], -1)
    return depth, end


d_back, root = walk(par)
d_fwd, leaf = walk(chi)
in_chain = (d_back + d_fwd) > 0

# Group by chain root to compute leave-one-out chain peer probabilities
chain_cnt = np.zeros(n, dtype=np.float64)
np.add.at(chain_cnt, root[in_chain], 1.0)
peer_cnt = np.where(in_chain, np.maximum(chain_cnt[root] - 1.0, 0.0), 0.0)
out["s2_chain_peer_cnt"] = peer_cnt.astype(np.float32)

for k in range(1, 12):
    pk = P1[:, k]
    tot_k = np.zeros(n, dtype=np.float64)
    np.add.at(tot_k, root[in_chain], pk[in_chain])
    peer_sum = np.where(in_chain, np.maximum(tot_k[root] - pk, 0.0), 0.0)
    peer_mean = np.where(peer_cnt > 0, peer_sum / np.maximum(peer_cnt, 1.0), 0.0)
    out[f"s2_chain_p{k}_mean"] = peer_mean.astype(np.float32)

# Incoming P2P payments to the chain root's sender within ±48h (separates mule_account chains from layering_chain!)
root_snd = snd[root]
root_t = t[root]
ev_in_cnt = Events(rcv_p2p, t, mask=(rcv_p2p >= 0))
in_cnt = ev_in_cnt.count(root_snd, root_t, -48 * HOUR, 48 * HOUR).astype(np.float64)
out["s2_root_in_p2p_cnt"] = in_cnt.astype(np.float32)
for k in (1, 2, 3, 6, 9):
    pk = P1[:, k]
    ev_in_k = Events(rcv_p2p, t, pk, mask=(rcv_p2p >= 0))
    in_sum = ev_in_k.sum(root_snd, root_t, -48 * HOUR, 48 * HOUR)
    out[f"s2_root_in_p{k}_sum"] = in_sum.astype(np.float32)
    out[f"s2_root_in_p{k}_mean"] = np.where(in_cnt > 0, in_sum / np.maximum(in_cnt, 1.0), 0.0).astype(np.float32)

# Also check if any incoming P2P payment to root_snd within ±48h sat on a band device!
f3 = pd.read_parquet(P / "features_v3.parquet", columns=["band"])
band = f3.band.values.astype(int)
for k in (1, 2, 3):
    ev_b = Events(rcv_p2p, t, (band == k).astype(np.float64), mask=(rcv_p2p >= 0))
    out[f"s2_root_in_band_{k}"] = ev_b.sum(root_snd, root_t, -48 * HOUR, 48 * HOUR).astype(np.float32)

F_s2 = pd.DataFrame(out)
F_s2.to_parquet(P / "features_stage2.parquet", index=False)
print(f"Built features_stage2.parquet: {F_s2.shape} in {time.time() - t0:.1f}s")
