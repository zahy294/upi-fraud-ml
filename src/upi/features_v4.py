"""Feature pack v4: timing between a small bait payment and the larger payback in the opposite direction.

A payback preceded by two baits is always fake_cashback; with one bait the delay separates it from phishing_collect.
Label-free, train+test together. Usage: python -I src/upi/features_v4.py <project_root>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from events import Events, DAY  # noqa: E402

ROOT = Path(sys.argv[1])
P = ROOT / "data/processed"
df = pd.read_parquet(P / "base_v0.parquet", columns=["txn_id", "t", "sender_id", "receiver_id", "receiver_type", "txn_type", "amount", "balance_before"])
t = df.t.values.astype(np.int64)
amt = df.amount.values
ba = df.balance_before.values / amt
snd = df.sender_id.str[2:].astype(np.int64).values
is_p2p = (df.receiver_type == "P2P").values
rnum = df.receiver_id.str[2:].astype(np.int64).values
rcv = np.where(is_p2p, rnum, rnum + 1_000_000)
pair = snd * 2_000_000 + rcv
pcode, puniq = pd.factorize(pair)
rev = np.where(is_p2p, rnum * 2_000_000 + snd, -1)
rcode = pd.Index(puniq).get_indexer(rev)                 # code of the opposite-direction pair, -1 if it never occurs
tiny = (amt <= 10) & is_p2p & (df.txn_type == "PAY").values
is11 = np.abs(ba - 1.1) < 0.0006
big = amt >= 500
W = 7 * DAY

F = pd.DataFrame({"txn_id": df.txn_id.values})
ev_tiny = Events(pcode, t, mask=tiny)
ev_11 = Events(pcode, t, mask=is11)
ev_big = Events(pcode, t, mask=big)
ev_all = Events(pcode, t)
# seen from the payback row: baits the receiver sent me earlier
F["bait_prev_dt_h"] = ev_tiny.prev_dt(rcode, t, W) / 3600
F["bait_cnt_before_7d"] = ev_tiny.count(rcode, t, -W, -1)
F["bait_cnt_before_1d"] = ev_tiny.count(rcode, t, -DAY, -1)
# seen from the bait row: what the receiver pays me back, and how many baits I sent
F["payback11_next_dt_h"] = ev_11.next_dt(rcode, t, W) / 3600
F["paybackbig_next_dt_h"] = ev_big.next_dt(rcode, t, W) / 3600
F["pair_tiny_cnt_7d"] = ev_tiny.count(pcode, t, -W, W)
F["pair_tiny_next_dt_h"] = ev_tiny.next_dt(pcode, t, W) / 3600
F["pair_tiny_prev_dt_h"] = ev_tiny.prev_dt(pcode, t, W) / 3600
# generic opposite-direction timing
F["rev_prev_dt_h"] = ev_all.prev_dt(rcode, t, W) / 3600
F["rev_next_dt_h"] = ev_all.next_dt(rcode, t, W) / 3600
F["is_tiny_p2p_pay"] = tiny.astype("float32")
num = [c for c in F.columns if c != "txn_id"]
F[num] = F[num].astype("float32")
F.to_parquet(P / "features_v4.parquet", index=False)
print("features_v4:", F.shape)
