"""Relational (cross-row) features on train+test combined and how well they separate labels.

Builds receiver bursts, money-forwarding, device sharing, pair/reverse-pair, refund history,
merchant concentration features with vectorised searchsorted windows. Saves them to
data/processed/relational_v0.parquet for reuse.

Usage: python -I scripts/eda/04_relational.py <project_root>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from common import LABELS, FRAUD, load, add_row_features, style, md_table  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(sys.argv[1])
FIG, TAB, PROC = ROOT / "reports/eda/figures", ROOT / "reports/eda/tables", ROOT / "data/processed"
PROC.mkdir(parents=True, exist_ok=True)
style()
SCALE, OFF = np.int64(2 ** 26), np.int64(2 ** 25)
MIN, HOUR, DAY = 60, 3600, 86400

df, ac, me = load(ROOT)
df = add_row_features(df, ac, me)
df = df.sort_values("t", kind="stable").reset_index(drop=True)
t = df.t.values.astype(np.int64)
amt = df.amount.values
snd = df.sender_id.str[2:].astype(np.int64).values                       # account number
is_p2p = (df.receiver_type == "P2P").values
rnum = df.receiver_id.str[2:].astype(np.int64).values
rcv = np.where(is_p2p, rnum, rnum + 1_000_000)                            # merchants in own range
dev = df.device_id.str[2:].astype(np.int64).values
rcv_acct = np.where(is_p2p, rnum, -1)                                     # receiver as an account, else -1


class Events:
    """Events sorted by (group, time) for O(log n) window queries."""

    def __init__(self, group, tt, vals=None, mask=None):
        if mask is not None:
            group, tt = group[mask], tt[mask]
            vals = vals[mask] if vals is not None else None
        o = np.lexsort((tt, group))
        self.K = group[o].astype(np.int64) * SCALE + tt[o] + OFF
        self.v = vals[o] if vals is not None else None
        self.n = len(self.K)

    def count(self, qg, qt, lo, hi):
        """# events of group qg with time in [qt+lo, qt+hi]. qg = -1 -> 0."""
        b = qg.astype(np.int64) * SCALE + qt + OFF
        c = np.searchsorted(self.K, b + hi, "right") - np.searchsorted(self.K, b + lo, "left")
        return np.where(qg >= 0, c, 0)

    def next_val(self, qg, qt, within):
        """value of first event strictly after qt and <= qt+within, else nan."""
        b = qg.astype(np.int64) * SCALE + qt + OFF
        i = np.searchsorted(self.K, b, "right")
        ic = np.minimum(i, self.n - 1)
        ok = (i < self.n) & (self.K[ic] <= b + within) & (qg >= 0)
        return np.where(ok, self.v[ic], np.nan)

    def prev_val(self, qg, qt, within):
        """value of last event strictly before qt and >= qt-within, else nan."""
        b = qg.astype(np.int64) * SCALE + qt + OFF
        i = np.searchsorted(self.K, b, "left") - 1
        ic = np.maximum(i, 0)
        ok = (i >= 0) & (self.K[ic] >= b - within) & (qg >= 0)
        return np.where(ok, self.v[ic], np.nan)


F = pd.DataFrame(index=df.index)
out_ev = Events(snd, t, amt)                    # outgoing payments keyed by sender account
in_ev = Events(rcv, t, amt)                     # incoming payments keyed by receiver
dev_ev = Events(dev, t)

# --- receiver bursts (symmetric windows; using the future is allowed: test is given as a whole) ---
F["rcv_cnt_1h"] = in_ev.count(rcv, t, -HOUR, HOUR)
F["rcv_cnt_24h"] = in_ev.count(rcv, t, -DAY, DAY)
F["rcv_cnt_total"] = df.groupby(rcv).txn_id.transform("size").values
day = (t // DAY)
F["rcv_uniq_snd_day"] = df.groupby([rcv, day]).sender_id.transform("nunique").values
F["rcv_uniq_snd_total"] = df.groupby(rcv).sender_id.transform("nunique").values
F["rcv_newpayee_share_day"] = df.groupby([rcv, day]).new_payee.transform("mean").values

# --- money moving on: does the (P2P) receiver send money out soon after? ---
F["rcv_out_30m"] = out_ev.count(rcv_acct, t, 1, 30 * MIN)
F["rcv_out_2h"] = out_ev.count(rcv_acct, t, 1, 2 * HOUR)
F["fwd_amt_ratio_30m"] = out_ev.next_val(rcv_acct, t, 30 * MIN) / amt
# --- money came in shortly before this payment (sender is a pass-through) ---
F["snd_in_before_30m"] = in_ev.count(snd, t, -30 * MIN, -1)
F["prev_in_amt_ratio_30m"] = amt / in_ev.prev_val(snd, t, 30 * MIN)
F["snd_out_cnt_10m"] = out_ev.count(snd, t, -10 * MIN, 10 * MIN)
F["snd_out_cnt_24h"] = out_ev.count(snd, t, -DAY, DAY)
F["snd_uniq_rcv_day"] = df.groupby([snd, day]).receiver_id.transform("nunique").values
F["snd_in_cnt_24h"] = in_ev.count(snd, t, -DAY, DAY)

# --- device ---
F["dev_cnt_5m"] = dev_ev.count(dev, t, -5 * MIN, 5 * MIN)
F["dev_cnt_1h"] = dev_ev.count(dev, t, -HOUR, HOUR)
F["dev_uniq_acct_total"] = df.groupby(dev).sender_id.transform("nunique").values
F["dev_uniq_acct_day"] = df.groupby([dev, day]).sender_id.transform("nunique").values
F["acct_uniq_dev_total"] = df.groupby(snd).device_id.transform("nunique").values

# --- pairs ---
pair = snd * 2_000_000 + rcv
pcode, puniq = pd.factorize(pair)
pair_ev = Events(pcode, t, amt)
F["pair_cnt_total"] = np.bincount(pcode)[pcode]
F["pair_cnt_7d"] = pair_ev.count(pcode, t, -7 * DAY, 7 * DAY)
F["pair_cnt_24h"] = pair_ev.count(pcode, t, -DAY, DAY)
rev = np.where(is_p2p, rcv_acct * 2_000_000 + snd, -1)
rcode = pd.Index(puniq).get_indexer(rev)          # -1 when the reverse pair never transacted
F["rev_pair_cnt_total"] = np.where(rcode >= 0, np.bincount(pcode, minlength=len(puniq))[np.maximum(rcode, 0)], 0)
small_ev = Events(pcode, t, amt, mask=amt <= 10)
big_ev = Events(pcode, t, amt, mask=amt >= 100)
F["rev_small_before_24h"] = small_ev.count(rcode, t, -DAY, -1)   # receiver sent me <=10 just before
F["rev_big_after_24h"] = big_ev.count(rcode, t, 1, DAY)          # receiver pays me >=100 just after

# --- refunds / sender history ---
is_ref = (df.txn_type == "REFUND").values
ref_ev = Events(snd, t, mask=is_ref)
F["snd_refund_cnt_30d"] = ref_ev.count(snd, t, -15 * DAY, 15 * DAY)
F["snd_refund_cnt_total"] = pd.Series(is_ref).groupby(snd).transform("sum").values
F["snd_collect_cnt_7d"] = Events(snd, t, mask=(df.txn_type == "COLLECT").values).count(snd, t, -7 * DAY, 7 * DAY)

# --- merchant concentration / round amounts ---
rnd = (df.amt_round_500.values == 1)
F["rcv_round500_cnt_7d"] = Events(rcv, t, mask=rnd).count(rcv, t, -7 * DAY, 7 * DAY)
F["pair_round500_cnt_total"] = pd.Series(rnd.astype(int)).groupby(pcode).transform("sum").values
top_share = df.groupby([rcv, "sender_id"]).size().rename("n").reset_index()
top_share["share"] = top_share.n / top_share.groupby("level_0" if "level_0" in top_share else top_share.columns[0]).n.transform("sum")
F["pair_share_of_rcv"] = F.pair_cnt_total / F.rcv_cnt_total

# --- time context ---
F["evening_pair_cnt_7d"] = Events(pcode, t, mask=(df.is_late_evening.values == 1)).count(pcode, t, -7 * DAY, 7 * DAY)

REL = list(F.columns)
F["txn_id"], F["split"], F["label"], F["month"] = df.txn_id, df.split, df.label, df.month
F.to_parquet(PROC / "relational_v0.parquet", index=False)
print("saved", PROC / "relational_v0.parquet", F.shape)

out = []
# ---------- density: are test entities sampled, or rows? ----------
dens = []
for nm, d in [("train Sep", df[(df.split == "train") & (df.month == "2026-09")]),
              ("test Oct", df[(df.split == "test") & (df.month == "2026-10")])]:
    dens.append({"set": nm, "rows": len(d), "uniq_senders": d.sender_id.nunique(),
                 "uniq_receivers": d.receiver_id.nunique(), "uniq_devices": d.device_id.nunique(),
                 "rows_per_sender": len(d) / d.sender_id.nunique(),
                 "rows_per_receiver": len(d) / d.receiver_id.nunique()})
dens = pd.DataFrame(dens).set_index("set")
out.append("## Density: September (train) vs October (test)\n" + md_table(dens, ".4g"))
print(dens)

tr = F[F.split == "train"]
te = F[F.split == "test"]
sep = tr[tr.month == "2026-09"]

# ---------- per-label means and medians ----------
mean = tr.groupby("label")[REL].mean().reindex(LABELS)
med = tr.groupby("label")[REL].median().reindex(LABELS)
out.append("\n## Relational features: mean by label (train)\n" + md_table(mean.T, ".3g"))
out.append("\n## Relational features: median by label (train)\n" + md_table(med.T, ".3g"))

# firing rates for the binary-ish signals
fires = pd.DataFrame({
    "rcv_cnt_1h>=5": tr.rcv_cnt_1h >= 5,
    "rcv_uniq_snd_day>=5": tr.rcv_uniq_snd_day >= 5,
    "rcv_out_30m>0": tr.rcv_out_30m > 0,
    "fwd_ratio_0.8-1.0": tr.fwd_amt_ratio_30m.between(0.8, 0.9999),
    "snd_in_before_30m>0": tr.snd_in_before_30m > 0,
    "prev_in_ratio_0.8-1.0": tr.prev_in_amt_ratio_30m.between(0.8, 0.9999),
    "dev_cnt_5m>=5": tr.dev_cnt_5m >= 5,
    "dev_uniq_acct_total>=3": tr.dev_uniq_acct_total >= 3,
    "pair_cnt_7d>=3": tr.pair_cnt_7d >= 3,
    "rev_pair>0": tr.rev_pair_cnt_total > 0,
    "rev_small_before_24h>0": tr.rev_small_before_24h > 0,
    "rev_big_after_24h>0": tr.rev_big_after_24h > 0,
    "snd_refund_cnt_30d>=3": tr.snd_refund_cnt_30d >= 3,
    "pair_round500>=2": tr.pair_round500_cnt_total >= 2,
    "evening_pair_cnt_7d>=3": tr.evening_pair_cnt_7d >= 3,
}).groupby(tr.label).mean().reindex(LABELS) * 100
out.append("\n## % of rows where a relational rule fires, by label (train)\n" + md_table(fires.T.round(2), ".3g"))
print(fires.T.round(1))

# same rules, Sep vs Oct firing on ALL rows (stability / density effect)
def rule_rates(d):
    return pd.Series({
        "rcv_cnt_1h>=5": (d.rcv_cnt_1h >= 5).mean(), "rcv_uniq_snd_day>=5": (d.rcv_uniq_snd_day >= 5).mean(),
        "rcv_out_30m>0": (d.rcv_out_30m > 0).mean(), "fwd_ratio_0.8-1.0": d.fwd_amt_ratio_30m.between(0.8, 0.9999).mean(),
        "snd_in_before_30m>0": (d.snd_in_before_30m > 0).mean(),
        "prev_in_ratio_0.8-1.0": d.prev_in_amt_ratio_30m.between(0.8, 0.9999).mean(),
        "dev_cnt_5m>=5": (d.dev_cnt_5m >= 5).mean(), "dev_uniq_acct_total>=3": (d.dev_uniq_acct_total >= 3).mean(),
        "pair_cnt_7d>=3": (d.pair_cnt_7d >= 3).mean(), "rev_pair>0": (d.rev_pair_cnt_total > 0).mean(),
        "rev_small_before_24h>0": (d.rev_small_before_24h > 0).mean(),
        "rev_big_after_24h>0": (d.rev_big_after_24h > 0).mean(),
        "snd_refund_cnt_30d>=3": (d.snd_refund_cnt_30d >= 3).mean(),
        "pair_round500>=2": (d.pair_round500_cnt_total >= 2).mean(),
        "evening_pair_cnt_7d>=3": (d.evening_pair_cnt_7d >= 3).mean()}) * 100
rr = pd.DataFrame({"Sep legit": rule_rates(sep[sep.label == "legit"]), "Sep all": rule_rates(sep),
                   "Oct all (test)": rule_rates(te)})
out.append("\n## Rule firing rate on all rows: Sep vs Oct (%)\n" + md_table(rr.round(3), ".4g"))
print(rr.round(3))

# ---------- per-label one-vs-legit AUC heatmap ----------
auc = pd.DataFrame(index=REL, columns=FRAUD, dtype=float)
for l in FRAUD:
    d = tr[tr.label.isin(["legit", l])]
    y = (d.label == l).values
    for c in REL:
        auc.loc[c, l] = roc_auc_score(y, d[c].fillna(-1).values)
sep_power = (auc - 0.5).abs()
out.append("\n## One-vs-legit AUC of each relational feature (train). 0.5 = useless, far from 0.5 = strong\n"
           + md_table(auc.round(3), ".3g"))
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(auc.values.astype(float), cmap="RdBu_r", vmin=0.2, vmax=0.8, aspect="auto")
ax.set_xticks(range(len(FRAUD)), FRAUD, rotation=40, ha="right")
ax.set_yticks(range(len(REL)), REL)
for i in range(len(REL)):
    for j in range(len(FRAUD)):
        v = auc.iat[i, j]
        if abs(v - 0.5) >= 0.1:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v - 0.5) > 0.25 else "#0b0b0b")
ax.grid(False)
ax.set_title("One-vs-legit AUC of relational features (labelled where |AUC-0.5| >= 0.1)")
fig.colorbar(im, ax=ax, shrink=0.6, label="AUC (red = higher for the fraud label)")
fig.tight_layout(); fig.savefig(FIG / "07_relational_auc_heatmap.png"); plt.close(fig)

# ---------- label noise / consistency ----------
noise = []
tr_full = df[df.split == "train"].join(F[REL])
# 1) inbound rows to a receiver within +-1h of a mule_account row: how many are labelled legit?
mule_rows = tr_full[tr_full.label == "mule_account"]
mule_rcv_t = Events(rcv[mule_rows.index], t[mule_rows.index])
near_mule = mule_rcv_t.count(rcv[tr_full.index], t[tr_full.index], -HOUR, HOUR) > 0
sub = tr_full[near_mule & (tr_full.receiver_type == "P2P")]
noise.append(("P2P rows paid to a receiver within ±1h of a mule_account inbound row",
              len(sub), sub.label.value_counts(normalize=True).mul(100).round(2).head(5).to_dict()))
# 2) rows from a device within ±5 min of a velocity_bot row
vb = tr_full[tr_full.label == "velocity_bot"]
vb_ev = Events(dev[vb.index], t[vb.index])
near_vb = vb_ev.count(dev[tr_full.index], t[tr_full.index], -5 * MIN, 5 * MIN) > 0
sub = tr_full[near_vb]
noise.append(("rows from the same device within ±5 min of a velocity_bot row", len(sub),
              sub.label.value_counts(normalize=True).mul(100).round(2).head(5).to_dict()))
# 3) REFUND rows by senders who have >=1 refund_abuse row
ra_snd = set(tr_full.loc[tr_full.label == "refund_abuse", "sender_id"])
sub = tr_full[tr_full.sender_id.isin(ra_snd) & (tr_full.txn_type == "REFUND")]
noise.append(("REFUND rows by senders who have any refund_abuse row", len(sub),
              sub.label.value_counts(normalize=True).mul(100).round(2).head(5).to_dict()))
# 4) all rows by senders who have >=1 refund_abuse row
sub = tr_full[tr_full.sender_id.isin(ra_snd)]
noise.append(("ALL rows by senders who have any refund_abuse row", len(sub),
              sub.label.value_counts(normalize=True).mul(100).round(2).head(5).to_dict()))
# 5) qr_vpa_mismatch == 1
sub = tr_full[tr_full.qr_vpa_mismatch == 1]
noise.append(("rows with qr_vpa_mismatch == 1", len(sub),
              sub.label.value_counts(normalize=True).mul(100).round(2).head(5).to_dict()))
# 6) rows paid to a P2P receiver that has >=1 fraud inbound row: label agreement among fraud rows
fr = tr_full[tr_full.label != "legit"]
g = fr.groupby("receiver_id").label.agg(["nunique", "size"])
multi = g[g["size"] >= 2]
noise.append(("receivers with >=2 fraud inbound rows having >1 distinct fraud label", len(multi),
              {"share_%": round((multi["nunique"] > 1).mean() * 100, 2)}))
g = fr.groupby("sender_id").label.agg(["nunique", "size"])
multi = g[g["size"] >= 2]
noise.append(("senders with >=2 fraud rows having >1 distinct fraud label", len(multi),
              {"share_%": round((multi["nunique"] > 1).mean() * 100, 2)}))
lines = ["\n## Label consistency / noise probes (train)"]
for desc, n, dist in noise:
    lines.append(f"- {desc}: n={n:,} -> {dist}")
out.append("\n".join(lines))
for x in noise:
    print(x)

# what does a sender's set of fraud labels look like (cross-type confusion)?
pairs = fr.groupby("sender_id").label.apply(lambda s: tuple(sorted(set(s))))
pairs = pairs[pairs.map(len) > 1].value_counts().head(15)
out.append("\n## Most common multi-label combinations on one sender (fraud rows only)\n" +
           "\n".join(f"- {k}: {v}" for k, v in pairs.items()))
pr = fr.groupby("receiver_id").label.apply(lambda s: tuple(sorted(set(s))))
pr = pr[pr.map(len) > 1].value_counts().head(15)
out.append("\n## Most common multi-label combinations on one receiver (fraud rows only)\n" +
           "\n".join(f"- {k}: {v}" for k, v in pr.items()))

# fraud receivers/devices from train reappearing in test
fr_rcv = set(fr.receiver_id); fr_snd = set(fr.sender_id); fr_dev = set(fr.device_id)
ted = df[df.split == "test"]
out.append("\n## Do train fraud entities reappear in October?\n"
           f"- test rows whose receiver had a fraud inbound row in train: {ted.receiver_id.isin(fr_rcv).mean()*100:.2f}% "
           f"(P2P only: {ted[ted.receiver_type=='P2P'].receiver_id.isin(fr_rcv).mean()*100:.2f}%)\n"
           f"- test rows whose sender had a fraud row in train: {ted.sender_id.isin(fr_snd).mean()*100:.2f}%\n"
           f"- test rows whose device had a fraud row in train: {ted.device_id.isin(fr_dev).mean()*100:.2f}%\n"
           f"- for reference, Sep rows whose receiver had a fraud inbound row in Jul-Aug: "
           f"{df[(df.split=='train')&(df.month=='2026-09')].receiver_id.isin(set(df[(df.split=='train')&(df.month<'2026-09')&(df.label!='legit')].receiver_id)).mean()*100:.2f}%")
sep_full = df[(df.split == "train") & (df.month == "2026-09")]
julaug_fr = df[(df.split == "train") & (df.month < "2026-09") & (df.label != "legit")]
for col in ["receiver_id", "sender_id", "device_id"]:
    seen = sep_full[col].isin(set(julaug_fr[col]))
    out.append(f"- Sep: label distribution of rows whose {col} had fraud in Jul-Aug (n={seen.sum():,}): "
               f"{sep_full[seen].label.value_counts(normalize=True).mul(100).round(1).head(4).to_dict()}")

(TAB / "04_relational.md").write_text("\n".join(out), encoding="utf-8")
print("wrote", TAB / "04_relational.md")
