# How uncertain is each October band edge, and which rows would change type if it moved?
# Run from a from-scratch run folder (needs data/processed and models/s1). Read-only.
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path('src/upi').resolve()))
from common import LABELS
P = Path('data/processed'); K = 11
df = pd.read_parquet(P/'base_v0.parquet', columns=['txn_id','t','sender_id','receiver_id','device_id','label','split'])
dn = df.device_id.str[2:].astype(np.int64).values
is_tr = (df.split=='train').values
E = json.load(open(P/'band_edges.json')); etr, ete = E['train'], E['test']
Pte = np.load('models/s1/test_proba.npy').astype(float); Ptr = np.load('models/s1/oof.npy').astype(float)
te_idx = np.where(~is_tr)[0]; tr_idx = np.where(is_tr)[0]
kind_all = pd.read_parquet(P/'features_v1.parquet', columns=['snd_drain_kind']).snd_drain_kind.values.astype(int)
def band_scores(probs, kind, dnum, start, width, lam_mule, lam_sim, occ):
    Q = probs[:, 1:].copy(); mule = Q[:, 0].copy(); drain = Q[:, 3] + Q[:, 4]; sim_shaped = kind == 1
    Q[:, 1] = Q[:, 1] + 0.1 * mule; Q[:, 3] = drain; Q[:, 4] = np.where(sim_shaped, 0.1 * drain, drain)
    sim = np.where(sim_shaped, drain, 0.0)
    sc = np.zeros((width, K)); np.add.at(sc, dnum - start, np.log(np.maximum(Q, 0.05)))
    for col, anchor, lam in ((0, mule, lam_mule), (3, sim, lam_sim)):
        has = np.zeros(width, bool); has[(dnum - start)[anchor >= 0.5]] = True; sc[~has, col] += np.log(1 - lam)
    present = np.zeros(width, bool); present[dnum - start] = True
    sc += np.where(present[:, None], np.log(occ)[None, :], np.log(1 - occ)[None, :])
    return sc
in_mule = (dn[tr_idx] >= etr[0]) & (dn[tr_idx] < etr[1]); lam_mule = float((Ptr[in_mule, 1] >= 0.5).sum() / (etr[1] - etr[0]))
in_sim = (dn[tr_idx] >= etr[3]) & (dn[tr_idx] < etr[4])
lam_sim = float(((Ptr[in_sim, 4] + Ptr[in_sim, 5] >= 0.5) & (kind_all[tr_idx][in_sim] == 1)).sum() / (etr[4] - etr[3]))
d_tr = dn[tr_idx]
occ = np.clip(np.array([len(np.unique(d_tr[(d_tr >= etr[k]) & (d_tr < etr[k + 1])])) / (etr[k + 1] - etr[k]) for k in range(K)]), 0.02, 0.98)
print('occupancy per train band', occ.round(3).tolist())
start, end = ete[0], ete[-1]-1; width = end-start+1
inblk = (dn[te_idx] >= start) & (dn[te_idx] <= end)
sc = band_scores(Pte[inblk], kind_all[te_idx][inblk], dn[te_idx][inblk], start, width, lam_mule, lam_sim, occ)
wtr = etr[-1]-etr[0]; rel = [(e-etr[0])/wtr for e in etr]; prior = np.array([r*width for r in rel[1:-1]]); sd = 0.02*width
cs = np.vstack([np.zeros(K), np.cumsum(sc, 0)])
cuts0 = [e-start for e in ete]            # 0 .. width
def total(cuts):
    s = sum(cs[cuts[k+1], k] - cs[cuts[k], k] for k in range(K))
    return s - (((np.array(cuts[1:-1]) - prior)**2)/(2*sd**2)).sum()
best = total(cuts0)
# test rows in the block
sub = pd.read_csv('out_real/s2_typ_all.csv'); lab2i = {l:i for i,l in enumerate(LABELS)}
idc, predcol = sub.columns[0], sub.columns[1]; sub = sub.rename(columns={idc: 'txn_id'})
pred = df.loc[~is_tr, ['txn_id']].merge(sub, on='txn_id', how='left')[predcol].map(lab2i).values
T = pd.DataFrame({'dnum': dn[te_idx][inblk], 'pred': pred[inblk], 's1': Pte[inblk].argmax(1), 's1p': Pte[inblk].max(1).round(2),
                  'pfraud': (1-Pte[inblk][:,0]).round(2), 'kind': kind_all[te_idx][inblk]})
T['ndev'] = T.groupby('dnum').dnum.transform('size')
print('rows in test fraud block', len(T), 'devices', T.dnum.nunique(), 'width', width)
short = ['legit','mule','bot','layer','sim','ato','phish','qr','refund','cashb','loan','collu']
for k in range(1, K):
    lo, hi = cuts0[k-1], cuts0[k+1]
    prof = np.array([total(cuts0[:k] + [c] + cuts0[k+1:]) for c in range(lo, hi+1)]) - best
    ok = np.where(prof >= -3.0)[0] + lo           # positions within 3 log units of the best
    ok1 = np.where(prof >= -1.0)[0] + lo
    a, b = start+ok.min(), start+ok.max()
    n_rows = ((T.dnum >= a) & (T.dnum < b)).sum()
    a1, b1 = start+ok1.min(), start+ok1.max()
    n1 = ((T.dnum >= a1) & (T.dnum < b1)).sum()
    print(f'\nedge {short[k]}|{short[k+1]} at {ete[k]} (prior {start+prior[k-1]:.0f}): within 1 unit [{a1},{b1}] rows {n1}; within 3 units [{a},{b}] rows {n_rows}; profile max {prof.max():.2f}')
    w = T[(T.dnum >= min(a, ete[k]-12)) & (T.dnum < max(b, ete[k]+12))].sort_values('dnum')
    print('   ' + ' '.join(f"{r.dnum-ete[k]:+d}:{short[r.pred]}/{short[r.s1]}{r.s1p:.1f}{'*' if r.ndev>1 else ''}" for r in w.itertuples()))
