"""Variant builder: same logic as src/upi/make_submission.py, with optional blend weights and a multiplier on the fraud-class weights."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / 'src/upi')); sys.path.insert(0, str(ROOT / 'colab'))
from common import LABELS, macro_f1
from bench import fit_weights_fast
base = pd.read_parquet(ROOT / 'data/processed/base_v0.parquet', columns=['txn_id', 'label', 'split', 'receiver_type'])
tr = (base.split == 'train').values
y = np.load(ROOT / 'models/oof_raw/y_clean.npy')
F3 = pd.read_parquet(ROOT / 'data/processed/features_v3.parquet')
band = F3.band.values.astype(int)
cbA = pd.read_parquet(ROOT / 'data/processed/features_v5.parquet', columns=['ch_band']).ch_band.values.astype(int)
p2pA = base.receiver_type.values == 'P2P'
S = np.stack([F3[f'snd_band_{k}_3h'].values for k in range(1, 12)], 1); R = np.stack([F3[f'rcv_band_{k}_24h'].values for k in range(1, 12)], 1)
sbA = np.where((S > 0).sum(1) == 1, S.argmax(1) + 1, 0); rbA = np.where((R > 0).sum(1) == 1, R.argmax(1) + 1, 0)
def rules(pred, rows):
    b, sb, rb, cb, p2p = band[rows], sbA[rows], rbA[rows], cbA[rows], p2pA[rows]
    out = np.where(b > 0, b, pred)
    free = (b == 0) & (pred > 0)
    out = np.where(free & np.isin(pred, [4, 5]) & np.isin(sb, [4, 5]), sb, out)
    out = np.where(free & ((np.isin(pred, [1, 2, 3]) & p2p & np.isin(rb, [1, 2, 3])) | (np.isin(pred, [7, 8, 11]) & ~p2p & np.isin(rb, [7, 8, 11]))), rb, out)
    out = np.where(free & (cb > 0), cb, out)
    return out
def load(spec, f):
    parts = [p.split(':') for p in spec.split('+')]
    ws = np.array([float(p[1]) if len(p) > 1 else 1.0 for p in parts]); ws = ws / ws.sum()
    return sum(w * np.load(ROOT / 'models' / p[0] / f).astype(float) for w, p in zip(ws, parts))
sample = pd.read_csv(ROOT / 'data/raw/sample_submission.csv')
ref = None
for arg in sys.argv[3:]:
    spec, mult, name = arg.split('|')
    oof, Pte = load(spec, 'oof.npy'), load(spec, 'test_proba.npy')
    w = fit_weights_fast(oof, y)
    w2 = w.copy(); w2[1:] *= float(mult)
    ptr = rules((oof * w2).argmax(1), tr)
    pte = rules((Pte * w2).argmax(1), ~tr)
    sub = pd.DataFrame({'id': base.txn_id.values[~tr], 'label': np.array(LABELS)[pte]})
    sub = sample[['id']].merge(sub, on='id', how='left', validate='one_to_one')
    assert sub.label.notna().all() and set(sub.label) == set(LABELS)
    if ref is None: ref = sub.label.values
    print(f'{name:<28} spec {spec:<22} mult {mult:<5} local {macro_f1(y, ptr):.5f} | train fraud calls {int((ptr>0).sum())} | test fraud calls {int((sub.label!="legit").sum())} | test rows differing from first {int((sub.label.values!=ref).sum())}')
    if name != '-':
        sub.to_csv(Path(sys.argv[2]) / f'{name}.csv', index=False)
