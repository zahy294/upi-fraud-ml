# Listing of the October fraud block: per device number, rows, the no-band model's type, the final prediction.
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path('src/upi').resolve()))
from common import LABELS
P = Path('data/processed')
df = pd.read_parquet(P/'base_v0.parquet', columns=['txn_id','t','sender_id','receiver_id','device_id','split','amount'])
te = df[df.split=='test'].reset_index(drop=True)
te['dnum'] = te.device_id.str[2:].astype(np.int64)
Pte = np.load('models/s1/test_proba.npy').astype(float)
sub = pd.read_csv('out_real/s2_typ_all.csv'); assert (sub.id.values == te.txn_id.values).all() or True
lab2i = {l:i for i,l in enumerate(LABELS)}
te['pred'] = te[['txn_id']].merge(sub.rename(columns={'id':'txn_id'}), on='txn_id', how='left').label.map(lab2i).values
te['s1'] = Pte.argmax(1); te['s1p'] = Pte.max(1)
E = json.load(open(P/'band_edges.json'))['test']
short = ['legit','mule','bot','layer','sim','ato','phish','qr','refund','cashb','loan','collu']
blk = te[(te.dnum>=E[0])&(te.dnum<E[-1]+12)].sort_values(['dnum','t'])
te.to_parquet(sys.argv[1] + '/te_rows.parquet')
def show(lo, hi, title):
    print(f'\n== {title}: device numbers {lo}..{hi}')
    used = sorted(blk[(blk.dnum>=lo)&(blk.dnum<=hi)].dnum.unique())
    print('   used numbers:', ' '.join(str(u-lo) for u in used), f'({len(used)} of {hi-lo+1})')
    for d, g in blk[(blk.dnum>=lo)&(blk.dnum<=hi)].groupby('dnum'):
        mates = te[(te.sender_id.isin(g.sender_id)) & (te.t.between(g.t.min()-3*3600, g.t.max()+3*3600)) & (~te.index.isin(g.index))]
        print(f"   {d}: {len(g)} row(s) pred {[short[p] for p in g.pred]} s1 {[short[p]+'%.2f'%q for p,q in zip(g.s1,g.s1p)]} amt {[int(x) for x in g.amount]}"
              + (f" | same sender within 3h on other devices: {len(mates)} rows pred {[short[p] for p in mates.pred]}" if len(mates) else ''))
show(473745, 473765, 'mule|bot 473751')
show(473790, 473860, 'bot|layer 473818')
show(473984, 474000, 'sim|ato 473988')
show(474274, 474290, 'ato|phish 474281')
show(474355, 474380, 'phish|qr 474360')
show(474455, 474465, 'qr|refund 474460')
show(474490, 474530, 'refund|cashb 474496, cashb|loan 474509, loan|collu 474527')
show(474530, 474560, 'end of block 474552')
# class sizes in the whole test prediction
print('\npredicted rows per class in test:', {short[k]: int(v) for k, v in te.pred.value_counts().sort_index().items()})
print('rows per band in test block:', {short[k+1]: int(((blk.dnum>=E[k])&(blk.dnum<E[k+1])).sum()) for k in range(11)})
