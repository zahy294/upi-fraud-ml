import sys, pandas as pd, numpy as np
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root / 'colab')); sys.path.insert(0, str(root / 'src/upi'))
from bench import Bench, fit_weights_fast
from common import LABELS
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30); pd.set_option('display.max_rows', 200)
B = Bench(root)
P = B.load('s2_a')
w = fit_weights_fast(P, B.y)
pred = B.rules((P * w).argmax(1))
base = pd.read_parquet(root / 'data/processed/base_v0.parquet', columns=['txn_id', 'split'])
tr = (base.split == 'train').values
t = pd.DataFrame({'txn_id': base.txn_id.values[tr], 'pred': pred, 'y_clean': B.y, 'pfraud': 1 - P[:, 0], 'band': B.band})
sub = pd.read_csv(root / 'submissions/final_s2_v7edges.csv')
li = {l: i for i, l in enumerate(LABELS)}
t2 = pd.DataFrame({'txn_id': sub.id.values, 'pred': sub.label.map(li).values, 'y_clean': -1, 'pfraud': np.nan, 'band': -1})
t = pd.concat([t, t2], ignore_index=True)
d = pd.read_parquet('raw_all.parquet')
d = d.merge(t, on='txn_id', how='left')
assert d.pred.notna().all()
d['pred'] = d.pred.astype(int)
d['y_raw'] = d.label.map(li).fillna(-1).astype(int)
d['month'] = d.timestamp.str[:7]
d.loc[d.month > '2026-10', 'month'] = '2026-10'
d.loc[(d.split == 'train') & (d.month > '2026-09'), 'month'] = '2026-09'
d.to_parquet('work.parquet')
short = ['legit','mule','bot','layer','sim','ato','phish','qr','refund','cashb','loan','collu']
c = pd.crosstab(d.month, d.pred); c.columns = short
print('predicted counts by month'); print(c)
print('ratio Oct / mean(Jul-Sep)'); print((c.loc['2026-10'] / c.loc[['2026-07','2026-08','2026-09']].mean()).round(2))
c2 = pd.crosstab(d.month[d.split=='train'], d.y_clean[d.split=='train']); c2.columns = short
print('clean label counts'); print(c2)
