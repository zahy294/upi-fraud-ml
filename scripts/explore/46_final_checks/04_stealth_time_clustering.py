import sys, pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 400)
d = pd.read_parquet('work.parquet')
short = ['legit','mule','bot','layer','sim','ato','phish','qr','refund','cashb','loan','collu']
t = d[d.split=='train'].copy()
t['ts'] = pd.to_datetime(t.timestamp).astype('int64')//10**9
st = t[(t.y_raw>0)&(t.pred==0)]
rng = np.random.default_rng(0)
lg = t[(t.y_raw==0)&(t.pred==0)]
for k in range(1,12):
    s = np.sort(st.ts[st.y_raw==k].values)
    g = np.diff(s)
    # control: same number of legit rows sampled at random (same time-of-day profile)
    c = np.sort(lg.ts.sample(len(s), random_state=k).values); gc = np.diff(c)
    q = [60, 300, 900, 1800]
    print(short[k], 'n', len(s), 'median gap', np.median(g), 'ctrl', np.median(gc), ' share gaps <', q, [round((g<x).mean(),4) for x in q], 'ctrl', [round((gc<x).mean(),4) for x in q])
