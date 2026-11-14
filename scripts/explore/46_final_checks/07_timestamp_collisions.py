import sys, pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 400)
d = pd.read_parquet('work.parquet', columns=['timestamp','split','y_raw','pred','sender_id','receiver_id','dnum','month'])
d['ts'] = pd.to_datetime(d.timestamp).astype('int64')//10**9
tr = d[d.split=='train']
vc = tr.groupby('ts').size()
n = len(tr); span = tr.ts.max()-tr.ts.min()
print('train rows', n, 'distinct seconds', len(vc), 'rows per second dist', vc.value_counts().sort_index().to_dict())
# Poisson expectation within core range using hour-of-day inhomogeneity: compare simple overall
core = tr[tr.month<='2026-09']
secs = 92*86400
lam = len(core)/secs
print('naive lambda', lam)
# collision share by label group
tr = tr.assign(cnt=tr.ts.map(vc))
tr['grp'] = np.where(tr.y_raw==0,'legit',np.where(tr.pred==0,'stealth','detected'))
print(tr.groupby('grp').cnt.agg(n='size', share_collide=lambda x: (x>1).mean(), mean='mean'))
# legit-only collisions: among legit rows, share whose second also holds another LEGIT row
lg = tr[tr.y_raw==0]
vcl = lg.groupby('ts').size()
print('legit rows sharing second with another legit row:', (lg.ts.map(vcl)>1).mean())
# per-hour profile expectation: P(collision) = 1-exp(-lambda_hour)
tr['hod'] = (tr.ts % 86400)//3600
print(tr.groupby('hod').agg(n=('cnt','size'), collide=('cnt', lambda x: (x>1).mean())).assign(expected=lambda t: 1-np.exp(-t.n/92/3600)).round(4).T.to_string())
# seconds-of-minute distribution for legit vs fraud
tr['sec'] = tr.ts % 60
print('sec-of-minute chi2/df legit', (lambda c: ((c-c.mean())**2/c.mean()).sum()/59)(np.bincount(tr.sec[tr.y_raw==0],minlength=60)), 'stealth', (lambda c: ((c-c.mean())**2/c.mean()).sum()/59)(np.bincount(tr.sec[tr.grp=='stealth'],minlength=60)))
