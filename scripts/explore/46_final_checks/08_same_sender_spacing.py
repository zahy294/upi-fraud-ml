import sys, pandas as pd, numpy as np
from scipy import stats
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 400)
d = pd.read_parquet('work.parquet', columns=['timestamp','split','y_raw','pred','snum','receiver_id','dnum','month','dclass'])
d['ts'] = pd.to_datetime(d.timestamp).astype('int64')//10**9
d['grp'] = np.where(d.split=='test','test',np.where(d.y_raw==0,'legit',np.where(d.pred==0,'stealth','detected')))
# legit block: generation order vs sender number / time
lb = d[(d.dnum>=428455)&(d.dnum<473632)].sort_values('dnum')
print('legit block spearman(dnum, snum)', stats.spearmanr(lb.dnum, lb.snum)[0], ' (dnum, ts)', stats.spearmanr(lb.dnum, lb.ts)[0])
# adjacent device numbers: same day? same sender neighbourhood?
print('adjacent legit-block rows: |ts diff| median days', np.median(np.abs(np.diff(lb.ts.values)))/86400, ' |snum diff| median', np.median(np.abs(np.diff(lb.snum.values))))
print('first 12 legit block rows'); print(lb.head(12)[['dnum','snum','timestamp','receiver_id']].to_string())
def gaps(key):
    s = d.sort_values([key,'ts'])
    k = s[key].values; t = s.ts.values
    prev = np.where(k[1:]==k[:-1], t[1:]-t[:-1], np.inf); prev = np.r_[np.inf, prev]
    nxt = np.r_[prev[1:], np.inf]
    s['gap'] = np.minimum(prev, nxt)
    return s
for key in ['snum','receiver_id','dnum']:
    s = gaps(key)
    t = s[s.split=='train']
    print('== nearest same-%s row gap' % key)
    qs = [60, 600, 3600, 6*3600, 86400, 3*86400, 7*86400, 30*86400]
    out = {}
    for g in ['legit','stealth']:
        x = t.gap[t.grp==g].values
        out[g] = [round((x<q).mean(),4) for q in qs] + [round(np.isinf(x).mean(),4)]
    print(pd.DataFrame(out, index=[f'<{q}s' for q in qs]+['none']).T.to_string())
    a = t.gap[t.grp=='legit'].replace(np.inf, 1e9).sample(200000, random_state=0); b = t.gap[t.grp=='stealth'].replace(np.inf,1e9)
    print('KS', stats.ks_2samp(a,b))
# rows per sender per day
d['day'] = d.ts//86400
c = d.groupby(['snum','day']).size().rename('sd')
d = d.join(c, on=['snum','day'])
print('rows of same sender on same day (train):'); print(pd.crosstab(d.grp[d.split=='train'], np.minimum(d.sd[d.split=='train'],5), normalize='index').round(4))
# rows per sender total
c = d[d.split=='train'].groupby('snum').size().rename('sn')
t = d[d.split=='train'].join(c, on='snum')
print('rows per sender (train) by group'); print(t.groupby('grp').sn.describe().round(2))
