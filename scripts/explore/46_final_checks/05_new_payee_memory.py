import sys, pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 400)
d = pd.read_parquet('work.parquet', columns=['split','sender_id','receiver_id','new_payee','timestamp','y_raw','pred','receiver_type','dclass','label'])
print('new_payee mean by split/receiver type'); print(d.groupby(['split','receiver_type']).new_payee.mean())
d['pair'] = d.sender_id + '>' + d.receiver_id
g = d.groupby('pair').agg(n=('new_payee','size'), ones=('new_payee','sum'), ntest=('split', lambda x: (x=='test').sum()))
print('pairs', len(g), 'pairs with >=2 rows', (g.n>=2).sum())
print(pd.crosstab(np.minimum(g.n,6), np.minimum(g.ones,4)))
# for pairs with exactly one 1 and n>=2: is the flagged row earliest in time?
m = g[(g.n>=2)&(g.ones==1)].index
s = d[d.pair.isin(m)].sort_values(['pair','timestamp'])
s['rank'] = s.groupby('pair').cumcount()
print('pairs n>=2 with exactly one new_payee=1: flagged row time-rank distribution'); print(s[s.new_payee==1]['rank'].value_counts().head(8))
print('share flagged first in time', (s[s.new_payee==1]['rank']==0).mean())
# all-zero pairs vs one-flag pairs
print('split of flagged row for pairs spanning both splits:')
sp = g[(g.n>=2)&(g.ones==1)&(g.ntest>0)&(g.ntest<g.n)].index
x = d[d.pair.isin(sp)&(d.new_payee==1)]
print(x.split.value_counts())
