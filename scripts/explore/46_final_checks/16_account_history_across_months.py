# Does an entity's fraud-label history in OTHER months predict ordinary-looking fraud in this month?
# (the idea behind a temporally restricted target encoding of account ids)
import pandas as pd, numpy as np
w = pd.read_parquet('work.parquet')
tr = w[w.split=='train'].copy()
tr['pair'] = tr.sender_id + '>' + tr.receiver_id
tr['isf'] = (tr.y_raw>0).astype(int)
tr['st'] = ((tr.y_raw>0)&(tr.pred==0)).astype(int)     # ordinary-looking fraud row
tr['pt'] = ((tr.y_raw>0)&(tr.pred>0)).astype(int)      # detected (patterned) fraud row
fb = (tr.dnum>=423001)&(tr.dnum<=428454)
ordm = (tr.pred==0)&(~fb)
print('ordinary rows', ordm.sum(), 'fraud share %.4f' % tr.loc[ordm,'isf'].mean())
def other_month(key, col):
    g = tr.groupby([key,'month'])[col].sum().rename('c').reset_index()
    tot = g.groupby(key).c.sum().rename('t')
    m = tr[[key,'month']].merge(g, on=[key,'month'], how='left').merge(tot, left_on=key, right_index=True, how='left')
    return (m.t - m.c).values
def show(name, v):
    d = pd.DataFrame({'v':v[ordm.values], 'f':tr.loc[ordm,'isf'].values, 'm':tr.loc[ordm,'month'].values})
    d['b'] = pd.cut(d.v, [-1,0,1,2,4,9,1e9], labels=['0','1','2','3-4','5-9','10+'])
    t = d.groupby('b', observed=True).f.agg(['size','mean'])
    pm = d.groupby(['b','m'], observed=True).f.mean().unstack().round(4)
    print(f'\n{name}'); print(pd.concat([t.rename(columns={"size":"n","mean":"share"}).round(4), pm], axis=1).to_string())
for key in ['sender_id','receiver_id','device_id','pair']:
    for col, lab in [('st','ordinary-looking fraud rows'),('pt','detected fraud rows')]:
        show(f'{key}: {lab} of the same {key} in other months', other_month(key, col))
# rate version: sender's other-month stealth share (stealth rows / rows), for senders with >=20 other-month rows
tr['one']=1
n_o = other_month('sender_id','one'); s_o = other_month('sender_id','st')
rate = np.where(n_o>=20, s_o/np.maximum(n_o,1), np.nan)
d = pd.DataFrame({'r':rate[ordm.values], 'f':tr.loc[ordm,'isf'].values}).dropna()
d['b'] = pd.qcut(d.r, 10, duplicates='drop')
print('\nsender other-month ordinary-fraud RATE deciles (senders with >=20 rows in other months)')
print(d.groupby('b', observed=True).f.agg(['size','mean']).round(4).to_string())
n_o = other_month('receiver_id','one'); s_o = other_month('receiver_id','st')
rate = np.where(n_o>=20, s_o/np.maximum(n_o,1), np.nan)
d = pd.DataFrame({'r':rate[ordm.values], 'f':tr.loc[ordm,'isf'].values}).dropna()
d['b'] = pd.qcut(d.r, 10, duplicates='drop')
print('\nreceiver other-month ordinary-fraud RATE deciles (receivers with >=20 rows in other months)')
print(d.groupby('b', observed=True).f.agg(['size','mean']).round(4).to_string())
