import sys, pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 400)
d = pd.read_parquet('work.parquet')
t = d[(d.split=='train')].copy()
t['ts'] = pd.to_datetime(t.timestamp)
t['stealth'] = ((t.y_raw>0)&(t.pred==0)).astype(int)
t = t[(t.pred==0)]   # ordinary-looking rows only
t['hour']=t.ts.dt.hour; t['dow']=t.ts.dt.dayofweek; t['dom']=t.ts.dt.day; t['doy']=t.ts.dt.dayofyear
for c in ['hour','dow','dom']:
    g = t.groupby(c).stealth.agg(['mean','size'])
    print(c, 'stealth share min/max', g['mean'].min().round(4), g['mean'].max().round(4), ' chi2/df', round((((g['mean']-t.stealth.mean())**2*g['size']/(t.stealth.mean()*(1-t.stealth.mean()))).sum())/(len(g)-1),2))
g = t.groupby('doy').stealth.agg(['mean','size'])
print('by day-of-year chi2/df', round((((g['mean']-t.stealth.mean())**2*g['size']/(t.stealth.mean()*(1-t.stealth.mean()))).sum())/(len(g)-1),2), 'min/max', g['mean'].min().round(4), g['mean'].max().round(4))
for c in ['txn_type','receiver_type','app','merchant_category','device_city','new_payee','pin_retries','dclass','app_version']:
    g = t.groupby(t[c].fillna('NA')).stealth.agg(['mean','size']); g = g[g['size']>=2000]
    print(c, 'share min/max', g['mean'].min().round(4), g['mean'].max().round(4), 'chi2/df', round((((g['mean']-t.stealth.mean())**2*g['size']/(t.stealth.mean()*(1-t.stealth.mean()))).sum())/max(len(g)-1,1),2))
acc = pd.read_csv(sys.argv[1] + '/data/raw/accounts.csv'); acc['num']=acc.account_id.str[2:].astype(int); acc=acc.set_index('num')
for c in ['home_city','bank_code','kyc_level','os']:
    v = t.snum.map(acc[c])
    g = t.groupby(v).stealth.agg(['mean','size']); g = g[g['size']>=2000]
    print('sender', c, 'share min/max', g['mean'].min().round(4), g['mean'].max().round(4), 'chi2/df', round((((g['mean']-t.stealth.mean())**2*g['size']/(t.stealth.mean()*(1-t.stealth.mean()))).sum())/max(len(g)-1,1),2))
t['oy'] = t.snum.map(acc.opened_on).str[:4]
g = t.groupby('oy').stealth.agg(['mean','size']); print(g.round(4).T.to_string())
# numeric deciles
for c in ['amount','balance_before','sim_age_hours','device_age_days','sender_account_age_days','login_to_txn_sec','ip_risk_score','snum','rnum']:
    q = pd.qcut(t[c], 20, duplicates='drop')
    g = t.groupby(q, observed=True).stealth.agg(['mean','size'])
    print(c, 'ventile share min/max', g['mean'].min().round(4), g['mean'].max().round(4), 'chi2/df', round((((g['mean']-t.stealth.mean())**2*g['size']/(t.stealth.mean()*(1-t.stealth.mean()))).sum())/(len(g)-1),2))
print('overall stealth share among ordinary rows', t.stealth.mean())
