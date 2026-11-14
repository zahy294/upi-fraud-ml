import sys, pandas as pd, numpy as np
from scipy import stats
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 400)
d = pd.read_parquet('work.parquet')
acc = pd.read_csv(sys.argv[1] + '/data/raw/accounts.csv'); acc['num']=acc.account_id.str[2:].astype(int); acc=acc.set_index('num')
t = d[(d.split=='train')&(d.pred.isin([4,5]))&(d.dclass!='new')].copy()
t['ts'] = pd.to_datetime(t.timestamp)
t = t.sort_values(['snum','ts'])
new = (t.snum!=t.snum.shift())|(t.ts.diff().dt.total_seconds()>6*3600)
t['ev'] = new.cumsum()
t['bal_ratio'] = t.balance_before / t.groupby('ev').balance_before.shift()
t['gap'] = t.groupby('ev').ts.diff().dt.total_seconds()
t['away'] = (t.device_city != t.snum.map(acc.home_city)).astype(int)
t['hour'] = t.ts.dt.hour
t['amt_ratio'] = t.amount/t.balance_before
ev = t.groupby('ev').agg(n=('amount','size'), lab=('y_clean', lambda x: x.value_counts().index[0]), raw=('y_raw', lambda x: x[x>0].value_counts().index[0] if (x>0).any() else 0),
    r_med=('bal_ratio','median'), gap_med=('gap','median'), gap_min=('gap','min'), hour=('hour','first'), sim=('sim_age_hours','min'), sim_max=('sim_age_hours','max'), dage=('device_age_days','median'),
    away=('away','mean'), a0=('amount','first'), b0=('balance_before','first'), ar0=('amt_ratio','first'), ar_med=('amt_ratio','median'), pin=('pin_retries','mean'), login=('login_to_txn_sec','median'), np_=('new_payee','mean'),
    p2p=('receiver_type', lambda x: (x=='P2P').mean()), nrcv=('receiver_id','nunique'), month=('month','first'), acc_age=('sender_account_age_days','first'), dclass=('dclass','first'),
    amt_last=('amount','last'), whole=('amount', lambda x: (x%1==0).mean()))
ev['r'] = ev.r_med.round(2)
print('events', len(ev)); print(pd.crosstab(ev.r.fillna(-1).clip(0,1), ev.lab))
e = ev[(ev.r==0.25)]
print('x0.25 home-device events by clean label', e.lab.value_counts().to_dict())
e = e[e.lab.isin([4,5])]
cols = ['n','gap_med','gap_min','hour','sim','sim_max','dage','away','a0','b0','ar0','ar_med','pin','login','np_','p2p','nrcv','acc_age','amt_last','whole']
rows=[]
for c in cols:
    x, y = e[c][e.lab==4].dropna(), e[c][e.lab==5].dropna()
    ks = stats.ks_2samp(x,y)
    rows.append((c, round(x.mean(),3), round(y.mean(),3), round(x.median(),3), round(y.median(),3), round(ks.statistic,3), ks.pvalue))
print(pd.DataFrame(rows, columns=['col','mean_sim','mean_ato','med_sim','med_ato','ks','p']).sort_values('p').to_string())
print('fresh sim (<72h) share by label', e.groupby('lab').sim.apply(lambda x: (x<72).mean()).to_dict())
print(pd.crosstab(e.hour, e.lab).T.to_string())
print(pd.crosstab(e.n, e.lab).T.to_string())
print(pd.crosstab(e.dclass, e.lab).T.to_string())
print(pd.crosstab(pd.cut(e.sim,[0,1,2,4,8,16,24,48,72,1e9]), e.lab).T.to_string())
print(pd.crosstab(pd.cut(e.gap_med,[0,60,120,300,600,1200,1800,3600,1e9]), e.lab).T.to_string())
