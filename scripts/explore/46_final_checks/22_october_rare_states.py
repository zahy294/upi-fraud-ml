import pandas as pd, numpy as np
pd.set_option('display.width', 260); pd.set_option('display.max_rows', 300)
w = pd.read_parquet('work.parquet')
w['month'] = w.timestamp.str[:7]
shop = ['grocery','food','pharmacy','fuel']
mq = (w.receiver_type=='MERCHANT') & (w.txn_type=='QR')
o = w[mq & (w.pred==0)].copy()
o['shop'] = o.merchant_category.isin(shop)
o['whole'] = (o.amount == o.amount.round())
print('merchant-QR rows the model calls legit, per month:')
t = o.groupby('month').agg(rows=('txn_id','size'), flag1=('qr_vpa_mismatch', lambda s: int((s==1).sum())), noflag=('qr_vpa_mismatch', lambda s: int(s.isna().sum())),
                           whole=('whole','sum'), shop=('shop','mean'))
t['flag1_per_10k'] = (1e4*t.flag1/t.rows).round(2); t['whole_per_10k'] = (1e4*t.whole/t.rows).round(1); print(t.to_string())
f1 = o[o.qr_vpa_mismatch==1]
print('\nflag = 1 rows: shop share per month', f1.groupby('month').shop.mean().round(3).to_dict())
print('flag = 1 rows: whole-rupee share per month', f1.groupby('month').whole.mean().round(3).to_dict())
print('flag = 1 rows in train by label:', f1[f1.split=='train'].label.value_counts().to_dict())
# every rare state among predicted-legit rows: October rate vs train rate
p = w[w.pred==0].copy()
p['whole'] = (p.amount == p.amount.round()); p['over'] = p.amount > p.balance_before
p['bal_1p1'] = np.isclose(p.balance_before, p.amount*1.1, rtol=1e-6)
p['login_lt5'] = p.login_to_txn_sec < 5; p['pin2'] = p.pin_retries >= 2; p['sim_lt72'] = p.sim_age_hours < 72; p['dev_lt2'] = p.device_age_days < 2
p['amt_gt20k'] = p.amount > 20000; p['amt_lt10'] = p.amount < 10
print('\nrare states among predicted-legit rows, per 10,000 rows, by receiver type and month:')
rows = []
for c in ['whole','over','bal_1p1','login_lt5','pin2','sim_lt72','dev_lt2','amt_gt20k','amt_lt10','new_payee']:
    for rt in ('MERCHANT','P2P'):
        g = p[p.receiver_type==rt].groupby('month')[c].mean()*1e4
        rows.append([c, rt] + g.round(1).tolist())
print(pd.DataFrame(rows, columns=['state','receiver']+sorted(p.month.unique())).to_string(index=False))
print('\ntxn type mix per month (% of predicted-legit rows):')
print((pd.crosstab([p.receiver_type, p.txn_type], p.month, normalize='columns')*100).round(3).to_string())
