# Is the missing QR flag a tell for October fraud-phase rows? Use device bands as known labels in both splits.
import pandas as pd, numpy as np
pd.set_option('display.width', 260); pd.set_option('display.max_rows', 300)
w = pd.read_parquet('work.parquet')
L = ['legit','mule','bot','layer','sim','ato','phish','qr','refund','cashb','loan','collu']
edges = {'train': [423001, 423656, 424153, 424625, 425033, 426763, 427163, 427844, 428065, 428200, 428302, 428455],
         'test':  [473632, 473751, 473822, 473920, 473988, 474281, 474360, 474460, 474496, 474509, 474527, 474552]}
legit_blk = {'train': (428455, 473631), 'test': (474552, 482013)}
w['mq'] = (w.receiver_type=='MERCHANT') & (w.txn_type=='QR')
w['noflag'] = w.qr_vpa_mismatch.isna()
w['fixed'] = w.amount.isin([1000, 2000, 2500, 5000, 10000])
for sp in ('train', 'test'):
    d = w[w.split==sp].copy(); e = edges[sp]
    d['bandk'] = np.searchsorted(np.array(e), d.dnum.values, 'right')
    inb = (d.dnum>=e[0])&(d.dnum<e[-1])
    d['grp'] = np.where(inb, d.bandk.map(lambda k: L[k] if 1<=k<=11 else '?'), np.where((d.dnum>=legit_blk[sp][0])&(d.dnum<=legit_blk[sp][1]), 'LEGIT BLOCK', 'home/other'))
    q = d[d.mq]
    t = q.groupby('grp').agg(merchant_qr_rows=('txn_id','size'), no_flag=('noflag','sum'), no_flag_fixed_amt=('fixed', lambda s: int((s & q.loc[s.index,'noflag']).sum())))
    t['no_flag_ordinary_amt'] = t.no_flag - t.no_flag_fixed_amt
    print(f'\n{sp}: merchant paid by QR, by device group (band = known fraud type)'); print(t.to_string())
    x = q[inb.reindex(q.index) & q.noflag & ~q.fixed]
    if len(x): print('   band rows with no flag and an ordinary amount:'); print(x[['timestamp','sender_id','receiver_id','amount','dnum','grp','pred']].to_string(index=False))
te = w[w.split=='test']; tr = w[w.split=='train']
# the 20 + 1 rows: any other shared trait? compare with ordinary merchant-QR test rows
x = te[te.mq & te.noflag & ~te.fixed]
o = te[te.mq & ~te.noflag & (te.pred==0)]
print('\nOctober merchant-QR rows with no flag and an ordinary amount:', len(x), '| ordinary merchant-QR rows with a flag:', len(o))
for c in ['new_payee','pin_retries','app','merchant_category','device_city','dclass']:
    a = x[c].value_counts(normalize=True).round(2).head(6).to_dict(); b = o[c].value_counts(normalize=True).round(2).head(6).to_dict()
    print(f'  {c}: no-flag {a} | with flag {b}')
for c in ['amount','balance_before','sim_age_hours','device_age_days','sender_account_age_days','login_to_txn_sec','ip_risk_score','app_version']:
    print(f'  {c}: no-flag median {x[c].median():.2f} (q10 {x[c].quantile(.1):.2f}, q90 {x[c].quantile(.9):.2f}) | with flag median {o[c].median():.2f} (q10 {o[c].quantile(.1):.2f}, q90 {o[c].quantile(.9):.2f})')
print('  days of month:', sorted(pd.to_datetime(x.timestamp).dt.day.tolist()))
print('  hours:', sorted(pd.to_datetime(x.timestamp).dt.hour.tolist()))
print('\ntrain: ordinary-amount merchant-QR rows with no flag:', int((tr.mq & tr.noflag & ~tr.fixed).sum()), 'of', int(tr.mq.sum()), '| test:', len(x), 'of', int(te.mq.sum()))
# other columns: nulls in test vs train among ordinary rows
print('\nnull share per column, rows the model calls legit: train vs test')
for c in ['merchant_category','device_city','sim_age_hours','device_age_days','new_payee','pin_retries','login_to_txn_sec','balance_before','app','app_version','ip_risk_score','amount','receiver_type','txn_type']:
    a = tr[tr.pred==0][c].isna(); b = te[te.pred==0][c].isna()
    print(f'  {c:26s} train {a.sum():8d} ({100*a.mean():.3f}%)   test {b.sum():7d} ({100*b.mean():.3f}%)')
