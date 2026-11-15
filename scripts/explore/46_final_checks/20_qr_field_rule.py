import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
w = pd.read_parquet('work.parquet')
tr = w[w.split=='train']; te = w[w.split=='test']
L = ['legit','mule','bot','layer','sim','ato','phish','qr','refund','cashb','loan','collu']
# the legit sampler's rule: a QR flag exists exactly when a merchant is paid by QR
for nm, d in (('train', tr), ('test', te)):
    normal = (d.receiver_type=='MERCHANT') & (d.txn_type=='QR')
    has = d.qr_vpa_mismatch.notna()
    print(f'\n{nm}: QR flag present x merchant-QR row'); print(pd.crosstab(has.rename('flag present'), normal.rename('merchant paid by QR')).to_string())
# known-legit rows: ordinary new devices (the fully occupied legit block)
for nm, d, lo, hi in (('train', tr, 428455, 473631), ('test', te, 474552, 482013)):
    b = d[(d.dnum>=lo)&(d.dnum<=hi)]
    odd = b[b.qr_vpa_mismatch.notna() & ~((b.receiver_type=='MERCHANT')&(b.txn_type=='QR'))]
    print(f'{nm} legit-block rows: {len(b)} | with a QR flag outside merchant-QR: {len(odd)} | P2P paid by QR: {int(((b.receiver_type=="P2P")&(b.txn_type=="QR")).sum())}')
odd = tr.qr_vpa_mismatch.notna() & ~((tr.receiver_type=='MERCHANT')&(tr.txn_type=='QR'))
print('\nTRAIN rows with a QR flag outside merchant-QR:', int(odd.sum()))
t = pd.crosstab([tr[odd].receiver_type, tr[odd].txn_type, tr[odd].label], tr[odd].pred.map(lambda p: 'model fraud' if p>0 else 'model legit'))
print(t.to_string())
r = tr[odd & (tr.pred==0)]
print('\nmodel-legit rows in that set by type and label:'); print(pd.crosstab([r.receiver_type, r.txn_type], r.label).to_string())
oddt = te.qr_vpa_mismatch.notna() & ~((te.receiver_type=='MERCHANT')&(te.txn_type=='QR'))
print('\nTEST rows with a QR flag outside merchant-QR:', int(oddt.sum()))
print(pd.crosstab([te[oddt].receiver_type, te[oddt].txn_type], te[oddt].pred.map(lambda p: L[p])).to_string())
x = te[oddt & (te.pred==0)]
print('\nTEST rows in that set currently predicted legit:', len(x))
print(x[['txn_id','timestamp','sender_id','receiver_id','receiver_type','txn_type','amount','new_payee','qr_vpa_mismatch','dclass','dnum']].to_string(index=False))
