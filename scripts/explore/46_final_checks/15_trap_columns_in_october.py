import pandas as pd, numpy as np
from scipy import stats
w = pd.read_parquet('work.parquet')
print(w['dclass'].value_counts(dropna=False).to_string())
print(w.groupby('split')['band'].value_counts().to_string()[:1500])
te = w[w.split=='test'].copy()
tr = w[w.split=='train'].copy()
# test groups: fraud-band device rows, legit-band device rows, everything else
fb = (te.dnum>=473632)&(te.dnum<=474551)
lb = (te.dnum>=474552)&(te.dnum<=482013)
te['grp'] = np.where(fb,'fraud_band',np.where(lb,'legit_band','other'))
te['flag'] = te.pred>0
print('\nTEST rows by group x predicted-fraud')
print(pd.crosstab(te.grp, te.flag).to_string())
def desc(d, name):
    q = d.ip_risk_score.quantile([.1,.25,.5,.75,.9,.99]).round(3).tolist()
    av = d.app_version.value_counts(normalize=True).sort_index().round(3).to_dict()
    print(f'{name:38s} n={len(d):7d} ip mean={d.ip_risk_score.mean():.4f} q={q} >0.4={np.mean(d.ip_risk_score>0.4):.3f} appv={av}')
print('\nTEST')
desc(te[te.grp=='legit_band'], 'legit band (known legit)')
desc(te[(te.grp=='other')&(~te.flag)], 'home/other, predicted legit')
desc(te[(te.grp=='other')&(te.flag)], 'home/other, predicted fraud')
desc(te[(te.grp=='fraud_band')&(te.flag)], 'fraud band, predicted fraud')
# fraud-band singles = band rows that the no-band model would not flag; approximate with single-row devices
cnt = te[fb].groupby('dnum').txn_id.transform('size')
sing = te[fb][cnt==1]
multi = te[fb][cnt>1]
desc(sing, 'fraud band, single-row devices')
desc(multi, 'fraud band, multi-row devices')
a = te[te.grp=='legit_band'].ip_risk_score; 
for nm, d in [('fraud band singles', sing), ('fraud band multi', multi), ('other pred fraud', te[(te.grp=='other')&(te.flag)]), ('other pred legit', te[(te.grp=='other')&(~te.flag)])]:
    ks = stats.ks_2samp(a, d.ip_risk_score)
    print(f'KS ip_risk legit band vs {nm:22s} D={ks.statistic:.4f} p={ks.pvalue:.3g}')
    ct = pd.crosstab(np.r_[np.zeros(len(a)), np.ones(len(d))], np.r_[te[te.grp=="legit_band"].app_version.values, d.app_version.values])
    print('   app_version chi2 p=%.3g' % stats.chi2_contingency(ct)[1])
print('\nTRAIN for comparison')
trf = (tr.dnum>=423001)&(tr.dnum<=428454); trl=(tr.dnum>=428455)&(tr.dnum<=473631)
desc(tr[trl], 'legit band')
c2 = tr[trf].groupby('dnum').txn_id.transform('size')
desc(tr[trf][c2==1], 'fraud band, single-row devices')
desc(tr[trf][c2>1], 'fraud band, multi-row devices')
desc(tr[(~trf)&(~trl)&(tr.pred==0)&(tr.y_raw>0)], 'home, pred legit, fraud label (stealth)')
desc(tr[(~trf)&(~trl)&(tr.pred==0)&(tr.y_raw==0)], 'home, pred legit, legit label')
desc(tr[(tr.pred>0)&(tr.y_raw>0)], 'pred fraud, fraud label')
