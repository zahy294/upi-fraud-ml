import sys, pandas as pd, numpy as np
from pathlib import Path
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 400)
ROOT = Path(sys.argv[1])
base = pd.read_parquet(ROOT / 'data/processed/base_v0.parquet', columns=['txn_id','split','device_id','device_age_days'])
tr = (base.split=='train').values
dn = base.device_id.str[2:].astype(int).values
files = sorted((ROOT/'models/s1').glob('*.npy')); print([f.name for f in files])
oof = np.load(ROOT/'models/s1/oof.npy'); te = np.load(ROOT/'models/s1/test_proba.npy')
pf = np.empty(len(base)); pf[tr] = 1-oof[:,0]; pf[~tr] = 1-te[:,0]
short = ['legit','mule','bot','layer','sim','ato','phish','qr','refund','cashb','loan','collu']
edges = {'train':[423001,423656,424153,424625,425033,426763,427163,427844,428065,428200,428302,428455],
         'test':[473632,473751,473818,473920,473988,474281,474360,474460,474496,474509,474527,474552]}
rows=[]
for sp in ['train','test']:
    e = edges[sp]
    for k in range(11):
        m = (dn>=e[k])&(dn<e[k+1])
        cnt = pd.Series(dn[m]).map(pd.Series(dn[m]).value_counts()).values
        single = cnt==1
        p = pf[m]
        rows.append({'split':sp,'band':short[k+1],'rows':int(m.sum()),'singles':int(single.sum()),'singles_flagged(p>0.5)':int((p[single]>0.5).sum()),'share':round(float((p[single]>0.5).mean()),3) if single.sum() else np.nan,
                     'multi_rows':int((~single).sum()),'multi_flagged':int((p[~single]>0.5).sum())})
r = pd.DataFrame(rows); print(r.to_string())
for sp in ['train','test']:
    x = r[(r.split==sp)&(~r.band.isin(['sim','ato']))]
    print(sp, 'non-drain bands: singles', x.singles.sum(), 'flagged', x['singles_flagged(p>0.5)'].sum(), 'share', round(x['singles_flagged(p>0.5)'].sum()/x.singles.sum(),3))
    x = r[(r.split==sp)]
    print(sp, 'all bands: singles', x.singles.sum(), 'flagged', x['singles_flagged(p>0.5)'].sum(), 'share', round(x['singles_flagged(p>0.5)'].sum()/x.singles.sum(),3), '| multi rows', x.multi_rows.sum(), 'flagged', x.multi_flagged.sum())
