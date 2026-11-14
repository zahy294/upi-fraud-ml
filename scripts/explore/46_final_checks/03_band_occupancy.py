import sys, pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 300)
d = pd.read_parquet('work.parquet')
short = ['legit','mule','bot','layer','sim','ato','phish','qr','refund','cashb','loan','collu']
edges = {'train':[423001,423656,424153,424625,425033,426763,427163,427844,428065,428200,428302,428455],
         'test':[473632,473751,473818,473920,473988,474281,474360,474460,474496,474509,474527,474552]}
for sp in ['train','test']:
    e = edges[sp]
    g = d[(d.dnum>=e[0])&(d.dnum<e[-1])]
    print(sp, 'rows on band devices', len(g), 'any other split rows on these numbers:', ((d.dnum>=e[0])&(d.dnum<e[-1])&(d.split!=sp)).sum())
    out = []
    for k in range(11):
        b = g[(g.dnum>=e[k])&(g.dnum<e[k+1])]
        nums = e[k+1]-e[k]
        occ = b.dnum.nunique()
        rpd = b.groupby('dnum').size()
        row = {'band':short[k+1], 'numbers':nums, 'occupied':occ, 'occ_share':round(occ/nums,3), 'rows':len(b), 'rows/dev max':rpd.max(), 'dev_1row':(rpd==1).sum(), 'dev_multi':(rpd>1).sum(),
               'nsenders':b.snum.nunique()}
        if sp=='train':
            row['lab_legit'] = (b.y_raw==0).sum(); row['lab_band'] = (b.y_raw==k+1).sum(); row['lab_other'] = ((b.y_raw>0)&(b.y_raw!=k+1)).sum()
        out.append(row)
    print(pd.DataFrame(out).to_string())
# legit block occupancy
for sp, lo, hi in [('train',428455,473632),('test',474552,482014)]:
    g = d[(d.dnum>=lo)&(d.dnum<hi)]
    rpd = g.groupby('dnum').size()
    print(sp,'legit block numbers',hi-lo,'occupied',g.dnum.nunique(),'rows',len(g),'rows/dev dist',rpd.value_counts().head(6).to_dict(), 'splits', g.split.value_counts().to_dict())
    if sp=='train': print(' labels', g.label.value_counts().to_dict())
    # is number monotone in time?
    f = g.groupby('dnum').timestamp.min().sort_index()
    t = pd.to_datetime(f).astype('int64').values
    print(' share of consecutive devices with increasing first timestamp', (np.diff(t)>0).mean(), 'spearman', pd.Series(t).corr(pd.Series(np.arange(len(t))), method='spearman'))
# 420000..423000 region besides pool
g = d[(d.dnum>=420000)&(d.dnum<=423000)]
print('420000-423000: rows', len(g), 'devs', g.dnum.nunique(), g.split.value_counts().to_dict())
