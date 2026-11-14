import sys, pandas as pd, numpy as np
root = sys.argv[1]
tr = pd.read_csv(root + '/data/raw/train.csv')
te = pd.read_csv(root + '/data/raw/test.csv')
te['label'] = 'TEST'
d = pd.concat([tr, te], ignore_index=True)
d['split'] = np.where(d.label.eq('TEST'), 'test', 'train')
d['snum'] = d.sender_id.str[2:].astype(int)
d['rnum'] = d.receiver_id.str[2:].astype(int)
d['dnum'] = d.device_id.str[2:].astype(int)
d['rpref'] = d.receiver_id.str[:2]
d['spref'] = d.sender_id.str[:2]
d['dpref'] = d.device_id.str[:2]
print(d.groupby(['split','spref']).snum.agg(['min','max','count','nunique']))
print(d.groupby(['split','rpref']).rnum.agg(['min','max','count','nunique']))
print(d.groupby(['split','dpref']).dnum.agg(['min','max','count','nunique']))
print(d.device_id.str.len().value_counts(), d.sender_id.str.len().value_counts(), d.receiver_id.str.len().value_counts())
d['fraud'] = ~d.label.isin(['legit','TEST'])
# device classes
d['dclass'] = np.select([d.dnum == d.snum, d.dnum < 420000, d.dnum <= 423000], ['home','other_home','botpool'], 'new')
print(pd.crosstab([d.split, d.dclass], d.label))
d.to_parquet(sys.argv[2] + '/raw_all.parquet')
