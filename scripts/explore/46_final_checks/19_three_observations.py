# The user's three observations from the ten-row tables, tested on every ordinary-looking train row.
import pandas as pd, numpy as np
pd.set_option('display.width', 250)
w = pd.read_parquet('work.parquet')
tr = w[w.split=='train'].copy()
fb = (tr.dnum>=423001)&(tr.dnum<=428454)
o = tr[(tr.pred==0)&(~fb)].copy()                      # rows the model calls legit
o['f'] = (o.y_raw>0).astype(int)
o['hour'] = pd.to_datetime(o.timestamp).dt.hour
o['whole'] = (np.abs(o.amount - o.amount.round()) < 1e-9).astype(int)
print('rows', len(o), 'fraud share %.4f' % o.f.mean())
def tab(keys, name):
    t = o.groupby(keys, observed=True).f.agg(n='size', fraud='sum', share='mean')
    pm = o.groupby(keys + ['month'], observed=True).f.mean().unstack()
    t = pd.concat([t, pm.add_prefix('share_')], axis=1)
    t[['share'] + [c for c in t.columns if c.startswith('share_')]] *= 100
    print(f'\n{name}  (share = % of rows with a fraud label)'); print(t.round(2).to_string())
tab(['receiver_type','txn_type'], '1. receiver type x payment type')
print('\n   mix of payment types: legit rows vs fraud-labelled rows (% within receiver type)')
mix = pd.crosstab([o.receiver_type, o.txn_type], o.f, normalize='columns').mul(100).round(2); mix.columns = ['legit rows %', 'fraud rows %']; print(mix.to_string())
tab(['hour'], '2. hour of day')
print('\n   share of rows between 18:00 and 22:59: legit %.2f%%, fraud-labelled %.2f%%' % (100*o[o.f==0].hour.between(18,22).mean(), 100*o[o.f==1].hour.between(18,22).mean()))
tab(['receiver_type','txn_type','whole'], '3. whole-rupee amount x payment type')
