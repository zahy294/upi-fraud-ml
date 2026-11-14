import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
d = pd.read_parquet('work.parquet', columns=['split','pred','y_raw','y_clean','app_version','ip_risk_score','dnum','dclass','pfraud','txn_type','receiver_type','month','amount'])
t = d[d.split=='train'].copy()
t['v3'] = (t.app_version<4).astype(int); t['ip'] = (t.ip_risk_score>0.5).astype(int); t['both'] = t.v3*t.ip
print('app_version values', sorted(t.app_version.unique())[:12])
grp = {
 'legit block (pure sampler, legit phase)': (t.dnum>=428455),
 'legit-labelled, pred legit, home/other dev': (t.y_raw==0)&(t.pred==0)&(t.dclass.isin(['home','other_home'])),
 'stealth (fraud label, pred legit)': (t.y_raw>0)&(t.pred==0),
 'detected fraud (pred fraud, fraud label)': (t.y_raw>0)&(t.pred>0),
 'legit label but pred fraud': (t.y_raw==0)&(t.pred>0),
}
rows=[]
for k,m in grp.items():
    g = t[m]; n=len(g)
    rows.append((k, n, g.v3.mean(), g.ip.mean(), g.both.mean(), int(g.both.sum())))
r = pd.DataFrame(rows, columns=['group','n','app3','ip>0.5','both','both_n']); print(r.round(4).to_string())
p = r.both[0]; n = r.n[1]
print('expected both-count in legit-labelled pred-legit rows from sampler rate:', round(p*n), '+-', round(np.sqrt(r.both[0]*(1-r.both[0])/r.n[0])*n), ' observed', r.both_n[1])
# by pfraud bin among legit-labelled pred-legit
g = t[(t.y_raw==0)&(t.pred==0)]
b = pd.cut(g.pfraud, [-1,1e-4,1e-3,1e-2,0.05,0.2,0.5,1.01])
print(g.groupby(b, observed=True).agg(n=('v3','size'), app3=('v3','mean'), ip=('ip','mean'), both=('both','mean')).round(4).to_string())
# ip_risk distribution tails
for k,m in list(grp.items())[:3]:
    g = t[m]; print(k, 'ip quantiles', g.ip_risk_score.quantile([.5,.9,.99,.999]).round(3).tolist(), ' app3|ip>0.5:', round(g.v3[g.ip==1].mean(),3), ' app3|ip<=0.5:', round(g.v3[g.ip==0].mean(),3))
# by txn_type / receiver_type among legit-labelled pred-legit: both rate
print(g.groupby(['txn_type','receiver_type']).agg(n=('both','size'), both=('both','mean'), app3=('v3','mean'), ip=('ip','mean')).round(4).to_string())
