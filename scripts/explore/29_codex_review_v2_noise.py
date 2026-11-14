"""Read-only missing-label sensitivity; saved submission/OOF only, no fitting."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','ARROW_NUM_THREADS'):
    os.environ[k]='4'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
pa.set_cpu_count(4);pa.set_io_thread_count(4)
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/upi'))
from common import LABELS,macro_f1
sys.stdout.reconfigure(encoding='utf-8',line_buffering=True)
df=pd.read_parquet(ROOT/'data/processed/base_v0.parquet',columns=['txn_id','label','split','month'])
tr=(df.split=='train').values; te=~tr
y=df.label[tr].map(dict(zip(LABELS,range(12)))).to_numpy(int)
yc=np.load(ROOT/'models/oof_raw/y_clean.npy')
F3=pd.read_parquet(ROOT/'data/processed/features_v3.parquet');F5=pd.read_parquet(ROOT/'data/processed/features_v5.parquet',columns=['ch_band'])
p2p=pd.read_parquet(ROOT/'data/processed/base_v0.parquet',columns=['receiver_type']).receiver_type.values=='P2P'
b=F3.band.to_numpy(int)
def rules(p,rows):
    S=F3[[f'snd_band_{k}_3h' for k in range(1,12)]].values[rows]
    R=F3[[f'rcv_band_{k}_24h' for k in range(1,12)]].values[rows]
    sb=np.where((S>0).sum(1)==1,S.argmax(1)+1,0);rb=np.where((R>0).sum(1)==1,R.argmax(1)+1,0)
    cb=F5.ch_band.values[rows].astype(int);free=(b[rows]==0)&(p>0)
    out=np.where(b[rows]>0,b[rows],p)
    out=np.where(free&np.isin(p,[4,5])&np.isin(sb,[4,5]),sb,out)
    out=np.where(free&np.isin(p,[1,2,3])&p2p[rows]&np.isin(rb,[1,2,3]),rb,out)
    out=np.where(free&np.isin(p,[7,8,11])&~p2p[rows]&np.isin(rb,[7,8,11]),rb,out)
    return np.where(free&(cb>0),cb,out)
P=np.load(ROOT/'models/oof_c3/oof.npy').astype(float);w=np.load(ROOT/'models/oof_c3/weights_s5_c3.npy')
pred=rules((P*w).argmax(1),tr)
cm=np.bincount(yc*12+pred,minlength=144).reshape(12,12).astype(float)
def score(cm):return float(np.mean(2*np.diag(cm)/(cm.sum(0)+cm.sum(1))))
bt=b[tr];known=bt>0
q=float((y[known]==0).mean())
print('band unreported q',q,'n',known.sum(),'raw_legit',((y==0)&known).sum())
for typ in range(1,12):
    mask=bt==typ
    print('band class',LABELS[typ],'n',mask.sum(),'raw_legit',((y==0)&mask).sum(),'q',float((y[mask]==0).mean()))
for m in ('2026-07','2026-08','2026-09'):
    mask=known&(df.month.values[tr]==m)
    print('band month',m,'n',mask.sum(),'q',float((y[mask]==0).mean()))
for missing in [0,.03,.05,.08,.10,.12,q]:
    # Assume the same missing-fraud probability among the undetected rows.
    # Allocate newly recognized FNs by the observed fraud-class mix among FNs.
    # The ordinary-fraud identities cannot be recovered; this is an expected CM.
    c=cm.copy();add=c[1:,0]*missing/(1-missing)
    c[1:,0]+=add;c[0,0]-=add.sum()
    print('MISSING_SCENARIO',missing,'added_fn',float(add.sum()),'score',score(c),'delta',score(c)-score(cm))
print('pred legit rawfraud',int(((pred==0)&(y>0)).sum()),'pred legit cleanfraud',int(((pred==0)&(yc>0)).sum()))

# Pair the public upload with its own saved probabilities and saved weights.
wt=ROOT/'models/oof_clean_v4/weights_s3_clean_v4.npy'
if wt.exists():
    pp=np.load(ROOT/'models/oof_clean_v4/oof.npy').astype(float)
    predv=rules((pp*np.load(wt)).argmax(1),tr)
    print('best_upload current_rules cleaned_score',macro_f1(yc,predv))
    print('best_upload original_own_band_rule cleaned_score',macro_f1(yc,np.where(bt>0,bt,(pp*np.load(wt)).argmax(1))))
    testp=np.load(ROOT/'models/oof_clean_v4/test_proba.npy').astype(float)
    testpre=(testp*np.load(wt)).argmax(1)
    upload=pd.read_csv(ROOT/'submissions/s3_clean_v4.csv').set_index('id').label.reindex(df.txn_id.values[te]).map(dict(zip(LABELS,range(12)))).to_numpy()
    print('best_upload reconstructed_test_diff_own_band',int((np.where(b[te]>0,b[te],testpre)!=upload).sum()))
    print('best_upload reconstructed_test_diff_current_rules',int((rules(testpre,te)!=upload).sum()))
else: print('best_upload weight file absent')

detail=pd.read_parquet(ROOT/'data/processed/base_v0.parquet',columns=['txn_id','t','timestamp','sender_id','receiver_id','amount','txn_type','split','label'],filters=[[('sender_id','==','AC355256')],[('txn_id','==','TX2bc6561e')]])
print('NEAREST_BEFORE_RATIO_FAILURE')
print(detail[detail.t.between(529505,536705)].to_string(index=False))

# Compare same text across checkout newline conventions, and feature chronology.
if len(sys.argv)>1:
    rep=Path(sys.argv[1])
    for rel in ['run_all.py','src/upi/run_oof.py','src/upi/clean_labels.py','src/upi/bands.py']:
        print('normalized_source_equal',rel,(ROOT/rel).read_text()==(rep/rel).read_text())
print('DONE')
