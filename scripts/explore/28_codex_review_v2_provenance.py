"""Read-only comparison with a fresh clone; no training or artifact writes.

Usage: required-python -I scripts/explore/28_codex_review_v2_provenance.py REPRO_ROOT
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','ARROW_NUM_THREADS'):
    os.environ[k]='4'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import pyarrow as pa
pa.set_cpu_count(4);pa.set_io_thread_count(4)
sys.stdout.reconfigure(encoding='utf-8',line_buffering=True)
ROOT=Path(__file__).resolve().parents[2];REP=Path(sys.argv[1])
sys.path.insert(0,str(ROOT/'src/upi'))
from common import LABELS
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()
for rel in ['run_all.py','src/upi/run_oof.py','src/upi/clean_labels.py','src/upi/bands.py','data/processed/band_edges.json']:
    print('FILE',rel,'identical',sha(ROOT/rel)==sha(REP/rel))
for rel in ['models/s1/oof.npy','models/s1/test_proba.npy','models/oof_raw/oof.npy','models/oof_raw/y_clean.npy']:
    a=np.load(ROOT/rel);b=np.load(REP/rel)
    print('ARRAY',rel, 'shape',a.shape,'equal',np.array_equal(a,b),'changed_values',int((a!=b).sum()),'maxabs',float(np.abs(a-b).max()),'meanabs',float(np.abs(a-b).mean()),'old_sha',sha(ROOT/rel),'fresh_sha',sha(REP/rel))
    if a.ndim==2: print('argmax_changes',int((a.argmax(1)!=b.argmax(1)).sum()))
base=pd.read_parquet(ROOT/'data/processed/base_v0.parquet',columns=['txn_id','label','split'])
tr=base.split=='train';base=base[tr].reset_index(drop=True)
y=base.label.map(dict(zip(LABELS,range(12)))).to_numpy(int)
a=np.load(ROOT/'models/oof_raw/y_clean.npy');b=np.load(REP/'models/oof_raw/y_clean.npy')
P=np.load(ROOT/'models/oof_raw/oof.npy');Q=np.load(REP/'models/oof_raw/oof.npy')
fa=1-P[:,0].astype(float);fb=1-Q[:,0].astype(float);changed=a!=b
print('label_differences',int(changed.sum()),'raw_legit',int((changed&(y==0)).sum()),'raw_fraud',int((changed&(y>0)).sum()),'tau85_crossings_changed',int((changed&((fa>=.85)!=(fb>=.85))).sum()),'tau50_crossings_changed',int((changed&((fa>=.5)!=(fb>=.5))).sum()),'tau50_crossings_all',int(((fa>=.5)!=(fb>=.5)).sum()))
z=base[changed].copy();z['old_pf']=fa[changed];z['new_pf']=fb[changed];z['old_clean']=np.array(LABELS)[a[changed]];z['new_clean']=np.array(LABELS)[b[changed]]
print(z.to_string(index=False))
print('copied_fresh_labels_match',np.array_equal(b,np.load(ROOT/'models/oof_final_local/y_clean_of_this_run.npy')))
for name in ['oof.npy','test_proba.npy','weights_final_local.npy']:
    print('copied_fresh_artifact',name,np.array_equal(np.load(ROOT/'models/oof_final_local'/name),np.load(REP/'models/oof_clean'/name)))
for rel in ['models/oof_raw/oof.npy','models/oof_raw/y_clean.npy','src/upi/features_v5.py','data/processed/features_v5.parquet']:
    from datetime import datetime
    print('LOCAL_MTIME',rel,datetime.fromtimestamp((ROOT/rel).stat().st_mtime).isoformat())
for name in ['run_all.log','run_all_resume.log']:
    path=REP/name
    if path.exists():
        for ln,line in enumerate(path.read_text(encoding='utf-8',errors='replace').splitlines(),1):
            if '[run ]' in line or '[skip]' in line or 'pooled out' in line: print('RUN_LOG',name,ln,line)
for name in ['s5_c3','s3_clean_v4']:
    a=pd.read_csv(ROOT/'submissions'/f'{name}.csv').set_index('id').label
    b=pd.read_csv(REP/'submissions/final_local.csv').set_index('id').label.reindex(a.index)
    print('SUBMISSION',name,'changed',int((a!=b).sum()),'sha256',sha(ROOT/'submissions'/f'{name}.csv'))
print('DONE')
