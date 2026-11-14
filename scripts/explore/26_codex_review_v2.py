"""Read-only v2 pipeline audit. No fitting; saved probabilities and weights only.

Run from the repo with the required Python executable and -I. Writes stdout only.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'ARROW_NUM_THREADS'):
    os.environ[_key] = '4'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import ast
import json
import importlib.metadata
import numpy as np
import pandas as pd
import pyarrow as pa
pa.set_cpu_count(4)
pa.set_io_thread_count(4)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/upi'))
from common import LABELS, macro_f1, per_class
from events import Events, HOUR, DAY
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
pd.set_option('display.width', 240)
pd.set_option('display.max_columns', 30)

def emit(name, value):
    print('\n###', name, flush=True)
    print(value.to_string() if isinstance(value, pd.DataFrame) else value, flush=True)

def chain_arrays(df):
    z = df[['t','sender_id','receiver_id','receiver_type','amount']].copy()
    z['row'] = np.arange(len(z))
    q = z[['row','t','sender_id','amount']].rename(columns={'sender_id':'k'}).sort_values('t')
    inn = z[z.receiver_type == 'P2P'][['row','t','receiver_id','amount']].rename(
        columns={'receiver_id':'k','row':'p_row','amount':'p_amt','t':'p_t'}).sort_values('p_t')
    m = pd.merge_asof(q, inn, left_on='t', right_on='p_t', by='k', direction='backward', allow_exact_matches=False, tolerance=7200).set_index('row')
    ratio = (m.amount / m.p_amt).reindex(z.row).values
    par = np.where((ratio >= .9) & (ratio <= 1), m.p_row.reindex(z.row).fillna(-1).values, -1).astype(int)
    q = z[z.receiver_type == 'P2P'][['row','t','receiver_id','amount']].rename(columns={'receiver_id':'k'}).sort_values('t')
    out = z[['row','t','sender_id','amount']].rename(columns={'sender_id':'k','row':'c_row','amount':'c_amt','t':'c_t'}).sort_values('c_t')
    m = pd.merge_asof(q, out, left_on='t', right_on='c_t', by='k', direction='forward', allow_exact_matches=False, tolerance=7200).set_index('row')
    ratio = (m.c_amt / m.amount).reindex(z.row).values
    chi = np.where((ratio >= .9) & (ratio <= 1), m.c_row.reindex(z.row).fillna(-1).values, -1).astype(int)
    def walk(ptr):
        depth = np.zeros(len(z), int); end = np.arange(len(z)); cur = ptr.copy()
        for _ in range(8):
            has = cur >= 0; depth += has; end = np.where(has, cur, end)
            cur = np.where(has, ptr[np.maximum(cur, 0)], -1)
        return depth, end, cur
    db, root, remaining = walk(par)
    dfw, leaf, remf = walk(chi)
    return par, chi, db, dfw, root, leaf, remaining, remf

def chain_band(b, root, inc):
    z = pd.DataFrame({'root':root[inc], 'band':b[inc]})
    g = z[z.band > 0].groupby('root').band.agg(['min','max'])
    one = g[g['min'] == g['max']]['max']
    return np.where(inc, pd.Series(root).map(one).fillna(0).values, 0).astype(int)

def rules(pred, rows, b, S, R, cb):
    bb = b[rows]; s = S[rows]; r = R[rows]; c = cb[rows]
    sb = np.where((s > 0).sum(1) == 1, s.argmax(1) + 1, 0)
    rb = np.where((r > 0).sum(1) == 1, r.argmax(1) + 1, 0)
    free = (bb == 0) & (pred > 0)
    masks = {'own':bb > 0,
             'drain':free & np.isin(pred,[4,5]) & np.isin(sb,[4,5]),
             'fanin':free & np.isin(pred,[1,2,3]) & p2p[rows] & np.isin(rb,[1,2,3]),
             'shop':free & np.isin(pred,[7,8,11]) & ~p2p[rows] & np.isin(rb,[7,8,11]),
             'chain':free & (c > 0)}
    out = pred.copy(); stages = {}
    for name, val in [('own',bb),('drain',sb),('fanin',rb),('shop',rb),('chain',c)]:
        out = np.where(masks[name], val, out); stages[name] = out.copy()
    return out, stages, masks

def neighbors(b):
    S = np.empty((len(b),11),np.float32); R = S.copy()
    for k in range(1,12):
        S[:,k-1] = Events(snd,t,mask=b==k).count(snd,t,-3*HOUR,3*HOUR)
        R[:,k-1] = Events(rcv,t,mask=b==k).count(rcv,t,-DAY,DAY)
    return S,R

emit('runtime', {'python':sys.version, 'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','pyarrow','scipy','scikit-learn','lightgbm']}})
base = pd.read_parquet(ROOT/'data/processed/base_v0.parquet', columns=['txn_id','t','month','timestamp','sender_id','receiver_id','receiver_type','device_id','amount','balance_before','txn_type','label','split','app_major','ip_risk_score'])
tr = (base.split == 'train').values; te = ~tr; ti = np.flatnonzero(tr); ei = np.flatnonzero(te)
month = base.month.astype(str).values.copy(); month[tr & (month > '2026-09')] = '2026-09'
t = base.t.to_numpy(np.int64); snd = base.sender_id.str[2:].to_numpy(np.int64)
dev = base.device_id.str[2:].to_numpy(np.int64); p2p = (base.receiver_type == 'P2P').values
rn = base.receiver_id.str[2:].to_numpy(np.int64); rcv = np.where(p2p,rn,rn+1000000)
y = base.label[tr].map(dict(zip(LABELS,range(12)))).to_numpy(int)
yc = np.load(ROOT/'models/oof_raw/y_clean.npy')
F3 = pd.read_parquet(ROOT/'data/processed/features_v3.parquet')
F5 = pd.read_parquet(ROOT/'data/processed/features_v5.parquet')
b = F3.band.to_numpy(int); cb = F5.ch_band.to_numpy(int)
S = F3[[f'snd_band_{k}_3h' for k in range(1,12)]].to_numpy()
R = F3[[f'rcv_band_{k}_24h' for k in range(1,12)]].to_numpy()
P = np.load(ROOT/'models/oof_c3/oof.npy').astype(float)
Pt = np.load(ROOT/'models/oof_c3/test_proba.npy').astype(float)
w = np.load(ROOT/'models/oof_c3/weights_s5_c3.npy')
pre = (P*w).argmax(1); pret = (Pt*w).argmax(1)
pred, stages, masks = rules(pre,tr,b,S,R,cb)
predt, stagest, maskst = rules(pret,te,b,S,R,cb)
emit('scores', {'weighted_before':macro_f1(yc,pre),'own_band':macro_f1(yc,stages['own']),'all_rules':macro_f1(yc,pred),'raw_all_rules':macro_f1(y,pred),'argmax_all_rules':macro_f1(yc,rules(P.argmax(1),tr,b,S,R,cb)[0])})
emit('per_class', per_class(yc,pred).assign(weight=w, test_predicted=np.bincount(predt,minlength=12)))
for name in stages:
    ch = (stages[name] != (pre if name=='own' else stages[list(stages)[list(stages).index(name)-1]]))
    cht = (stagest[name] != (pret if name=='own' else stagest[list(stagest)[list(stagest).index(name)-1]]))
    emit('rule '+name, {'train_changed':int(ch.sum()),'test_changed':int(cht.sum()),'clean_right':int((stages[name][ch]==yc[ch]).sum()),'raw_right':int((stages[name][ch]==y[ch]).sum()), 'score':macro_f1(yc,stages[name])})
    wrong = ch & (stages[name]!=yc)
    ix = ti[np.flatnonzero(wrong)[:4]]
    if len(ix):
        z=base.iloc[ix].copy(); z['band']=b[ix]; z['clean']=np.array(LABELS)[yc[wrong][:4]]; z['before']=np.array(LABELS)[pre[wrong][:4]]; z['after']=np.array(LABELS)[stages[name][wrong][:4]]; emit('wrong examples '+name,z)
emit('monthly',pd.DataFrame([{'month':m,'score':macro_f1(yc[month[tr]==m],pred[month[tr]==m]),'raw':macro_f1(y[month[tr]==m],pred[month[tr]==m])} for m in sorted(set(month[tr]))]))

# Extract only pure helper definitions, never execute the band builder.
tree=ast.parse((ROOT/'src/upi/bands.py').read_text())
ns={'np':np,'K':11}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'bands_helpers','exec'),ns)
segment,band_scores=ns['segment'],ns['band_scores']
edges=json.loads((ROOT/'data/processed/band_edges.json').read_text())
ed=np.array(edges['train']); edte=np.array(edges['test']); width=ed[-1]-ed[0]
Ps=np.load(ROOT/'models/s1/oof.npy').astype(float)
kind=pd.read_parquet(ROOT/'data/processed/features_v1.parquet',columns=['snd_drain_kind']).snd_drain_kind.to_numpy(int)
dm=(dev[tr]>=ed[0])&(dev[tr]<ed[1]); ds=(dev[tr]>=ed[3])&(dev[tr]<ed[4])
am=dm&(Ps[:,1]>=.5); ass=ds&(Ps[:,4]+Ps[:,5]>=.5)&(kind[tr]==1)
lm=am.sum()/(ed[1]-ed[0]); ls=ass.sum()/(ed[4]-ed[3])
emit('density row/device issue',{'mule_rows':int(am.sum()),'mule_unique_devices':int(np.unique(dev[tr][am]).size),'lambda_mule':lm,'sim_rows':int(ass.sum()),'sim_unique_devices':int(np.unique(dev[tr][ass]).size),'lambda_sim':ls})
blk=(dev[tr]>=ed[0])&(dev[tr]<ed[-1])
sc=band_scores(Ps[blk],kind[tr][blk],dev[tr][blk],ed[0],width,lm,ls)
cuts=segment(sc,prior=(edte[1:-1]-edte[0])/(edte[-1]-edte[0])*width,sd=.02*width)
chk=np.r_[ed[0],ed[0]+np.array(cuts),ed[-1]]
bb=b.copy(); bb[ti[blk]]=np.searchsorted(chk,dev[tr][blk],side='right')
emit('blind train bands',{'edges':chk.tolist(),'edge_errors':(chk-ed).tolist(),'row_changes':int((bb!=b).sum())})
par,chi,db,dfw,root,leaf,remaining,remf=chain_arrays(base)
inc=(db+dfw)>0
assert np.array_equal(chain_band(b,root,inc),cb)
assert np.array_equal(db,F5.ch_depth_back.values) and np.array_equal(dfw,F5.ch_depth_fwd.values)
bs,br=neighbors(bb); bcb=chain_band(bb,root,inc)
bp=rules(pre,tr,bb,bs,br,bcb)[0]
emit('blind train frozen model',{'macro':macro_f1(yc,bp),'delta':macro_f1(yc,bp)-macro_f1(yc,pred),'changed_predictions':int((bp!=pred).sum()),'right_to_wrong':int(((pred==yc)&(bp!=yc)).sum()),'wrong_to_right':int(((pred!=yc)&(bp==yc)).sum())})
emit('blind wrong class changes',pd.crosstab(pd.Series(np.array(LABELS)[pred[bp!=pred]],name='oracle_pred'),pd.Series(np.array(LABELS)[bp[bp!=pred]],name='blind_pred')))
for m in sorted(set(month[tr])):
    score=np.zeros((width,11)); mask=blk&(month[tr]!=m)&(y>0)
    np.add.at(score,(dev[tr][mask]-ed[0],y[mask]-1),1.)
    cut=segment(score); edge=np.r_[ed[0],ed[0]+np.array(cut),ed[-1]]
    held=blk&(month[tr]==m)
    emit('leave month out label edges '+m,{'edges':edge.tolist(),'held_rows_changed':int((np.searchsorted(edge,dev[tr][held],side='right')!=b[tr][held]).sum())})

# Point perturbations: sensitivity, not an estimate of test errors.
Pst=np.load(ROOT/'models/s1/test_proba.npy').astype(float)
tblk=(dev[te]>=edte[0])&(dev[te]<edte[-1]); tw=edte[-1]-edte[0]
tsc=band_scores(Pst[tblk],kind[te][tblk],dev[te][tblk],edte[0],tw,lm,ls)
testcuts=segment(tsc,prior=(ed[1:-1]-ed[0])/width*tw,sd=.02*tw)
assert np.array_equal(np.r_[edte[0],edte[0]+np.array(testcuts),edte[-1]],edte)
emit('test edges reproduced',edte.tolist())
emit('chain integrity',{'parent_links':int((par>=0).sum()),'child_links':int((chi>=0).sum()),'parent_not_reciprocal':int(((par>=0)&(chi[np.maximum(par,0)]!=np.arange(len(par)))).sum()),'child_not_reciprocal':int(((chi>=0)&(par[np.maximum(chi,0)]!=np.arange(len(chi)))).sum()),'over8parent':int((remaining>=0).sum()),'over8child':int((remf>=0).sum()),'linked_non_PAY_rows':int((inc&(base.txn_type.values!='PAY')).sum())})
cr=(par>=0)&(tr!=tr[np.maximum(par,0)])
cc=(chi>=0)&(tr!=tr[np.maximum(chi,0)])
emit('cross split chain links',{'parent':int(cr.sum()),'child':int(cc.sum())})
if cr.any(): emit('cross split parent examples',base.iloc[np.unique(np.r_[np.flatnonzero(cr),par[cr]])[:12]])

tails=tr&(base.month.astype(str).values>'2026-09')
emit('October train tails',base[tails].groupby('label',observed=True).size())
touch={}
for name,key,win in [('sender3h',snd,3*HOUR),('sender7d',snd,7*DAY),('receiver24h',rcv,DAY),('device10min',dev,600)]:
    v=Events(key,t,mask=tails).count(key[te],t[te],-win,win)>0
    touch[name]=int(v.sum())
emit('test rows touching tails',touch)
nt=b.copy(); nt[tails]=0
ns_,nr_=neighbors(nt); nc=chain_band(nt,root,inc)
ntp=rules(pret,te,nt,ns_,nr_,nc)[0]
emit('remove tail band sources',{'tail_band_rows':int((tails&(b>0)).sum()),'test_S_changed':int(np.any(S[te]!=ns_[te],1).sum()),'test_R_changed':int(np.any(R[te]!=nr_[te],1).sum()),'test_chain_band_changed':int((cb[te]!=nc[te]).sum()),'test_predictions_changed':int((ntp!=predt).sum())})

fresh=ROOT/'models/oof_final_local'
if fresh.exists():
    yf=np.load(fresh/'y_clean_of_this_run.npy'); Pf=np.load(fresh/'oof.npy').astype(float); Ptf=np.load(fresh/'test_proba.npy').astype(float); wf=np.load(fresh/'weights_final_local.npy')
    pfr=rules((Pf*wf).argmax(1),tr,b,S,R,cb)[0]; pft=rules((Ptf*wf).argmax(1),te,b,S,R,cb)[0]
    dy=yf!=yc
    emit('fresh run comparison',{'label_differences':int(dy.sum()),'oof_changed_rows':int(np.any(P!=Pf,1).sum()),'test_changed_rows':int(np.any(Pt!=Ptf,1).sum()),'test_final_pred_differences':int((pft!=predt).sum()),'fresh_score_own_labels':macro_f1(yf,pfr),'fresh_score_original_labels':macro_f1(yc,pfr),'old_score_fresh_labels':macro_f1(yf,pred),'old_score_old_labels':macro_f1(yc,pred)})
    fpre=(Pf*wf).argmax(1);fpret=(Ptf*wf).argmax(1)
    _,fs,fm=rules(fpre,tr,b,S,R,cb);_,fts,ftm=rules(fpret,te,b,S,R,cb)
    fblind=rules(fpre,tr,bb,bs,br,bcb)[0]
    ftail=rules(fpret,te,nt,ns_,nr_,nc)[0]
    emit('latest candidate frozen score checks',{'weighted_before':macro_f1(yf,fpre),'own_band':macro_f1(yf,fs['own']),'all_rules':macro_f1(yf,pfr),'raw_all_rules':macro_f1(y,pfr),'argmax_all_rules':macro_f1(yf,rules(Pf.argmax(1),tr,b,S,R,cb)[0]),'blind_score':macro_f1(yf,fblind),'blind_delta':macro_f1(yf,fblind)-macro_f1(yf,pfr),'blind_changed':int((fblind!=pfr).sum()),'remove_tail_test_changed':int((ftail!=pft).sum())})
    previous=fpre;previoust=fpret
    for name in fs:
        changed=fs[name]!=previous;changedt=fts[name]!=previoust
        emit('latest rule '+name,{'train_changed':int(changed.sum()),'test_changed':int(changedt.sum()),'clean_right':int((fs[name][changed]==yf[changed]).sum()),'raw_right':int((fs[name][changed]==y[changed]).sum()),'score':macro_f1(yf,fs[name])})
        previous=fs[name];previoust=fts[name]
    emit('label differences',pd.crosstab(pd.Series(np.array(LABELS)[yc[dy]],name='old'),pd.Series(np.array(LABELS)[yf[dy]],name='fresh')))
    rawp=np.load(ROOT/'models/oof_raw/oof.npy').astype(float); pf=1-rawp[:,0]
    z=base.iloc[ti[dy]].copy(); z['old_clean']=np.array(LABELS)[yc[dy]]; z['fresh_clean']=np.array(LABELS)[yf[dy]]; z['raw_pf']=pf[dy]; emit('changed label details',z)

# Counterfactual mixture: scale detected predictions by observed October/OOF rates,
# preserve within-prediction precision and the class distribution of missed fraud.
# This assumes identical undetected-fraud density; it does not observe test labels.
cm=np.bincount(yc*12+pred,minlength=144).reshape(12,12).astype(float)
rate_ratio=(np.bincount(predt,minlength=12)/te.sum())/(np.bincount(pred,minlength=12)/tr.sum())
def cmscore(c): return float(np.mean(2*np.diag(c)/np.maximum(c.sum(0)+c.sum(1),1e-15)))
mix=cm.copy(); mix[:,1:]*=rate_ratio[None,1:]
mix[0,0]=tr.sum()-mix.sum()+mix[0,0]
emit('October detection-rate mixture scenario',{'score':cmscore(mix),'delta':cmscore(mix)-cmscore(cm),'ratios':dict(zip(LABELS,rate_ratio.round(4)))})
for k in range(1,12):
    one=cm.copy(); one[:,k]*=rate_ratio[k]; one[0,0]=tr.sum()-one.sum()+one[0,0]
    emit('mixture only '+LABELS[k],cmscore(one)-cmscore(cm))

# Public subset sampling uncertainty, conditional on local confusion frequencies.
# A multinomial table is sufficient for macro-F1; no model/threshold fitting.
rng=np.random.default_rng(20261114); n_public=round(te.sum()*.3)
sim=rng.multinomial(n_public,(cm/cm.sum()).ravel(),size=2000).reshape(-1,12,12)
scores=(2*np.diagonal(sim,axis1=1,axis2=2)/np.maximum(sim.sum(1)+sim.sum(2),1)).mean(1)
emit('public 30 percent sampling scenario',{'rows':n_public,'mean':float(scores.mean()),'sd':float(scores.std()),'p025':float(np.quantile(scores,.025)),'p975':float(np.quantile(scores,.975))})
emit('DONE','No fits; no data/model/submission writes.')
