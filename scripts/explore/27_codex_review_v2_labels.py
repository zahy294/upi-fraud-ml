"""Additional read-only review: label votes, event context and artifact provenance.

Uses the no-fit audit's runtime limit and arrays. Stdout only; no pipeline calls.
"""
from pathlib import Path
import runpy
g = runpy.run_path(str(Path(__file__).with_name('26_codex_review_v2.py')))
globals().update({k:v for k,v in g.items() if not k.startswith('__')})
rawp = np.load(ROOT/'models/oof_raw/oof.npy').astype(float)
pf = 1-rawp[:,0]; top=rawp[:,1:].argmax(1)+1
sf = pd.read_parquet(ROOT/'data/processed/features_v2.parquet',columns=['S_flag']).S_flag.to_numpy()==1
bt=b[tr]; tt=t[tr]; ss=snd[tr]; rr=rcv[tr]; dd=dev[tr]
pair=ss*2000000+rr; pc,pu=pd.factorize(pair)
rc=pd.Index(pu).get_indexer(np.where(p2p[tr],rn[tr]*2000000+ss,-1))
re=np.where(p2p[tr]|sf[tr],rr,-1)
det=(pf>=.5)|(bt>0)
votes=np.zeros((len(y),12))
for k in range(1,12):
    ind=((y==k)&det).astype(float)
    v=Events(ss,tt,ind).sum(ss,tt,-7*DAY,7*DAY)-ind
    v+=Events(re,tt,ind,mask=re>=0).sum(re,tt,-DAY,DAY)-ind*(re>=0)
    v+=Events(dd,tt,ind).sum(dd,tt,-600,600)-ind
    v+=Events(pc,tt,ind).sum(rc,tt,-7*DAY,7*DAY)
    votes[:,k]=np.maximum(v,0)
tot=votes.sum(1); evt=votes.argmax(1)
evt=np.where((votes.max(1)>=1)&(votes.max(1)/np.maximum(tot,1e-9)>=.6),evt,0)
un=(bt==0)&(y==0)&(pf>=.85)
typ=(bt==0)&(y>0)&det&(evt>0)&(evt!=y)
ban=(bt>0)&(y!=bt)
yr=y.copy();yr[un]=np.where(evt[un]>0,evt[un],top[un]);yr[typ]=evt[typ];yr[ban]=bt[ban]
assert np.array_equal(yr,yc)
fallback=un&(evt==0)
emit('cleaning branches',{'unreported':int(un.sum()),'unreported_model_fallback':int(fallback.sum()),'type':int(typ.sum()),'band':int(ban.sum()),'changed_total':int((yc!=y).sum())})
for name,mask in [('unreported',un),('fallback subset',fallback),('event type',typ),('band',ban)]:
    alt=yc.copy();alt[mask]=y[mask]
    emit('undo clean branch '+name,{'macro':macro_f1(alt,pred),'delta':macro_f1(yc,pred)-macro_f1(alt,pred),'current_pred_agrees_clean':int((pred[mask]==yc[mask]).sum()),'current_pred_agrees_raw':int((pred[mask]==y[mask]).sum()),'app3_share':float((base.app_major.values[tr][mask]=='3').mean())})

# Fraction of vote sources from another outer month (even before detector-mediated leakage).
cross=np.zeros(len(y),bool)
for mth in sorted(set(month[tr])):
    source=det&(y>0)&(month[tr]!=mth); query=month[tr]==mth
    vc=Events(ss,tt,mask=source).count(ss,tt,-7*DAY,7*DAY)
    vc+=Events(re,tt,mask=source&(re>=0)).count(re,tt,-DAY,DAY)
    vc+=Events(dd,tt,mask=source).count(dd,tt,-600,600)
    vc+=Events(pc,tt,mask=source).count(rc,tt,-7*DAY,7*DAY)
    cross|=query&(vc>0)
emit('cross-month raw-label vote exposure',{'rows':int(cross.sum()),'relabelled':int((cross&(yc!=y)).sum()),'changed_type':int((cross&typ).sum()),'unreported':int((cross&un).sum())})

# Event-vote cleaning without forcibly assigning own band, using bands as a diagnostic reference.
known=bt>0
yn=y.copy(); u=(y==0)&(pf>=.85); c=(y>0)&det&(evt>0)&(evt!=y)
yn[u]=np.where(evt[u]>0,evt[u],top[u]);yn[c]=evt[c]
emit('cleaner checked on known band rows',{'n':int(known.sum()),'raw_wrong':int(((y!=bt)&known).sum()),'vote_cleaner_wrong':int(((yn!=bt)&known).sum()),'raw_right_to_wrong':int(((y==bt)&(yn!=bt)&known).sum()),'wrong_to_right':int(((y!=bt)&(yn==bt)&known).sum())})

# Show entire band-linked forwarding roots for cases where benchmark disagrees.
wrong=(stages['chain']!=yc)&(stages['chain']!=stages['shop'])
for row in ti[wrong][:7]:
    ids=np.flatnonzero(root==root[row]); z=base.iloc[ids].copy()
    z['band']=b[ids];z['chain_band']=cb[ids]
    yall=np.full(len(base),-1);yall[tr]=yc
    z['clean']=np.where(yall[ids]>=0,np.array(LABELS)[np.maximum(yall[ids],0)],'TEST')
    emit('chain benchmark disagreement root '+str(root[row]),z)

# Concrete non-reciprocal links and broad merchant-rule queries.
non=(par>=0)&(chi[np.maximum(par,0)]!=np.arange(len(par)))
for row in np.flatnonzero(non)[:3]:
    ids=np.unique(np.r_[row,par[row],chi[par[row]]]);ids=ids[ids>=0]
    emit('non reciprocal parent example',base.iloc[ids])
for spl,rows,p0,pm,sm in [('train',tr,pre,pred,masks),('test',te,pret,predt,maskst)]:
    shop=sm['shop']&(pm!=p0); idx=np.flatnonzero(rows)[shop]
    emit(spl+' changed shop session membership',{'changed':len(idx),'S_flag_false':int((~sf[idx]).sum())})
    if len(idx):
        row=idx[0]; context=(rcv==rcv[row])&(np.abs(t-t[row])<=DAY)&((b>0)|(np.arange(len(base))==row))
        z=base[context].copy();z['band']=b[context];z['S_flag']=sf[context]
        emit(spl+' shop band context',z)

# Label-free windows can receive tail rows, but examine their actual recipe masks.
F1=pd.read_parquet(ROOT/'data/processed/features_v1.parquet',columns=['is_1p1','is_1p02'])
emit('F1 selected columns',F1.columns.tolist())
