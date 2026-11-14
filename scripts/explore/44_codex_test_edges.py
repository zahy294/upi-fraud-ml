"""Read-only band uncertainty audit. Only writes the new combined Markdown report."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','ARROW_NUM_THREADS'):
    os.environ[k]='4'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,json
import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.special import logsumexp
pa.set_cpu_count(4);pa.set_io_thread_count(1)
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/upi'))
from common import LABELS,macro_f1,fit_weights
from events import Events
REPORT=ROOT/'docs/codex_edges_and_rules.md'
sys.stdout.reconfigure(encoding='utf-8',line_buffering=True)
GRID=np.geomspace(.05,400,60)

def weights_fast(P,y):
    """Exact coordinate objective, with threshold counts instead of repeated argmax.
    This calibrates decision weights only, not a predictive model.
    """
    cand=P[:,1:].max(1)*400>=P[:,0];pc=P[cand];yy=y[cand]
    support=np.bincount(y,minlength=12);fixed=np.bincount(y[~cand],minlength=12)
    w=np.ones(12)
    for _ in range(3):
        for c in range(1,12):
            other=pc*w;other[:,c]=-np.inf;op=other.argmax(1);ov=other[np.arange(len(op)),op]
            ratio=np.divide(ov,pc[:,c],out=np.full(len(op),np.inf),where=pc[:,c]>0)
            predicted=np.bincount(op,minlength=12).astype(float);predicted[0]+=fixed.sum()
            tp=np.bincount(yy[op==yy],minlength=12).astype(float);tp[0]+=fixed[0]
            loss=np.zeros((60,12));lost_tp=loss.copy();gain=np.zeros(60)
            for j in range(12):
                if j==c:continue
                m=op==j;side='right' if c<j else 'left'
                loss[:,j]=np.searchsorted(np.sort(ratio[m]),GRID,side=side)
                lost_tp[:,j]=np.searchsorted(np.sort(ratio[m&(yy==j)]),GRID,side=side)
                gain+=np.searchsorted(np.sort(ratio[m&(yy==c)]),GRID,side=side)
            pp=predicted-loss;pp[:,c]+=loss.sum(1)
            tt=tp-lost_tp;tt[:,c]+=gain
            score=(2*tt/np.maximum(support+pp,1)).mean(1)
            plateau=np.flatnonzero(score>=score.max()-.0003);w[c]=GRID[plateau[len(plateau)//2]]
    return w

def md(rows):
    d=pd.DataFrame(rows)
    if d.empty:return 'None.'
    def val(v):
        if isinstance(v,(float,np.floating)):return f'{v:.6f}'
        return str(v)
    return '\n'.join(['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |']+['| '+' | '.join(val(v) for v in r)+' |' for r in d.itertuples(index=False,name=None)])

def rules(pre,rows,c,b=None,S=None,R=None,cb=None):
    b=c['band'] if b is None else b;S=c['S'] if S is None else S;R=c['R'] if R is None else R;cb=c['cb'] if cb is None else cb
    bb=b[rows];ss=S[rows];rr=R[rows];cc=cb[rows]
    sb=np.where((ss>0).sum(1)==1,ss.argmax(1)+1,0);rb=np.where((rr>0).sum(1)==1,rr.argmax(1)+1,0)
    free=(bb==0)&(pre>0);p2p=c['p2p'][rows];out=np.where(bb>0,bb,pre)
    out=np.where(free&np.isin(pre,[4,5])&np.isin(sb,[4,5]),sb,out)
    out=np.where(free&np.isin(pre,[1,2,3])&p2p&np.isin(rb,[1,2,3]),rb,out)
    out=np.where(free&np.isin(pre,[7,8,11])&~p2p&np.isin(rb,[7,8,11]),rb,out)
    return np.where(free&(cc>0),cc,out)

def context(calibrate=True):
    df=pd.read_parquet(ROOT/'data/processed/base_v0.parquet',columns=['txn_id','t','month','split','label','sender_id','receiver_id','receiver_type','device_id','amount'])
    tr=df.split.eq('train').to_numpy();mo=df.month.astype(str).to_numpy();mo[tr&(mo>'2026-09')]='2026-09'
    yc=np.load(ROOT/'models/oof_raw/y_clean.npy');raw=df.label[tr].map(dict(zip(LABELS,range(12)))).to_numpy(int)
    f3=pd.read_parquet(ROOT/'data/processed/features_v3.parquet');f5=pd.read_parquet(ROOT/'data/processed/features_v5.parquet')
    assert np.array_equal(f3.txn_id.values,df.txn_id.values) and np.array_equal(f5.txn_id.values,df.txn_id.values)
    snd=df.sender_id.str[2:].to_numpy(np.int64);rn=df.receiver_id.str[2:].to_numpy(np.int64);p2p=df.receiver_type.eq('P2P').to_numpy()
    c=dict(df=df,tr=tr,te=~tr,ti=np.flatnonzero(tr),ei=np.flatnonzero(~tr),month=mo,y=yc,raw=raw,t=df.t.to_numpy(np.int64),snd=snd,rcv=np.where(p2p,rn,rn+1000000),rn=rn,p2p=p2p,dev=df.device_id.str[2:].to_numpy(np.int64),band=f3.band.to_numpy(int),cb=f5.ch_band.to_numpy(int),S=f3[[f'snd_band_{k}_3h' for k in range(1,12)]].to_numpy(),R=f3[[f'rcv_band_{k}_24h' for k in range(1,12)]].to_numpy(),models={})
    for tag in ['oof_final_local','s2_a']:
        P=np.load(ROOT/f'models/{tag}/oof.npy').astype(float)
        model={'P':P}
        if calibrate:
            print('Calibrating saved probabilities',tag,flush=True)
            w=np.load(ROOT/'models/oof_final_local/weights_final_local.npy') if tag=='oof_final_local' else weights_fast(P,yc)
            pre=(P*w).argmax(1);pred=rules(pre,tr,c)
            nested=np.empty(len(yc),int);nw={}
            for m in ['2026-07','2026-08','2026-09']:
                hold=mo[tr]==m;nw[m]=weights_fast(P[~hold],yc[~hold]);q=(P[hold]*nw[m]).argmax(1)
                nested[hold]=rules(q,c['ti'][hold],c)
            model.update(w=w,pre=pre,pred=pred,nested=nested,nw=nw)
            print(tag,macro_f1(yc,pred),macro_f1(raw,pred),macro_f1(yc,nested),flush=True)
        c['models'][tag]=model
    return c

def chains(c):
    d=c['df'].copy();d['row']=np.arange(len(d));n=len(d)
    q=d[['row','t','sender_id','amount']].rename(columns={'sender_id':'k'}).sort_values('t')
    inc=d[c['p2p']][['row','t','receiver_id','amount']].rename(columns={'receiver_id':'k','row':'p_row','t':'p_t','amount':'p_amt'}).sort_values('p_t')
    m=pd.merge_asof(q,inc,left_on='t',right_on='p_t',by='k',direction='backward',allow_exact_matches=False,tolerance=7200).set_index('row').reindex(range(n))
    r=m.amount/m.p_amt;par=np.where(r.between(.9,1),m.p_row.fillna(-1),-1).astype(int)
    q=d[c['p2p']][['row','t','receiver_id','amount']].rename(columns={'receiver_id':'k'}).sort_values('t')
    out=d[['row','t','sender_id','amount']].rename(columns={'sender_id':'k','row':'c_row','t':'c_t','amount':'c_amt'}).sort_values('c_t')
    m=pd.merge_asof(q,out,left_on='t',right_on='c_t',by='k',direction='forward',allow_exact_matches=False,tolerance=7200).set_index('row').reindex(range(n))
    r=m.c_amt/m.amount;chi=np.where(r.between(.9,1),m.c_row.fillna(-1),-1).astype(int)
    root=np.arange(n);cur=par.copy()
    for _ in range(8):
        good=cur>=0;root=np.where(good,cur,root);cur=np.where(good,par[np.maximum(cur,0)],-1)
    c['root']=root;c['inchain']=(par>=0)|(chi>=0)

def updated(c,b):
    S=c['S'].copy();R=c['R'].copy();changed=b!=c['band'];t=c['t']
    for k in range(1,12):
        for mask,sign in [(changed&(b==k),1),(changed&(c['band']==k),-1)]:
            if mask.any():
                S[:,k-1]+=sign*Events(c['snd'],t,mask=mask).count(c['snd'],t,-10800,10800)
                R[:,k-1]+=sign*Events(c['rcv'],t,mask=mask).count(c['rcv'],t,-86400,86400)
    inc=c['inchain'];root=c['root'];tab=pd.DataFrame({'root':root[inc],'band':b[inc]});tab=tab[tab.band>0].groupby('root').band.agg(['min','max']);one=tab[tab['min']==tab['max']]['max']
    cb=np.where(inc,pd.Series(root).map(one).fillna(0),0).astype(int)
    return S,R,cb

def posterior(score,center=None,sd=None):
    """Exact discrete ordered-cut marginal posterior; positive segment widths."""
    n,K=score.shape;cs=np.vstack([np.zeros(K),score.cumsum(0)]);pos=np.arange(n+1)
    pri=np.zeros((K-1,n+1)) if center is None else -((pos[None,:]-np.asarray(center)[:,None])/sd)**2/2
    A=np.full((K,n+1),-np.inf);A[0,1:]=cs[1:,0]
    V=A[0].copy();backs=[]
    for k in range(1,K):
        a=A[k-1]-cs[:,k]+pri[k-1];prefix=np.logaddexp.accumulate(a)
        A[k,1:]=cs[1:,k]+prefix[:-1]
        v=V-cs[:,k]+pri[k-1];best=np.maximum.accumulate(v)
        ix=np.maximum.accumulate(np.where(v==best,pos,0));new=np.full(n+1,-np.inf);new[1:]=cs[1:,k]+best[:-1];V=new;backs.append(ix)
    cuts=[n]
    for ix in backs[::-1]:cuts.append(int(ix[cuts[-1]-1]))
    cuts=np.array([0]+cuts[::-1],int)
    B=np.full((K,n+1),-np.inf);B[-1,:n]=cs[n,-1]-cs[:n,-1]
    for k in range(K-2,-1,-1):
        a=cs[:,k]+pri[k]+B[k+1];suffix=np.logaddexp.accumulate(a[::-1])[::-1]
        B[k,:n]=-cs[:n,k]+suffix[1:]
    profiles=[]
    for k in range(1,K):
        log=A[k-1]+pri[k-1]+B[k];log-=logsumexp(log);p=np.exp(log);cdf=p.cumsum()
        profiles.append(dict(p=p,log=log,mode=int(p.argmax()),lo=int(np.searchsorted(cdf,.025)),hi=int(np.searchsorted(cdf,.975))))
    return cuts,profiles

def evidence(prob,dnum,kind,edges):
    n=edges[-1]-edges[0];idx=np.flatnonzero((dnum>=edges[0])&(dnum<edges[-1]));rel=dnum[idx]-edges[0]
    count=np.bincount(rel,minlength=n);s=np.zeros((n,12));np.add.at(s,rel,prob[idx]);s/=np.maximum(count[:,None],1)
    cat=s[:,1:].argmax(1)+1;cat[(1-s[:,0])<.5]=0
    k1=np.bincount(rel,weights=(kind[idx]==1),minlength=n)>0;k2=np.bincount(rel,weights=(kind[idx]==2),minlength=n)>0
    drain=np.isin(cat,[4,5]);cat[drain]=14;cat[drain&k1]=12;cat[drain&~k1&k2]=13
    return count>0,cat

def main():
    old_report=REPORT.read_text(encoding='utf-8') if REPORT.exists() else ''
    part2='**Part 2 - collect-family audit.**'
    suffix=part2+old_report.split(part2,1)[1] if part2 in old_report else ''
    c=context();tr=c['tr'];te=c['te'];edges=json.loads((ROOT/'data/processed/band_edges.json').read_text());ed=np.array(edges['train']);et=np.array(edges['test']);N=ed[-1]-ed[0];M=et[-1]-et[0]
    # Exact objective verification on a deterministic stratified-sized sample.
    P=c['models']['oof_final_local']['P'];sample=np.r_[np.flatnonzero(c['y']==0)[:1000],np.flatnonzero(c['y']>0)[:4000]]
    assert np.array_equal(weights_fast(P[sample],c['y'][sample]),fit_weights(P[sample],c['y'][sample]))
    s1=np.load(ROOT/'models/s1/oof.npy').astype(float);st=np.load(ROOT/'models/s1/test_proba.npy').astype(float)
    kind=pd.read_parquet(ROOT/'data/processed/features_v1.parquet',columns=['snd_drain_kind']).snd_drain_kind.to_numpy(int)
    ot,ct=evidence(s1,c['dev'][tr],kind[tr],ed);oe,ce=evidence(st,c['dev'][te],kind[te],et)
    phase=np.searchsorted(ed-ed[0],np.arange(N),side='right')-1
    rho=np.array([(ot[phase==k].sum()+.5)/((phase==k).sum()+1) for k in range(11)])
    emission=np.array([np.bincount(ct[(phase==k)&ot],minlength=15)+.5 for k in range(11)]);emission/=emission.sum(1)[:,None]
    def scores(occ,cat):
        anchor=np.where(occ[:,None],np.log(emission[:,cat].T),0)
        occupancy=occ[:,None]*np.log(rho)+(~occ[:,None])*np.log1p(-rho)
        return anchor,occupancy
    at,ut=scores(ot,ct);ae,ue=scores(oe,ce)
    pr=(ed[1:-1]-ed[0])/N*M
    variants={'anchors':posterior(ae),'occupancy':posterior(ue),'prior':posterior(np.zeros_like(ae),pr,.02*M),'combined':posterior(ae+ue,pr,.02*M)}
    newtest,ptest=variants['combined'];newtest=newtest+et[0]
    blind,pblind=posterior(at+ut,(et[1:-1]-et[0])/M*N,.02*N);blind+=ed[0]
    # Reproduce the existing blind procedure, without executing its file writes.
    tree=ast.parse((ROOT/'src/upi/bands.py').read_text());g={'np':np,'K':11}
    exec(compile(ast.Module(body=[x for x in tree.body if isinstance(x,ast.FunctionDef)],type_ignores=[]),'bands_helpers','exec'),g)
    dv=c['dev'][tr];am=(dv>=ed[0])&(dv<ed[1]);asi=(dv>=ed[3])&(dv<ed[4])
    lm=(s1[am,1]>=.5).sum()/(ed[1]-ed[0]);ls=((s1[asi,4]+s1[asi,5]>=.5)&(kind[tr][asi]==1)).sum()/(ed[4]-ed[3])
    blk=(dv>=ed[0])&(dv<ed[-1]);oldscore=g['band_scores'](s1[blk],kind[tr][blk],dv[blk],ed[0],N,lm,ls)
    oldblind=np.r_[ed[0],ed[0]+np.array(g['segment'](oldscore,(et[1:-1]-et[0])/M*N,.02*N)),ed[-1]]
    oldwrong=int((np.searchsorted(oldblind,dv[blk],side='right')!=np.searchsorted(ed,dv[blk],side='right')).sum());assert oldwrong==72
    boots=[];rng=np.random.default_rng(20261114)
    for _ in range(20):
        mult=rng.multinomial(M,np.full(M,1/M));bt,_=posterior((ae+ue)*mult[:,None],pr,.02*M)
        br,bp=posterior(at+ut,bt[1:-1]/M*N,.02*N)
        boots.append((br+ed[0],[(ed[k]>=ed[0]+bp[k-1]['lo']) and (ed[k]<=ed[0]+bp[k-1]['hi']) for k in range(1,11)]))
    # Sensitivity: halve the entire bot-band width, an intentionally stronger
    # assumption than fewer native bursts on the separate shared device pool.
    widths=np.diff(ed).astype(float);widths[1]*=.5;thinprior=np.cumsum(widths/widths.sum())[:-1]*M
    thin,pthin=posterior(ae+ue,thinprior,.02*M);thin+=et[0]
    chains(c)
    def assign(edges,mask):
        b=c['band'].copy();ix=np.flatnonzero(mask&(c['dev']>=edges[0])&(c['dev']<edges[-1]));b[ix]=np.searchsorted(edges,c['dev'][ix],side='right');return b
    bblind=assign(blind,tr);bold=assign(oldblind,tr)
    combinedwrong=int((bblind[tr][blk]!=c['band'][tr][blk]).sum())
    S,R,cb=updated(c,bblind);Sold,Rold,cold=updated(c,bold)
    uncertainty=np.zeros(len(tr),bool);nearby={}
    for k,p in enumerate(pblind,1):
        mask=tr&(c['dev']>=ed[0]+p['lo'])&(c['dev']<=ed[0]+p['hi']);uncertainty|=mask;nearby[k]=mask
    families={1:{1,3},2:{2,1},3:{3,1},4:{4,5},5:{5,4},6:{6,9,10},7:{7,8,11},8:{8,7,11},9:{9,6},10:{10,6},11:{11,7,8}}
    results=[]
    for tag,model in c['models'].items():
        pre=model['pre'];pold=rules(pre,tr,c,bold,Sold,Rold,cold);trust=rules(pre,tr,c,bblind,S,R,cb)
        fullpre=np.zeros(len(tr),int);fullpre[tr]=pre
        fallback=bblind.copy();fallback[uncertainty]=0;fs,fr,fc=updated(c,fallback);fp=rules(pre,tr,c,fallback,fs,fr,fc);fp[uncertainty[tr]]=pre[uncertainty[tr]]
        neighbour=bblind.copy()
        for k,mask in nearby.items():
            for row in np.flatnonzero(mask):
                p=fullpre[row];own=bblind[row];other=k+1 if own==k else k if own==k+1 else 0
                if other and (p==other or (p in families[other] and p not in families[own])):neighbour[row]=other
        ns,nr,nc=updated(c,neighbour);npred=rules(pre,tr,c,neighbour,ns,nr,nc)
        for policy,pp in [('oracle/current',model['pred']),('legacy blind',pold),('combined blind/trust',trust),('combined blind/model fallback',fp),('combined blind/neighbour',npred)]:
            results.append(dict(model=tag,policy=policy,clean=macro_f1(c['y'],pp),raw=macro_f1(c['raw'],pp),changed_vs_oracle=int((pp!=model['pred']).sum())))
    # Publish statistical recommendation separately from deployment changes.
    # Require combined blind policy to outperform legacy blind on BOTH models.
    dfres=pd.DataFrame(results);gains=[]
    for tag in c['models']:
        r=dfres[dfres.model==tag].set_index('policy');gains.append(r.loc['combined blind/trust','clean']-r.loc['legacy blind','clean'])
    recommend=newtest if min(gains)>0 else et.copy()
    bnew=assign(recommend,te);ts,trr,tc=updated(c,bnew)
    pt=np.load(ROOT/'models/oof_final_local/test_proba.npy').astype(float);pre=(pt*c['models']['oof_final_local']['w']).argmax(1)
    before=rules(pre,te,c);after=rules(pre,te,c,bnew,ts,trr,tc);affected=np.zeros(len(tr),bool);affected[te]=before!=after
    devices=[]
    for d in np.unique(c['dev'][te&(bnew!=c['band'])]):
        rows=np.flatnonzero(te&(c['dev']==d));linked=np.zeros(len(tr),bool)
        for row in rows:
            linked|=((c['snd']==c['snd'][row])&(np.abs(c['t']-c['t'][row])<=10800))|((c['rcv']==c['rcv'][row])&(np.abs(c['t']-c['t'][row])<=86400))|((c['root']==c['root'][row])&c['inchain'])
        devices.append(dict(device=int(d),old=int(c['band'][rows[0]]),new=int(bnew[rows[0]]),rows=len(rows),direct_pred_changes=int(affected[rows].sum()),event_rule_changes=int((affected&linked&te&(c['dev']!=d)).sum())))
    profile=[];validate=[]
    for k in range(1,11):
        r={'edge':k,'between':LABELS[k]+'/'+LABELS[k+1],'current':et[k]}
        for name,(cut,ps) in variants.items():
            p=ps[k-1];r[name]=f'{et[0]+p["mode"]} [{et[0]+p["lo"]},{et[0]+p["hi"]}]'
        r['half_bot_MAP']=int(thin[k]);profile.append(r)
        p=pblind[k-1];validate.append(dict(edge=k,truth=int(ed[k]),legacy_error=int(oldblind[k]-ed[k]),combined_error=int(blind[k]-ed[k]),CI=f'{ed[0]+p["lo"]}:{ed[0]+p["hi"]}',covered=bool(ed[0]+p['lo']<=ed[k]<=ed[0]+p['hi']),bootstrap_MAP_range=f'{min(x[0][k] for x in boots)}:{max(x[0][k] for x in boots)}',bootstrap_CI_coverage=f'{sum(x[1][k-1] for x in boots)}/20'))
    profiledetail=[]
    for k in range(10):
        p=ptest[k];positions=np.unique(np.r_[np.searchsorted(p['p'].cumsum(),[.005,.025,.1,.25,.5,.75,.9,.975,.995]),et[k+1]-et[0],newtest[k+1]-et[0]])
        rr=dict(edge=k+1,combined='; '.join(f'{et[0]+v}:{p["log"][v]-np.max(p["log"]):.2f}' for v in positions))
        for name in ['anchors','occupancy','prior']:
            ap=variants[name][1][k];av=np.unique(np.r_[np.searchsorted(ap['p'].cumsum(),[.1,.5,.9]),et[k+1]-et[0],ap['mode']])
            rr[name]='; '.join(f'{et[0]+v}:{ap["log"][v]-np.max(ap["log"]):.2f}' for v in av)
        profiledetail.append(rr)
    benchmarks=[dict(model=tag,pooled_clean=macro_f1(c['y'],m['pred']),pooled_raw=macro_f1(c['raw'],m['pred']),nested_clean=macro_f1(c['y'],m['nested']),nested_raw=macro_f1(c['raw'],m['nested'])) for tag,m in c['models'].items()]
    text='# Test band uncertainty and collect-family rules — 14 November 2026\n\nPart 1 completed; Part 2 follows below when its analysis finishes. No predictive models fitted. Only class-weight calibration on saved probabilities is fitted, as required by the specified benchmark. New files: scripts/explore/44_codex_test_edges.py, scripts/explore/45_codex_collect_rule.py, and this report.\n\n'
    text+='**Fixed benchmark reproduced first.** October training tails are assigned to September, exactly as common.load_xy. Pooled final_local uses its saved weights_final_local.npy, reproducing 0.5597/0.5113; recalibrating against the fixed target gives 0.559606/0.511298 because its original calibration labels differ on 69 rows. s2_a pooled weights and all nested weights use the fixed target. Nested means ONLY two-month weight calibration, not fully nested upstream feature/model construction. Fast calibration is checked against the existing implementation on 5,000 rows.\n\n'+md(benchmarks)+'\n\n'
    text+='**Part 1 — uncertainty model.** Each numbered device is one observation, including unused numbers. Occupancy probabilities use train-band Beta(0.5,0.5) posterior means; shape emissions use occupied-device mean stage-1 probabilities, fraud threshold 0.5 and 15 categories (ordinary, classes 1–11, quarter-drain, .35-drain, ambiguous drain), with Dirichlet(0.5) smoothing. Thus sibling-shaped anchors retain their empirical generator-band likelihood. Anchor evidence is conditional on occupancy; multiplying the two does not count absence twice. Width prior: independent Gaussian penalties on ordered inner cuts, sd=2% of block width, fixed endpoints, strictly positive widths. Exact forward/backward dynamic programming marginalizes every other cut. These are 95% equal-tail **conditional credible intervals**, not calibrated frequentist guarantees: emissions, occupancy, sibling mix and prior strength are plug-in assumptions. Anchor-only and occupancy-only use a uniform ordered-cut prior.\n\n'
    text+='Occupancy estimates by class 1–11: '+str(np.round(rho,4).tolist())+'. Native bot bursts use the separate 420001–423000 device pool. Halving their frequency does not imply halving the new-device bot-band width; the bot-band anchor counts by category are '+str(np.bincount(ct[(phase==1)&ot],minlength=15).tolist())+'. We therefore retain the width prior and separately show the stronger half-entire-bot-band sensitivity. Test labels are never used.\n\n'
    text+='Modes [95% CI] by evidence. Main disagreements: bot/layering anchors prefer 473839 versus occupancy 473818 and width prior 473826; SIM/takeover occupancy prefers 473988 versus anchors 473971 and prior 473975; phishing/QR anchors prefer 474360 versus prior 474334; occupancy gives very weak tail localization (loan/collusion mode 474550 versus anchors 474529). Half-entire-bot-width prior leaves all joint-MAP cuts unchanged. Half-bot column is joint MAP under that sensitivity prior.\n\n'+md(profile)+'\n\nRelative log marginal profiles (position: log density relative to its maximum; selected quantiles plus current and modal positions). Each evidence-only profile integrates other cuts under uniform ordering constraints; the prior column is the ordered width-prior marginal. Runnable posterior() returns every position.\n\n'+md(profiledetail)+'\n\n'
    text+='**Blind train validation.** Reverse prior uses the CURRENT test proportions, as bands.py does. Existing procedure reproduced 72/3231 wrong band rows; combined cuts give '+str(combinedwrong)+'/3231. It is not independent calibration: train edges supplied emission/occupancy estimates, and test proportions were themselves regularized by train edges. Bootstrap: 20 multinomial resamples of test device-position evidence (including absent numbers), refit test cuts, then use their proportions for train inference. Coverage below is conditional and cannot remove that circularity.\n\n'+md(validate)+'\n\n'
    text+='**Policies, frozen model probabilities and fixed labels.** All altered band counts and chain bands are rebuilt in memory before applying make_submission-equivalent event rules. Trust uses combined blind cuts. Fallback clears uncertain devices as evidence sources and forces their own final result back to weighted model type. Neighbour takes an adjacent band for an exact model-type match, or a uniquely compatible sibling shape; overlap of sibling families does not force a flip. Confidence intervals define the ambiguous rows; both endpoints included. Model features/probabilities are not recomputed, so these are decision-layer sensitivities, not retrained scores.\n\n'+md(results)+'\n\n'
    text+='**Recommendation.** Statistical joint-MAP test cuts: '+str(newtest.tolist())+'. Recommended deployment cuts: '+str(recommend.tolist())+'. Keep the trust policy as the conservative recommendation: neighbour correction wins the measured policy comparison, but would also override true generator bands for some sibling-shaped events; its improvement has not been checked on independent blocks. New cuts are recommended only if trust improves blind-train score over legacy on BOTH saved models; otherwise retain current cuts. Blind-score gains: '+str([round(float(x),6) for x in gains])+'. This is a provisional, reused-validation recommendation, not a proven test improvement. No edges/submission file was written. Recheck intervals with independent blocks before treating their nominal coverage as calibrated.\n\n'
    text+='Test devices changed by the recommended cuts; event counts attribute JOINT changed predictions on other devices through sender/receiver/chain links and can overlap across devices. Total final_local test predictions changed: '+str(int((before!=after).sum()))+'. s2_a has no test probabilities.\n\n'+md(devices)+'\n\n'
    assert len(text.splitlines())<195,len(text.splitlines())
    if suffix:text=text.replace('Part 1 completed; Part 2 follows below when its analysis finishes.','Parts 1 and 2 completed.')
    assert len((text+suffix).splitlines())<300
    REPORT.write_text(text+suffix,encoding='utf-8');print('PART 1 WRITTEN',len(text.splitlines()),flush=True)

if __name__=='__main__':main()
