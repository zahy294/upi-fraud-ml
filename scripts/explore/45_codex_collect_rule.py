"""No-fit audit and exhaustive collect-family rule scan. Only writes the new report.

Run after 44_codex_test_edges.py; both scripts may regenerate their own report section.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','ARROW_NUM_THREADS'):
    os.environ[_key]='4'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import importlib.util
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('edge_audit',Path(__file__).with_name('44_codex_test_edges.py'))
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
np,pd=A.np,A.pd
FAMILY=[6,9,10]
MONTHS=['2026-07','2026-08','2026-09']
MARKER='**Part 2 - collect-family audit.**'


def lookup(q,p,mask):
    """Number of masked donor rows at query key; namespaces already separated."""
    valid=mask&(p>=0)
    size=max(int(p[valid].max(initial=0)),int(q.max(initial=0)))+1
    out=np.bincount(p[valid],minlength=size)[np.maximum(q,0)]
    return np.where(q>=0,out,0)


def label_counts(q,p,flag,tr,y,mc,include_legit=False):
    out=np.zeros((len(q),12),np.int32)
    for k in range(0 if include_legit else 1,12):
        mask=flag&tr&(y==k)&(p>=0)
        total=lookup(q,p,mask)
        same=lookup(np.where(q>=0,q*4+mc,-1),np.where(p>=0,p*4+mc,-1),mask)
        out[:,k]=total-same
    return out


def apply_vote(old,flag,counts,m,s):
    total=counts.sum(1);top=counts.argmax(1);most=counts.max(1)
    unique=(counts==most[:,None]).sum(1)==1
    mask=flag&np.isin(old,FAMILY)&np.isin(top,FAMILY)&unique&(total>=m)&(most>=s*total)
    new=old.copy();new[mask]=top[mask]
    return new


def scores(y,z,months):
    return np.array([A.macro_f1(y,z)]+[A.macro_f1(y[months==mo],z[months==mo]) for mo in MONTHS])


def score_text(v):
    return '/'.join(f'{x:.6f}' for x in v)


def moves(y,old,new,months):
    rows=[];change=old!=new
    for sel in [np.ones(len(y),bool)]+[months==mo for mo in MONTHS]:
        m=change&sel
        rows.append('/'.join(str(int(x)) for x in [(m&(new==y)).sum(),(m&(old==y)).sum(),(m&(old!=y)&(new!=y)).sum()]))
    return '; '.join(rows)


def quantiles(a):
    return '/'.join(f'{v:.2f}' for v in np.quantile(a,[.5,.9,.99])) if len(a) else 'NA'


def main():
    if not A.REPORT.exists():raise FileNotFoundError('Run 44_codex_test_edges.py first')
    c=A.context();tr=c['tr'];n=len(tr);mc=np.full(n,3,int)
    for j,mo in enumerate(MONTHS):mc[tr&(c['month']==mo)]=j
    mt=c['month'][tr];y=np.full(n,-1,int);y[tr]=c['y']
    prob=np.zeros((n,12));prob[tr]=c['models']['oof_final_local']['P']
    prob[~tr]=np.load(ROOT/'models/oof_final_local/test_proba.npy')
    flag=(1-prob[:,0]>=.5)|(c['band']>0)
    ev=prob.copy();ev[c['band']>0]=np.eye(12)[c['band'][c['band']>0]]
    prediction=np.zeros(n,int);prediction[tr]=c['models']['oof_final_local']['pred']
    prediction[~tr]=A.rules((prob[~tr]*c['models']['oof_final_local']['w']).argmax(1),~tr,c)
    collect=flag&np.isin(prediction,FAMILY)
    rcv_account=np.where(c['p2p'],c['rn'],-1)
    links={'snd':(c['snd'],c['snd'],True),'rcv':(c['rcv'],c['rcv'],True),
           'dev':(c['dev'],c['dev'],True),'s2r':(c['snd'],rcv_account,False),'r2s':(rcv_account,c['snd'],False)}
    cover=[];leak=[];donors={};self_transfers=int((flag&c['p2p']&(c['snd']==c['rn'])).sum())
    for name,(q,p,has_self) in links.items():
        print('Audit link',name,flush=True)
        columns=['txn_id',f'x_{name}_near_n',f'x_{name}_far_n']+[f'x_{name}_lab{k}' for k in range(1,12)]+[f'x_{name}_far_p{k}' for k in range(1,12)]
        f=pd.read_parquet(ROOT/'data/processed/features_v6.parquet',columns=columns)
        assert np.array_equal(f.txn_id.values,c['df'].txn_id.values)
        lab=label_counts(q,p,flag,tr,y,mc)
        stored=f[[f'x_{name}_lab{k}' for k in range(1,12)]].to_numpy()
        diff=lab[:,1:]-stored
        count_near=A.Events(p,c['t'],mask=flag&(p>=0)).count(q,c['t'],-86400,86400)
        count_all=lookup(q,p,flag)
        near=count_near-(flag&has_self)
        far=count_all-count_near
        near_error=float(np.abs(near-f[f'x_{name}_near_n'].values).max())
        far_error=float(np.abs(far-f[f'x_{name}_far_n'].values).max())
        lt=lab.sum(1);fp=f[[f'x_{name}_far_p{k}' for k in range(1,12)]].to_numpy().sum(1)
        probability_error=0.
        valid=flag&(p>=0)
        for k in range(1,12):
            sums=np.bincount(p[valid],weights=ev[valid,k],minlength=max(int(p.max()),int(q.max()))+1)
            total=np.where(q[collect]>=0,sums[np.maximum(q[collect],0)],0)
            near_mass=A.Events(p,c['t'],ev[:,k],mask=valid).sum(q[collect],c['t'][collect],-86400,86400)
            probability_error=max(probability_error,float(np.abs(total-near_mass-f[f'x_{name}_far_p{k}'].values[collect]).max()))
        for month in range(4):
            sel=collect&(mc==month);nd=2 if month<3 else 3
            entry=dict(month=MONTHS[month] if month<3 else 'test',link=name,n=int(sel.sum()),
                lab_coverage=float((lt[sel]>0).mean()),lab_q=quantiles(lt[sel]),lab_mean_per_donor_month=float(lt[sel].mean()/nd),
                far_coverage=float((far[sel]>0).mean()),far_n_q=quantiles(far[sel]),far_prob_q=quantiles(fp[sel]))
            for k in FAMILY:
                entry[f'lab{k}_q']=quantiles(lab[sel,k]);entry[f'far_p{k}_q']=quantiles(f.loc[sel,f'x_{name}_far_p{k}'].values)
            cover.append(entry)
        training_changed=[];val_unsafe=[];val_band=[];same_label_perturb=[]
        for h in range(3):
            held_donors=flag&tr&(mc==h)&(y>0)
            held_counts=lookup(q,p,held_donors)
            training_changed.append(int((tr&(mc!=h)&(held_counts>0)).sum()))
            # Algebraic finite difference: replacing every own-month label changes total
            # and own-month subtotal identically, with detector/bands held fixed.
            delta=lookup(q,p,held_donors)-lookup(np.where(q>=0,q*4+mc,-1),np.where(p>=0,p*4+mc,-1),held_donors)
            same_label_perturb.append(int(np.count_nonzero(delta[tr&(mc==h)])))
            unsafe=flag&(p>=0)&((~tr)|(mc!=h))&(c['band']==0)
            ue=A.Events(p,c['t'],mask=unsafe)
            qq=collect&tr&(mc==h)
            unear=ue.count(q[qq],c['t'][qq],-86400,86400)
            uall=lookup(q[qq],p,unsafe)
            val_unsafe.append(f'{int((uall>0).sum())}/{int(((uall-unear)>0).sum())}')
            localband=flag&tr&(mc==h)&(c['band']>0)&(p>=0)
            bc=lookup(q,p,localband)-(localband&has_self)
            val_band.append(int((qq&(bc>0)).sum()))
        leak.append(dict(link=name,lab_mismatch_rows=int((diff!=0).any(1).sum()),lab_max_error=float(np.abs(diff).max()),near_far_count_error=f'{near_error:g}/{far_error:g}',far_probability_max_error=f'{probability_error:.3g}',
            own_month_perturb='/'.join(map(str,same_label_perturb)),outer_train_changed='/'.join(map(str,training_changed)),
            collect_unsafe_any_far='; '.join(val_unsafe),collect_ownmonth_band='/'.join(map(str,val_band))))
        if name in ['snd','dev']:
            lab[:,0]=label_counts(q,p,flag,tr,y,mc,True)[:,0]
            donors[name]=lab
        del f,stored,diff
    density=[]
    for j in range(4):
        sel=mc==j
        density.append(dict(month=MONTHS[j] if j<3 else 'test',rows=int(sel.sum()),detected=int((sel&flag).sum()),collect=int((sel&collect).sum()),detected_clean_legit=int((sel&flag&(y==0)).sum()) if j<3 else 'NA'))
    baseline=[];scan=[];best={}
    for tag,model in c['models'].items():
        cache={}
        for mode in ['pred','nested']:
            old=model[mode]
            cache[mode]=(scores(c['y'],old,mt),scores(c['raw'],old,mt))
            baseline.append(dict(model=tag,mode='pooled' if mode=='pred' else mode,clean=score_text(cache[mode][0]),raw=score_text(cache[mode][1])))
        choices=[]
        for link in ['snd','dev']:
            cnt=donors[link][tr]
            for m in [1,2,3]:
                for s in [.6,.8,1.]:
                    row=dict(model=tag,link=link,m=m,s=s);result={}
                    for mode in ['pred','nested']:
                        old=model[mode];new=apply_vote(old,flag[tr],cnt,m,s)
                        sc=scores(c['y'],new,mt);sr=scores(c['raw'],new,mt)
                        short='P' if mode=='pred' else 'N'
                        row[f'{short}_clean']=score_text(sc);row[f'{short}_raw']=score_text(sr)
                        row[f'{short}_moves_clean']=moves(c['y'],old,new,mt);row[f'{short}_moves_raw']=moves(c['raw'],old,new,mt)
                        result[mode]=dict(clean=sc,raw=sr,gain=sc-cache[mode][0],new=new)
                    gain=result['nested']['gain'];eligible=(gain[0]>=.001)&(gain[1:].min()>=0)
                    row['keep']=bool(eligible);scan.append(row)
                    choices.append(dict(link=link,m=m,s=s,gain=gain,eligible=bool(eligible),scores=result['nested']['clean'],
                        band_changes=int(((result['nested']['new']!=model['nested'])&(c['band'][tr]>0)).sum())))
        allowed=[x for x in choices if x['eligible']]
        best[tag]=max(allowed or choices,key=lambda z:z['gain'][0])
        print('BEST',tag,best[tag],flush=True)
    transitions=[];verdict=[]
    for tag,z in best.items():
        verdict.append(dict(model=tag,setting=f'{z["link"]},m={z["m"]},s={z["s"]}',nested_clean=score_text(z['scores']),nested_gain=score_text(z['gain']),band_rows_changed=z['band_changes'],verdict='KEEP' if z['eligible'] else 'DROP'))
        if tag!='oof_final_local':continue
        old=prediction[~tr];new=apply_vote(old,flag[~tr],donors[z['link']][~tr],z['m'],z['s'])
        changed=old!=new
        for a,b in sorted(set(zip(old[changed],new[changed]))):
            transitions.append(dict(model=tag,from_type=A.LABELS[a],to_type=A.LABELS[b],rows=int(((old==a)&(new==b)).sum())))
    text=MARKER+'\n\n'
    text+='**Leak review (source and no-fit measurements).** `src/upi/features_v6.py:85-92` subtracts the query month from label counts, so its direct label lookup excludes its own label and ALL same-month labels. Reconstructing all 55 label columns below checks the saved table. Holding the detector, bands and donor labels outside the query month fixed, arbitrary replacement of every same-month label has exactly zero direct effect. This is conditional safety, not end-to-end safety. `features_v6.py:46-51` admits foreign-month OOF probabilities from models trained on the query month and test probabilities trained on every month; these affect both flag selection and near/far evidence. Own-month bands also encode label-derived train edges (`bands.py:75-85`), and cleaned donor labels come from a global event vote (`clean_labels.py:62-89`), so indirect label paths remain.\n\n'
    text+='**More serious outer-fold leak:** `run_oof.py:38` loads one global feature table before splitting at lines 65-67. For held-out July, an August training row may use a July donor label even though the July validation row excludes July donors. Removing held-out labels from ALL outer-training rows changes the counts shown below. This can inflate the second-stage gain; nested class weights do not fix it. No-fit test performed: mask each held-out month from label donors and count changed training rows; separately count collect queries exposed to foreign/test probability donors, with band donors separated. Recompute/freeze donor features per OUTER fold and inner-cross-fit upstream predictions for a clean future evaluation. A new fit would be required to estimate the score inflation; these exposure counts cannot quantify it. A stored final predictor, if available, could be scored on scrubbed validation features without fitting as a sensitivity test, but saved OOF arrays alone cannot do that.\n\n'
    text+='Leak table: slash triples are July/August/September. `outer_train_changed` counts available training rows BEFORE legit downsampling with at least one held-out cleaned-fraud-label donor; `collect_unsafe_any_far` is collect-query count with foreign/test non-band probability donors at any distance / >24h; own-month band column excludes self for same-role links. All 55 far-probability columns are reconstructed on collect queries, with maximum float32 rounding error reported. Count reconstruction checks near/far endpoints and self subtraction. **Self-exclusion bug:** '+str(self_transfers)+' flagged self-transfer retains its own near evidence in s2r/r2s because `has_self=False` at lines 66/72; replace the constant with `me = flag & (qkey == pkey) & (qkey >= 0)`. Direct label counts still exclude its month.\n\n'+A.md(leak)+'\n\n'
    text+='**Train/test support.** Selection: final_local fraud probability >=0.5 OR band>0, AND its band-adjusted weighted prediction in {6,9,10}. This common detector is used for BOTH rule baselines, as for features_v6; no threshold fitted. Month tails use the benchmark September assignment. Counts:\n\n'+A.md(density)+'\n\n'
    text+='Link distributions include zero-support rows. q fields are median/p90/p99; far_prob_q sums all eleven fraud-probability columns (fraud mass, not an average), followed by individual collect-family label/probability column quantiles. Label counts here follow v6 and exclude cleaned-legit donors. Mean label count per donor month divides train by 2 and test by 3. Far columns use all four months and exclude <=24h, whereas labels use only other training months at ANY distance: they are not comparable exposure windows. October has roughly half the rows but an extra labeled donor month. Nonetheless sender label coverage falls from 37.5-39.6% to 22.7%, device coverage from about 32% to 17.2%; sender count per donor month falls from 0.260-0.297 to 0.221. Receiver coverage instead rises to 43.3% from 34.9-36.9%. Thus a single density correction is inadequate; include donor exposure, count and share separately.\n\n'+A.md(cover)+'\n\n'
    text+='**Rule definition and full scan.** Start from each model after the existing band rules. A query must pass the COMMON detector above and currently predict 6, 9 or 10. Count all detected TRAIN donor rows on the same sender (snd) or device (dev), excluding the query training month; test uses all three months. Include cleaned label 0 in the denominator, unlike v6. T must be the unique modal label, T in {6,9,10}, total donor rows >=m and votes(T)/total >=s; ties abstain. Then replace the prediction with T, including a band row if it qualifies (band changes reported). No raw label, trap or test label enters the rule. No probability/weight recalibration after the rule. This is cross-month voting using globally cleaned labels and already-generated OOF detectors; it inherits their indirect paths, so it is NOT a fully nested feature/model evaluation.\n\n'
    text+='Baselines and scan score cells are pooled/July/August/September macro-F1. P=pooled weight calibration; N=two-month weight calibration applied to the third. Move cells have four groups in that same order, each R/W/X = wrong-to-right / right-to-wrong / wrong-to-wrong. Raw and cleaned assessments share predictions. Scanned 18 settings per model (36 model-settings); selecting the best on these same months adds selection optimism. No significance claim.\n\n'+A.md(baseline)+'\n\n'+A.md(scan)+'\n\n'
    text+='**Verdict.** Select the highest nested pooled gain among settings meeting BOTH gain >=0.001 and no negative cleaned-label month; if none qualify, show the highest-gain rejected setting. The winning final_local sender rule changes 155 train rows: 89 corrections, 57 regressions and 9 wrong-to-wrong under cleaned labels (raw: 89/39/27). It gains 0.001722 nested, not 0.010; almost all its useful support is single-donor history, and demanding >=2 donors makes the sender rule harmful. On s2_a the same unanimous rule corrects 18 but damages 34 nested rows, with 1 wrong-to-wrong; all positive-change settings fail. The best rejected settings make no changes. This supports keeping the explicit rule only on final_local under the requested benchmark, not adding it to s2_a or interpreting s2_a as a leakage-free improvement.\n\n'+A.md(verdict)+'\n\n'
    text+='Test changes for the selected final_local setting (counterfactual if verdict is DROP), with current edges/band rules unchanged to isolate the history rule:\n\n'+A.md(transitions)+'\n\n'
    text+='`models/s2_a` contains OOF probabilities only: its selected rule has **no computable test transition count** without test probabilities. Transferring final_local predictions would not be an s2_a result. No model fitted and no submission written.\n\n'
    text+='**Reproduce.** Run `C:/Users/ASUS/AppData/Local/Programs/Python/Python311/python.exe -I scripts/explore/44_codex_test_edges.py` then the same executable with `-I scripts/explore/45_codex_collect_rule.py`. Both cap numerical threads at 4 and write only their sections of this new report. Source artifacts read: base_v0, features_v1/v3/v5/v6, band_edges.json, s1 probabilities, final_local probabilities/saved weights, s2_a OOF, and oof_raw/y_clean.npy. `44` also exposes the complete discrete posterior arrays in `posterior()`. No files under data/, models/ or submissions/ are written.\n'
    before=A.REPORT.read_text(encoding='utf-8').split(MARKER,1)[0]
    before=before.replace('Part 1 completed; Part 2 follows below when its analysis finishes.','Parts 1 and 2 completed.')
    result=before+text
    assert len(result.splitlines())<300,len(result.splitlines())
    A.REPORT.write_text(result,encoding='utf-8')
    print('REPORT COMPLETE',len(result.splitlines()),flush=True)


if __name__=='__main__':main()
