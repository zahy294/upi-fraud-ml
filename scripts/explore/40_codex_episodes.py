"""Read-only episode analysis; writes ONLY docs/codex_episodes.md.
Run with Python -I. No fitting. Four numerical threads maximum.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','ARROW_NUM_THREADS'):
    os.environ[k]='4'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast
import numpy as np
import pandas as pd
import pyarrow as pa
pa.set_cpu_count(4);pa.set_io_thread_count(4)
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/upi'))
from common import LABELS
OUT=ROOT/'docs/codex_episodes.md'
SUMMARY = '''**Findings from this run.** Sender identity carries useful cross-month type information for the collect family, especially cashback and loan events emitted in collect shape. Among family-labelled donor/query rows, >30-day same-sender raw-label agreement is 89.77%, 90.13%, 92.37% in July/August/September, against degree-matched chance 65.17%, 72.55%, 65.33%. At 7–30 days it is 91.78%, 89.58%, 91.18% against 65.21%, 69.93%, 73.70%. These are descriptive results from the declared search, not significance tests.

For cashback→phishing mistakes, same-sender >30-day links reach 53/166 reporting events: 30 correct votes, 10 wrong, 13 abstentions. Monthly correct/wrong/abstain counts are July 12/5/6, August 5/2/3, September 13/3/4. At 7–30 days: 31 linked, 22 correct, 3 wrong, 6 abstentions. Device history largely duplicates this evidence, so do not add sender and device counts as independent recoveries.

For loan→phishing mistakes, same-sender >30-day links reach 23/71 events: 17 correct, 2 wrong, 4 abstentions. Monthly counts are July 7/1/1, August 5/0/1, September 5/1/2. The 7–30-day result is 8 correct, 0 wrong, 2 abstentions, but September has no coverage in that bin. Persistent sender type beyond 30 days is therefore a concrete feature candidate, with limited coverage and nonzero errors; it is not a universal type override.

Negative or misleading cases: >30-day same-sender voting on phishing→cashback gives only 10 correct, 44 wrong, 18 abstentions among 72 linked events; on mule→layering, 2/58/6 among 66; on bot→mule, 0/13/2 among 15. Takeover→SIM has only one such sender link and its vote is wrong; receiver links reach 104/158 events but give 14 correct, 62 wrong, 28 abstentions. Receiver-pool reuse does not establish shared type.

The broad two-hop graph produces 100,529,440 other-month, distinct-event row links beyond one day. Its apparently promising mule→layering >30-day voting (176 correct, 57 wrong, 36 abstentions) is a class-prevalence effect rather than evidence of type assortativity: family raw-label agreement/chance is July 52.68%/52.75%, August 52.86%/52.82%, September 53.15%/53.31%. Wrong-only voting accuracy cannot establish usefulness on correctly typed controls. No positive two-hop feature recommendation follows from that number.

**Low-score episode detection is negative.** Same-sender ANY-distance raw-fraud rates are July 786/33,364 = 2.36%, August 760/34,558 = 2.20%, September 815/33,625 = 2.42%. Same-receiver rates are 4,101/183,727 = 2.23%, 4,112/178,861 = 2.30%, 3,935/169,183 = 2.33%. These remain near the 2.30% reference. Pair ANY-distance samples contain only 72, 62, 69 rows. None of the 45 cells passes both >6% and >=300 rows, even in a single month. No additional stealth detector was found.

The full histories below are intentionally large: 30 examples contain 509,572 row appearances, including repetitions where account histories overlap. CSV code blocks retain every history row while avoiding fixed-width padding. Read the summary/tables first; the approximately 65 MB file is chiefly the explicitly requested complete histories.'''
if OUT.exists(): raise FileExistsError('Refusing to replace an existing report: '+str(OUT))
sys.stdout.reconfigure(encoding='utf-8',line_buffering=True)
df=pd.read_parquet(ROOT/'data/processed/base_v0.parquet')
n=len(df); tr=df.split.eq('train').to_numpy(); ti=np.flatnonzero(tr)
t=df.t.to_numpy(np.int64); month=df.month.astype(str).to_numpy()
# Calendar months, including the October-dated train tails as a separate donor month.
y=df.label.map(dict(zip(LABELS,range(12)))).fillna(-1).to_numpy(int)
yc=np.full(n,-1);yc[tr]=np.load(ROOT/'models/oof_raw/y_clean.npy')
P=np.empty((n,12),np.float32)
P[tr]=np.load(ROOT/'models/oof_final_local/oof.npy')
P[~tr]=np.load(ROOT/'models/oof_final_local/test_proba.npy')
pf=1-P[:,0].astype(float)
F3=pd.read_parquet(ROOT/'data/processed/features_v3.parquet');band=F3.band.to_numpy(int)
F5=pd.read_parquet(ROOT/'data/processed/features_v5.parquet',columns=['ch_band'])
p2p_all=df.receiver_type.eq('P2P').to_numpy()
ns=dict(np=np,F3=F3,F5=F5,band=band,p2p_all=p2p_all)
tree=ast.parse((ROOT/'src/upi/make_submission.py').read_text())
exec(compile(ast.Module(body=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='band_rules'],type_ignores=[]),'rules','exec'),ns)
w=np.load(ROOT/'models/oof_final_local/weights_final_local.npy')
pred=ns['band_rules']((P.astype(float)*w).argmax(1),np.arange(n))[0]
det=(pf>=.5)|(band>0); di=np.flatnonzero(det&tr)
snd=df.sender_id.to_numpy();rcv=df.receiver_id.to_numpy();dev=df.device_id.to_numpy()
pair=pd.factorize(pd.MultiIndex.from_arrays([snd,rcv]))[0]
# Reporting events: detected sender sessions, not label-defined components.
z=pd.DataFrame({'row':np.flatnonzero(det),'snd':snd[det],'t':t[det]}).sort_values(['snd','t','row'])
z['event']=((z.snd!=z.snd.shift())|(z.t.diff()>86400)).cumsum().to_numpy()
ev=np.full(n,-1);ev[z.row]=z.event
eventrows=z.groupby('event').row.apply(lambda x:x.to_numpy()).to_dict()
families=[(5,4,[4,5]),(1,3,[1,3]),(6,9,[6,9,10]),(2,1,[1,2]),(9,6,[6,9,10]),(10,6,[6,9,10])]
months=['2026-07','2026-08','2026-09'];DAY=86400
roles=['sender','receiver','sender_as_receiver','receiver_as_sender','device','pair','two_hop']
keymap={'sender':(snd,snd),'receiver':(rcv,rcv),'sender_as_receiver':(snd,rcv),'receiver_as_sender':(rcv,snd),'device':(dev,dev),'pair':(pair,pair)}
def edges(role,query):
    if role!='two_hop':
        a,b=keymap[role]
        return pd.DataFrame({'q':query,'key':a[query]}).merge(pd.DataFrame({'j':di,'key':b[di]}),on='key')[['q','j']].drop_duplicates()
    # Full four-month unlabeled topology. A != B, A->R and B->R at any times.
    # Both accounts must have detected rows to enter the query/donor join.
    arcs=df[['sender_id','receiver_id']].drop_duplicates()
    qa=arcs[arcs.sender_id.isin(snd[query])].rename(columns={'sender_id':'a','receiver_id':'r'})
    sa=arcs[arcs.sender_id.isin(snd[di])].rename(columns={'sender_id':'b','receiver_id':'r'})
    peers=qa.merge(sa,on='r');peers=peers[peers.a!=peers.b][['a','b']].drop_duplicates()
    return pd.DataFrame({'q':query,'a':snd[query]}).merge(peers,on='a').merge(pd.DataFrame({'j':di,'b':snd[di]}),on='b')[['q','j']].drop_duplicates()
def restricted(e):
    q=e.q.to_numpy();j=e.j.to_numpy()
    good=(month[q]!=month[j])&(ev[q]!=ev[j])
    e=e.loc[good].copy();e['days']=np.abs(t[e.q]-t[e.j])/DAY
    return e
def distmask(d,dist):
    return {'1-7d':(d>1)&(d<=7),'7-30d':(d>7)&(d<=30),'>30d':d>30,'0-1d':d<=1,'ANY':np.ones(len(d),bool)}[dist]
def table(frame):
    if frame.empty:return '(no rows)\n'
    # Avoid optional tabulate dependency.
    cols=list(frame.columns);out=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for row in frame.itertuples(index=False,name=None):
        out.append('| '+' | '.join('—' if pd.isna(v) else (f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v)) for v in row)+' |')
    return '\n'.join(out)+'\n'
records=[];wrongrec=[];example_links={};allwrong={};allroleedges={}
for a,b,fam in families:
    label=LABELS[a]+' → '+LABELS[b]
    allwrong[label]=np.flatnonzero(tr&det&(yc==a)&(pred==b)&np.isin(month,months))
for role in roles:
    print('role',role,flush=True)
    e=restricted(edges(role,di));e=e[e.days>1]
    allroleedges[role]=e
    for a,b,fam in families:
        label=LABELS[a]+' → '+LABELS[b]
        # Family baseline requires raw and predicted query types both in family.
        pool=det&tr&np.isin(y,fam)&np.isin(pred,fam)
        for m in months:
            qids=np.flatnonzero(pool&(month==m));wrong=allwrong[label];wrong=wrong[month[wrong]==m]
            for dist in ['1-7d','7-30d','>30d']:
                ee=e[distmask(e.days.to_numpy(),dist)]
                ep=ee[ee.q.isin(qids)];qq=ep.q.to_numpy();jj=ep.j.to_numpy()
                # Independence chance preserving linked endpoint class marginals.
                chance=float(np.dot(np.bincount(y[qq],minlength=12),np.bincount(y[jj],minlength=12))/len(ep)**2) if len(ep) else np.nan
                same=float((y[qq]==y[jj]).mean()) if len(ep) else np.nan
                ef=ep[np.isin(y[ep.j],fam)];fq=ef.q.to_numpy();fj=ef.j.to_numpy()
                fc=float(np.dot(np.bincount(y[fq],minlength=12),np.bincount(y[fj],minlength=12))/len(ef)**2) if len(ef) else np.nan
                records.append(dict(confusion=label,month=m,role=role,distance=dist,query_rows=len(qids),linked_rows=ep.q.nunique(),event_pairs=len(ep.assign(qe=ev[qq],je=ev[jj]).drop_duplicates(['qe','je'])),row_links=len(ep),same_raw=same,chance=chance,family_links=len(ef),family_same=float((y[fq]==y[fj]).mean()) if len(ef) else np.nan,family_chance=fc))
                ew=ee[ee.q.isin(wrong)].copy();wq=ew.q.to_numpy();wj=ew.j.to_numpy()
                # Unique source event votes; source event mode is computed only
                # from linked donor rows outside the target calendar month.
                votes=ew[y[wj]>0].assign(qe=ev[wq[y[wj]>0]],je=ev[wj[y[wj]>0]],lab=y[wj[y[wj]>0]])
                if len(votes):
                    counts=votes.drop_duplicates(['qe','j']).groupby(['qe','je','lab']).size().rename('n').reset_index()
                    mx=counts.groupby(['qe','je']).n.transform('max'); winners=counts[counts.n==mx]
                    winners=winners[winners.groupby(['qe','je']).lab.transform('size')==1]
                    vc=winners.groupby(['qe','lab']).size().rename('n').reset_index()
                    vm=vc.groupby('qe').n.transform('max');win=vc[vc.n==vm]
                    win=win[win.groupby('qe').lab.transform('size')==1]
                    chosen=dict(zip(win.qe,win.lab))
                else:chosen={}
                we=np.unique(ev[wrong]);linked=np.unique(ev[wq]);right=sum(v==a for v in chosen.values());bad=sum(v!=a for v in chosen.values())
                wr=wrongrec.append(dict(confusion=label,month=m,role=role,distance=dist,wrong_rows=len(wrong),wrong_events=len(we),linked_rows=ew.q.nunique(),linked_events=len(linked),event_coverage=len(linked)/len(we) if len(we) else np.nan,right_events=right,wrong_events_vote=bad,abstain_events=len(linked)-len(chosen),right_share=right/len(chosen) if chosen else np.nan,any_right_events=len(np.unique(ev[wq[y[wj]==a]])),source_rows=len(np.unique(wj)),source_row_correct=float((y[wj]==a).mean()) if len(wj) else np.nan))
                # Illustration only, not a fitted selection rule: record one link per query event.
                for qe,block in ew.groupby(ev[wq]):
                    if (label,int(qe)) not in example_links:
                        row=block.iloc[0];example_links[(label,int(qe))]=(int(row.q),int(row.j),role,float(row.days))
    print('edges',len(e),flush=True)
R=pd.DataFrame(records);W=pd.DataFrame(wrongrec)
low=(pf<.2)&tr&np.isin(month,months)
lowrec=[]
for role in ['sender','receiver','pair']:
    print('low score',role,flush=True)
    ee=edges(role,np.flatnonzero(low));q=ee.q.to_numpy();j=ee.j.to_numpy()
    ee=ee[month[q]!=month[j]].copy();ee['days']=np.abs(t[ee.q]-t[ee.j])/DAY
    for m in months:
        denominator=low&(month==m)
        for dist in ['0-1d','1-7d','7-30d','>30d','ANY']:
            ids=ee.loc[(month[ee.q]==m)&distmask(ee.days.to_numpy(),dist),'q'].unique()
            fraud=(y[ids]>0).sum();basefraud=(y[denominator]>0).sum();baselegit=(y[denominator]==0).sum()
            lowrec.append(dict(month=m,role=role,distance=dist,rows=len(ids),fraud=int(fraud),legit=int(len(ids)-fraud),fraud_rate=fraud/len(ids) if len(ids) else np.nan,base_rate=basefraud/denominator.sum(),fraud_coverage=fraud/basefraud,legit_coverage=(len(ids)-fraud)/baselegit,band0_rows=int((band[ids]==0).sum()),band0_fraud_rate=float((y[ids[band[ids]==0]]>0).mean()) if (band[ids]==0).any() else np.nan))
L=pd.DataFrame(lowrec)
print('writing report',flush=True)
with OUT.open('x',encoding='utf-8') as out:
    def write(s):out.write(s+'\n\n')
    write('# Multi-day account episodes — 14 November 2026')
    write('Analysis only. No fitting, no test labels, no trap-based selection. Read PROBLEM_STATEMENT.md and the implementation-plan update boxes first. New files only: `scripts/explore/40_codex_episodes.py` and this report. All computed tables and all complete example histories are embedded below; no auxiliary files are written.')
    write('**Method and leakage boundary.** For a July/August/September query, every labelled donor row is TRAIN and from a different CALENDAR month. Query raw/cleaned labels are used only to define evaluation cohorts and score agreement, never to select a neighbour or cast its vote. October-dated training tails can be donors; test rows never supply labels. July can use August/September: this is cross-month analysis, not past-only deployment validation. Saved OOF probabilities and bands are given inputs; they were not regenerated with nested folds, so these measurements are descriptive associations, not a new honest OOF model score. Test probabilities are used only for detection/printing and unlabeled topology.')
    write('Detected = `1-P[:,0]>=0.5 or band>0`. Predictions = saved final_local class weights followed by the existing make_submission band rules. Cleaned targets are exactly `models/oof_raw/y_clean.npy`, as requested, even though they differ slightly from the targets used to fit final_local. No band==0 restriction is imposed. Calendar months are used, with the 147 October training tails excluded from July–September query denominators.')
    write('**Events and links.** A reporting event is a maximal session of detected rows of one sender, with consecutive gaps <=24h, built without labels over all four months. This intentionally does not merge all victims of a mule/bot event; counts are sender-session units and cannot be compared directly with prior connected-component counts. Cross-event links exclude the same reporting event and require absolute row separation >1 day. Bins are `(1,7]`, `(7,30]`, and `(30,infinity)` days. Cross-month filtering greatly reduces the first bin; this is expected, not evidence against within-month repeats. Counts use deduplicated row pairs; event-pair counts deduplicate their session IDs.')
    write('Role definitions: sender=S(q)==S(j); receiver=R(q)==R(j); sender_as_receiver=S(q)==R(j); receiver_as_sender=R(q)==S(j); device=D(q)==D(j); pair=ordered (S,R) equal. String IDs keep account/merchant namespaces separate, so cross-role comparisons naturally exclude merchants. Two-hop means DISTINCT senders A,B have both paid some same receiver R anywhere in the four-month raw topology; linked rows are detected rows by A and B. Shared receiver can be account or merchant. Bridge times are unrestricted and label-free; distance is between the two detected endpoint rows. This broad two-hop definition tests account neighbourhoods, not just simultaneous fan-in.')
    write('**Agreement and chance.** Family baseline query rows have BOTH raw and predicted type in the confusion family: drains={SIM,takeover}; mule/layer={mule,layer}; bot/mule={bot,mule}; collect={phishing,cashback,loan}. All detected donor labels including raw legit contribute to `same_raw`. Chance is the dot product of the linked endpoint raw-label marginals (degree-weighted independence); `family_*` additionally restricts donor raw type to the family. No permutation/significance test was performed. Rows and links are correlated within accounts/events. The three collect confusion blocks share a family baseline, so those repeats are not independent discoveries.')
    write('**Wrong-event votes.** Cohort = detected rows with requested cleaned true type and final prediction. A source session casts one vote: its unique modal positive raw label among the linked other-month rows. Source ties abstain. Target session prediction is the unique modal source-session vote; ties/no positive votes abstain. `right_share` conditions on non-abstaining votes; `event_coverage` includes links whose votes abstain. `any_right_events` is an optimistic availability measure, not a deployable selection rule. `source_row_correct` is the raw-label agreement across linked row pairs. This avoids letting one 60-row bot burst cast 60 independent votes.')
    write('**Search accounting.** Seven roles × three distance bins × six confusion cohorts × three months = 378 agreement cells and 378 wrong-event cells (with shared family baselines). Low-score enrichment: three roles × five distance selections × three months = 45 cells, plus the explicitly reported band-zero sensitivity. No threshold search, model fitting, significance claim, or additional stealth attribute search was done.')
    write(SUMMARY)
    write('## Observed results')
    write('Exact wrong-cohort counts under the specified artifacts/session definition:')
    write(table(pd.DataFrame([dict(confusion=k,rows=len(v),events=len(np.unique(ev[v]))) for k,v in allwrong.items()])))
    # Aggregate success/negative results without selecting thresholds.
    agg=W.groupby(['confusion','role','distance'])[['wrong_events','linked_events','right_events','wrong_events_vote','abstain_events']].sum().reset_index()
    agg['coverage']=agg.linked_events/agg.wrong_events
    agg['vote_accuracy']=agg.right_events/(agg.right_events+agg.wrong_events_vote)
    write('Pooled wrong-event summary (a session appearing in multiple months is counted in each month; consult monthly tables for stability):')
    write(table(agg))
    qualifying=L[(L.rows>=300)&(L.fraud_rate>.06)]
    stable=qualifying.groupby(['role','distance']).month.nunique()
    stable=stable[stable==3]
    write('**Low-score result:** '+('No link/distance passes >6% raw-fraud rate on >=300 rows in EACH of July, August and September.' if len(stable)==0 else 'All-month candidates: '+str(list(stable.index)))+' Monthly cells individually passing: '+str(len(qualifying))+'/45. These are raw-fraud rates, not complete-label fraud prevalence; unreported fraud remains among raw legit.')
    write('## Monthly agreement: full search')
    write(table(R))
    write('## Wrongly typed events: full monthly coverage and votes')
    write(table(W))
    write('## Low-score episode association')
    write('All query rows have fraud probability <0.2. Donor selection depends only on other-month detection, not donor label. Each query row is counted once per role/bin even if many donors match. ANY includes all distances, including <=1 day; the first bin is included to avoid silently dropping nearby cross-month events. `fraud_coverage` and `legit_coverage` compare the fractions of all low-score fraud/legit rows that have this link. Compare `fraud_rate` to the requested 0.023 reference and to the empirical monthly `base_rate`; band0 columns check that known-device fraud is not driving a result.')
    write(table(L))
    write('## Five complete examples per confusion')
    write('Illustrations are chosen deterministically: prefer wrong sender-session events with an other-month link, then earliest transaction order; fill with unlinked events if necessary. This is not a random sample or accuracy estimate. For each example, include every raw row over the full available four-month dataset in either direction for every entity at the endpoints of the query session and ONE displayed linked source session. For two-hop examples add their shared receiver(s). Do not recursively expand newly encountered counterparties, which would engulf the graph. Test labels/cleaned labels remain absent. All history rows are printed without truncation; dates in November are preserved when present as event tails. The chosen source is illustrative, not necessarily a correct or majority vote.')
    histcols=['txn_id','timestamp','split','sender_id','receiver_id','device_id','txn_type','merchant_category','amount','balance_before','band','pf','prediction','raw','clean','session']
    hist=df.copy();hist['band']=band;hist['pf']=pf.round(6);hist['prediction']=np.array(LABELS)[pred];hist['raw']=np.where(tr,np.array(LABELS)[np.maximum(y,0)],'UNLABELLED');hist['clean']=np.where(tr,np.array(LABELS)[np.maximum(yc,0)],'UNLABELLED');hist['session']=ev
    for label,ids in allwrong.items():
        events=sorted(np.unique(ev[ids]),key=lambda e:(0 if (label,int(e)) in example_links else 1,int(eventrows[e][0])))[:5]
        for ix,qe in enumerate(events,1):
            link=example_links.get((label,int(qe)));qr=eventrows[qe]
            source=eventrows[ev[link[1]]] if link else np.array([],int)
            involved=np.r_[qr,source];entities=set(snd[involved])|set(rcv[involved])
            if link and link[2]=='two_hop':
                shared=set(rcv[snd==snd[link[0]]])&set(rcv[snd==snd[link[1]]]);entities|=shared
            mask=df.sender_id.isin(entities)|df.receiver_id.isin(entities)
            write(f'### {label}, example {ix}, query session {qe}')
            write('Link: '+str(link)+' (base row indices q,j,role,distance days); entities: '+', '.join(sorted(entities))+f'; full history rows: {int(mask.sum())}.')
            write('```csv\n'+hist.loc[mask,histcols].to_csv(index=False)+'\n```')
    write('## Reproduction and limits')
    write('Run from the repo: `C:/Users/ASUS/AppData/Local/Programs/Python/Python311/python.exe -I scripts/explore/40_codex_episodes.py`. The script refuses to overwrite an existing report; use a fresh checkout/output state for an exact replay. Its only file write is this Markdown. NumPy/BLAS/NumExpr/Arrow worker pools are limited to four before imports. No model or class-weight fit is called. Account-level correlations can be real without improving predictions: receiver pools and broad two-hop links can dilute type evidence. Strong within-month repeats can disappear under cross-month-only donors; the supplied 95.3% result is not automatically comparable to this stricter design. Deployable features would require cross-fitted, event-balanced history estimates and a further held-out validation, not reusing target-month raw labels.')
print('DONE',OUT, 'bytes',OUT.stat().st_size,flush=True)
