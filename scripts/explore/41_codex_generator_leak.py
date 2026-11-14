"""Generator-order audit. Python -I; no fitting; writes only its new Markdown report.

Sparse Fisher-Yates prefixes are checked against actual APIs before the seed sweep.
Negative results are bounded candidate-family tests, not proof of no generator leak.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','ARROW_NUM_THREADS'):
    os.environ[key]='4'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import random,hashlib,zlib,uuid,itertools,time,importlib.metadata,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.utils import shuffle as sklearn_shuffle
sys.stdout.reconfigure(encoding='utf-8',line_buffering=True)
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'docs/codex_generator_leak.md'
SUMMARY = '''**Result: no second usable generation-order leak found in the tested families. Zero additional train or test rows have a recovered generation position or phase. There is therefore no new position-based label-purity result to report; counting accidental ID matches as recovered rows would be misleading.**

- Shuffle: 10,021 seeds, three independent shuffle algorithms, three row counts and both directions; 180,378 scores. Largest absolute tail-anchor rank correlation is 0.40714, versus the pre-set 0.85 screening threshold. The exact sparse check uses 4,096 final permutation positions and was verified against actual APIs. It is not a reconstruction of each entire permutation.
- ID streams: 32,421 seeded 4,096-word prefixes; largest overlap is nine IDs, versus 1.74 expected for one arbitrary prefix. After selecting the strongest prefix seed in each broad family, full-length follow-up gives 768/772/769 matches versus 775.17 expected. Only 21/22/22 matched creation-device anchors exist, with correlations −0.309/0.011/0.107: no recovered order.
- MT19937: no exact recurrence matches in file/device order; the unordered necessary-relation test gives 1,685 hits versus 1,698.97 expected, with 1,738 in a matched-size random control. This cannot exclude an MT stream with shuffled, omitted or interleaved outputs. It is not successful state reconstruction.
- Hashes: zero corresponding-row matches in the tested content recipes. Full-domain counter hashes, extra salts/layouts, seeded Faker provider recipes, packed-counter CRC32 inverses and Murmur3 inverses produce no coherent order or excess stream coverage.
- Timestamp, decimal and other numeric structure shows recipe/phase differences, not an ordering key. The exact numeric-RNG probes are deliberately limited to simple scaled/rounded uniforms; arbitrary distributions, affine offsets and skipped RNG calls are not excluded.

All 1,824,642 IDs are unique across both files, unlike unconstrained independent 32-bit draws. This indicates uniqueness enforcement or an injective construction, but orders **zero** rows; it is not a fraud feature. A leaderboard jump alone does not establish that a second leak exists.

**Positive controls passed:** synthetic shuffled monotone device IDs recover rho=1 in both directions for legacy NumPy, Generator and Python shuffles; untempering a known MT19937 stream satisfies all 9,376 tested recurrences. Environment: Python 3.11.0, NumPy 2.4.6, pandas 2.3.3, SciPy 1.16.3, scikit-learn 1.7.2. No models were fitted. The report lists each test family, candidate range, result and important exclusions; it does not claim exhaustive coverage of arbitrary generators.

Created files: this report and [the runnable audit](../scripts/explore/41_codex_generator_leak.py). The code refuses to overwrite a pre-existing report; replay in a fresh output state. Main-sweep timing below excludes the additional provider, inverse-transform and partial-order follow-ups.'''
if OUT.exists():raise FileExistsError(OUT)
start=time.time();sections=[];positives=[]
def say(title,value):
    print(title,flush=True)
    sections.append('## '+title+'\n\n'+value+'\n')
def table(rows):
    d=pd.DataFrame(rows)
    if d.empty:return 'No candidates.'
    def fmt(x):return f'{x:.6g}' if isinstance(x,(float,np.floating)) else str(x)
    return '\n'.join(['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |']+['| '+' | '.join(map(fmt,r))+' |' for r in d.itertuples(index=False,name=None)])
cols=['txn_id','timestamp','sender_id','receiver_id','device_id','amount','balance_before','sim_age_hours','device_age_days']
frames={name:pd.read_csv(ROOT/f'data/raw/{name}.csv',usecols=cols+(['label'] if name=='train' else []),dtype=str) for name in ['train','test']}
devices={k:d.device_id.str[2:].to_numpy(np.int64) for k,d in frames.items()}
ids={k:np.fromiter((int(x[2:],16) for x in d.txn_id),dtype=np.uint32) for k,d in frames.items()}
allids=np.unique(np.r_[ids['train'],ids['test']]);idset=set(map(int,allids))
def hits(a):
    a=np.asarray(a,dtype=np.uint32);ix=np.searchsorted(allids,a);return (ix<len(allids))&(allids[np.minimum(ix,len(allids)-1)]==a)
seeds=sorted(set(range(10001))|{12345,54321,31337,65535,65536,123456,314159,271828,8675309,0xdeadbeef,2024,2025,2026,20260701,20260801,20260901,20261001,20261113,20261114,20261115,13112026,14112026,15112026})
smallseeds=sorted(set(range(101))|set(seeds).difference(range(10001))|{123,321,777,999,1024,1234,1337,1729,2048,4096})
say('Scope and inputs',f'Train {len(frames["train"]):,}; test {len(frames["test"]):,}; unique combined 32-bit IDs {len(allids):,}. Shuffles use {len(seeds)} seeds: 0–10000 inclusive plus '+str(sorted(set(seeds)-set(range(10001))))+'. Labels never select shuffle/stream candidates. DV420001–423000 is a reused bot pool, not a creation counter, so monotonicity anchors are DV>423000. Both shuffle directions are checked. Known-device ordering is a sparse constraint, not the original unshuffled row index. Repeated-device rows and interleaved RNG calls weaken it. No row-level stealth classifier is fitted.')

def random_indices(n,k,s,api):
    bits=n.bit_length();assert n-k>2**(bits-1)
    size=4*k+64;mask=(1<<bits)-1
    if api=='RandomState':raw=np.random.RandomState(s).randint(0,2**32,size=size,dtype=np.uint32).astype(np.uint64)&mask
    elif api=='Generator':raw=np.random.default_rng(s).bit_generator.random_raw((size+1)//2).view(np.uint32)[:size].astype(np.uint64)&mask
    else:
        r=random.Random(s);raw=np.fromiter((r.getrandbits(bits) for _ in range(size)),dtype=np.uint64)
    accepted=raw<n
    for _ in range(100):
        before=np.cumsum(accepted,dtype=np.int64)-accepted;new=raw<=n-1-before
        if np.array_equal(new,accepted):break
        accepted=new
    else:raise RuntimeError('rejection-sampling fixed point failed')
    assert accepted.sum()>=k
    return raw[accepted][:k].astype(np.int64)
def prefix(n,k,s,api):
    state={};out=np.empty(k,np.int64)
    for z,j in enumerate(random_indices(n,k,s,api)):
        i=n-1-z;j=int(j);out[z]=state.get(j,j);state[j]=state.get(i,i);state.pop(i,None)
    return out
def fullperm(n,s,api):
    if api=='RandomState':return np.random.RandomState(s).permutation(n)
    if api=='Generator':return np.random.default_rng(s).permutation(n)
    p=list(range(n));random.Random(s).shuffle(p);return np.array(p)
def corr(x,y):
    if len(x)<20:return np.nan
    return float(np.corrcoef(np.argsort(np.argsort(x)),np.argsort(np.argsort(y)))[0,1])
sizes=[len(frames['train']),len(frames['test']),sum(map(len,frames.values()))];validation=[]
for n in sizes:
    for seed in [0,42,10000]:
        rs=fullperm(n,seed,'RandomState');ge=fullperm(n,seed,'Generator')
        assert np.array_equal(pd.Series(np.arange(n)).sample(frac=1,random_state=seed).to_numpy(),rs)
        assert np.array_equal(sklearn_shuffle(np.arange(n),random_state=seed),rs)
        a=np.arange(n);np.random.default_rng(seed).shuffle(a);assert np.array_equal(a,ge)
        assert np.array_equal(np.random.default_rng(seed).permuted(np.arange(n)),ge)
        for api in ['RandomState','Generator','Python']:
            expected=rs if api=='RandomState' else ge if api=='Generator' else fullperm(n,seed,api)
            assert np.array_equal(prefix(n,4096,seed,api),expected[-1:-4097:-1])
            validation.append(dict(n=n,seed=seed,api=api,exact=True))
say('Sparse shuffle validation',table(validation)+'\n\nOn these versions pandas.sample(integer random_state) and sklearn.utils.shuffle equal RandomState.permutation; Generator permutation/shuffle/permuted on a 1-D row index are identical. Exact prefix validation covers all three lengths and three seeds. This tests fresh seeded RNGs, not arbitrary prior RNG consumption. The prefix returns the last 4096 final permutation positions exactly, using rejection-sampled Fisher–Yates swaps. No sampling approximation is used for those positions.')
for api in ['RandomState','Generator','Python']:
    n=260582;perm=fullperm(n,2026,api);v=prefix(n,4096,2026,api);tail=np.arange(n-1,n-4097,-1)
    original=np.arange(n)+423001;forward=original[perm];inverse=np.empty(n,int);inverse[perm]=original
    assert corr(v,forward[tail])>.999999 and corr(tail,inverse[v])>.999999
top=[];candidate=[]
for size_name,dd in [('train',devices['train']),('test',devices['test']),('train+test',np.r_[devices['train'],devices['test']])]:
    n=len(dd);tail=np.arange(n-1,n-4097,-1)
    for api in ['RandomState','Generator','Python']:
        best=[]
        for seed in seeds:
            v=prefix(n,4096,seed,api)
            for direction,x,vals in [('file=original[perm]',v,dd[tail]),('file[perm]=original',tail,dd[v])]:
                anchor=vals>423000;c=corr(x[anchor],vals[anchor])
                if len(best)<3 or abs(c)>best[-1]['abs_rho']:
                    best.append(dict(file=size_name,api=api,seed=seed,direction=direction,anchors=int(anchor.sum()),rho=c,abs_rho=abs(c)))
                    best=sorted(best,key=lambda r:r['abs_rho'],reverse=True)[:3]
                if abs(c)>.85:candidate.append((size_name,api,seed,direction,c))
        top+=best;print('shuffle done',size_name,api,flush=True)
say('Shuffle sweep results',f'{len(seeds)*9:,} seed/API/length candidates; both directions = {len(seeds)*18:,} scores. Positive screening criterion fixed before search: |Spearman rho|>0.85 with >=20 creation-device anchors. Strong candidates: {len(candidate)}. This rules out the tested exact fresh-state shuffles when their tail contains the creation-order signal; it does not rule out RNG burn-in, multi-stage shuffling, prior RNG use, or a different concatenation order. Combined-file test assumes raw train followed by raw test.\n\n'+table(top))
for name,api,s,direction,c in candidate:
    dn=devices[name] if name in devices else np.r_[devices['train'],devices['test']]
    perm=fullperm(len(dn),s,api);pos=perm if direction=='file=original[perm]' else np.argsort(perm)
    m=dn>423000;positives.append(dict(test='shuffle_full',file=name,api=api,seed=s,rows=len(dn),rho=corr(pos[m],dn[m])))

block=[]
for name,dn in devices.items():
    anchor=np.flatnonzero(dn>423000)
    for width in [32,64,128,256,512,1024,4096,16384,65536]:
        for off in [0,width//2,width-1]:
            group=(anchor+off)//width;ascending=total=strong=count=0
            for g in np.unique(group):
                idx=anchor[group==g]
                if len(idx)<4:continue
                diff=np.diff(dn[idx]);ascending+=int((diff>0).sum());total+=len(diff);count+=1
                strong+=int((diff>0).all() or (diff<0).all())
            block.append(dict(file=name,width=width,offset=off,blocks=count,adjacent_pairs=total,increasing_fraction=ascending/total if total else np.nan,fully_monotone_blocks=strong))
say('Partial and block shuffle',table(block)+'\n\nOffsets 0, half-block and last position are tested. Fully monotone tiny blocks can occur by chance (four anchors: probability 2/4!); counts are descriptive, not automatically positives. No labels select widths.')

stream=[];stream_candidates=[]
def inspect_stream(name,seed,a,scope):
    a=np.asarray(a,dtype=np.uint32);hit=hits(a);h=int(hit.sum());expected=len(a)*len(allids)/2**32
    stream.append(dict(api=name,seed=seed,draws=len(a),hits=h,chance=expected,scope=scope))
    if h>=max(30,expected*8):stream_candidates.append((name,seed,h))
# Broad seed search uses 4096 initial words. Interleaved consumption is a separate limitation.
for i,s in enumerate(seeds):
    r=random.Random(s)
    inspect_stream('Python getrandbits32',s,np.fromiter((r.getrandbits(32) for _ in range(4096)),np.uint32),'all seeds')
    inspect_stream('RandomState uint32 [0,2^32)',s,np.random.RandomState(s).randint(0,2**32,4096,dtype=np.uint32),'all seeds')
    inspect_stream('Generator uint32 [0,2^32)',s,np.random.default_rng(s).integers(0,2**32,4096,dtype=np.uint32),'all seeds')
    if i%2000==0:print('ID seeds',i,flush=True)
for s in smallseeds:
    r=random.Random(s);inspect_stream('Python randrange32',s,[r.randrange(2**32) for _ in range(4096)],'expanded constants/0-100')
    r=random.Random(s);inspect_stream('Python choices hex8',s,[int(''.join(r.choices('0123456789abcdef',k=8)),16) for _ in range(4096)],'expanded constants/0-100')
    r=random.Random(s);u=[uuid.UUID(int=r.getrandbits(128),version=4).hex for _ in range(4096)]
    for part in ['first8','last8']:
        inspect_stream('seeded UUID4 '+part,s,[int(v[:8] if part=='first8' else v[-8:],16) for v in u],'Faker uuid4 recipe proxy; package unavailable')
    for api in ['RandomState','Generator']:
        for dtype in [np.int64,np.uint64,np.int32]:
            for bound in ([2**31-1] if dtype==np.int32 else [2**31,2**32-1,2**32]):
                rng=np.random.RandomState(s) if api=='RandomState' else np.random.default_rng(s)
                a=rng.randint(0,bound,4096,dtype=dtype) if api=='RandomState' else rng.integers(0,bound,4096,dtype=dtype)
                inspect_stream(f'{api} {dtype.__name__} upper={bound}',s,a,'expanded constants/0-100')
ST=pd.DataFrame(stream)
say('Seeded transaction-ID stream tests',f'Full results summarized per API below: {len(stream):,} candidate streams, each first 4096 values. Most hits at any seed are shown; expected chance uses the observed ID-set size. Threshold >=max(30,8×chance). Candidates: {stream_candidates}. Faker is not installed: its general providers were NOT run; seeded stdlib UUID4 first/last-eight recipes were tested as explicit proxies. No package was installed. Prefix negatives do not reject large burn-in, salted encoding, another Faker provider/version, or an RNG mixed with other field draws.\n\n'+table(ST.sort_values('hits',ascending=False).groupby('api',sort=False).head(1).to_dict('records')))

def untemper(x):
    x=np.asarray(x,dtype=np.uint32).copy();x^=x>>18;x^=(x<<15)&np.uint32(0xefc60000)
    z=x.copy()
    for _ in range(5):z=x^((z<<7)&np.uint32(0x9d2c5680))
    x=z;z=x.copy()
    for _ in range(3):z=x^(z>>11)
    return z
def twist(a,b):
    z=(a&np.uint32(0x80000000))|(b&np.uint32(0x7fffffff))
    return (z>>1)^np.where(z&1,np.uint32(0x9908b0df),np.uint32(0))
control=untemper(np.random.RandomState(2026).randint(0,2**32,10000,dtype=np.uint32))
assert np.array_equal(control[397:-227]^twist(control[:-624],control[1:-623]),control[624:])
mt=[]
for name in frames:
    for order in ['file','device']:
        ix=np.arange(len(frames[name])) if order=='file' else np.flatnonzero(devices[name]>423000)[np.argsort(devices[name][devices[name]>423000],kind='stable')]
        state=untemper(ids[name][ix]);predstate=state[397:-227]^twist(state[:-624],state[1:-623]);h=int((predstate==state[624:]).sum())
        mt.append(dict(file=name,order=order,recurrences=len(predstate),matches=h,chance=len(predstate)/2**32))
# Seed-free necessary relation over unordered set; not enough to reconstruct chronology.
states=untemper(allids);low=np.unique(states&np.uint32(0x7fffffff));rng=np.random.default_rng(78123)
counts=[]
for null in [False,True]:
    a=states if not null else rng.integers(0,2**32,len(states),dtype=np.uint32)
    target=np.unique(a&np.uint32(0x7fffffff));total=0;trials=2_000_000
    for _ in range(4):
        delta=a[rng.integers(len(a),size=500000)]^a[rng.integers(len(a),size=500000)]
        odd=delta>>31;joined=((delta^np.where(odd,np.uint32(0x9908b0df),np.uint32(0)))<<1)|odd
        candidate_low=joined&np.uint32(0x7fffffff);loc=np.searchsorted(target,candidate_low)
        total+=int(((loc<len(target))&(target[np.minimum(loc,len(target)-1)]==candidate_low)).sum())
    counts.append(dict(kind='matched-size uniform null' if null else 'untempered observed ID set',sampled_pairs=trials,necessary_relation_hits=total,independence_expectation=trials*len(target)/2**31))
say('Seed-free MT19937',table(mt)+'\n\n'+table(counts)+'\n\nInverting the twist of a sampled pair XOR tests whether its required next-state low 31 bits occurs in the ID set. This is only a necessary unordered-set relation; it omits unknown lag positions and the predecessor high bit. With millions of words the set is dense enough for many accidental completions. A negative/near-null result does NOT exclude MT19937 under shuffling and missing/interleaved draws. Device-order recurrence assumes consecutive outputs per observed device, which is stronger than known creation order. File-order recurrence assumes no shuffle. Neither assumption is established. No state recovery claim is made.')

hashes={'md5':lambda b:hashlib.md5(b).hexdigest(),'sha1':lambda b:hashlib.sha1(b).hexdigest(),'sha256':lambda b:hashlib.sha256(b).hexdigest(),'crc32':lambda b:f'{zlib.crc32(b):08x}'}
content=[];sample=frames['train'].iloc[np.linspace(0,len(frames['train'])-1,1024,dtype=int)]
for fields in itertools.permutations(['sender_id','receiver_id','timestamp','amount']):
    for sep in ['',':','|',',','_',' ']:
        for amountstyle in ['raw','fixed2']:
            pieces=[]
            for row in sample.to_dict('records'):
                row['amount']=row['amount'] if amountstyle=='raw' else f'{float(row["amount"]):.2f}'
                pieces.append(sep.join(row[f] for f in fields).encode())
            for name,fn in hashes.items():
                for end in ['first','last'] if name!='crc32' else ['first']:
                    got=[fn(v)[:8] if end=='first' else fn(v)[-8:] for v in pieces]
                    h=sum(a==b[2:].lower() for a,b in zip(got,sample.txn_id))
                    content.append(dict(fields='+'.join(fields),separator=repr(sep),amount=amountstyle,hash=name,slice=end,rows=1024,matches=h))
say('Hashes of row content',f'{len(content)} fixed recipes tested on 1024 evenly spaced TRAIN rows, with exact per-row ID equality; no labels used. All 24 orders of sender/receiver/timestamp/amount, six separators, raw or two-decimal amount, md5/sha1/sha256 first/last eight and crc32. Maximum matches: {max(x["matches"] for x in content)}. This does not cover arbitrary salts, JSON serialization or other field subsets.\n\n'+table(sorted(content,key=lambda x:x['matches'],reverse=True)[:8]))
counter=[];salts=['','TX','txn','transaction','train','test','42','2026','upi']
for salt in salts:
    print('counter salt',repr(salt),flush=True)
    for name,fn in hashes.items():
        # Complete per-file counter domains covered by union [0,max file length].
        h=0;first=[]
        for i in range(max(sizes)+1):
            v=int(fn((salt+str(i)).encode())[:8],16)
            if v in idset:
                h+=1
                if len(first)<5:first.append(i)
        expected=(max(sizes)+1)*len(allids)/2**32
        counter.append(dict(salt=repr(salt),hash=name,counters=max(sizes)+1,hits=h,chance=expected,first_hit_counters=str(first)))
        if h>8*expected:positives.append(dict(test='counter_hash',salt=salt,hash=name,matches=h))
say('Counter hashes with prefixes/salts',table(counter)+'\n\nEach recipe scans counters 0 through train+test length inclusive, covering both per-file lengths and the combined length. Prefix is literal, decimal counter unpadded, digest first eight. Chance collisions are not evidence of recovered rows. Suffix salts, nondecimal encodings, padding and arbitrary offsets are outside this bounded family.')

sequential=[];phases=[]
edges=json.loads((ROOT/'data/processed/band_edges.json').read_text())
for name,d in frames.items():
    dn=devices[name];ix=np.flatnonzero(dn>423000);ix=ix[np.argsort(dn[ix],kind='stable')]
    ts=pd.to_datetime(d.timestamp);numeric={c:pd.to_numeric(d[c]).to_numpy() for c in ['amount','balance_before','sim_age_hours','device_age_days']}
    numeric.update(second=ts.dt.second.to_numpy(),minute=ts.dt.minute.to_numpy(),seconds_of_day=(ts.dt.hour*3600+ts.dt.minute*60+ts.dt.second).to_numpy())
    for col,values in numeric.items():
        for transform in ['value','fraction'] if col in ['balance_before','sim_age_hours','device_age_days','amount'] else ['value']:
            v=values if transform=='value' else np.round(values-np.floor(values),6)
            a=v[ix];sequential.append(dict(file=name,column=col,transform=transform,anchors=len(ix),rho_device=float(spearmanr(dn[ix],a).statistic),lag1=float(np.corrcoef(a[:-1],a[1:])[0,1]),unique=len(np.unique(a))))
    b=np.searchsorted(edges[name],dn,side='right');b=np.where((dn>=edges[name][0])&(dn<edges[name][-1]),b,0)
    for phase in range(12):
        m=(b==phase)&(dn>423000)
        phases.append(dict(file=name,phase=phase,n=int(m.sum()),mean_second=float(ts.dt.second[m].mean()),mean_minute=float(ts.dt.minute[m].mean()),sim_fraction_mean=float((numeric['sim_age_hours'][m]%1).mean()),device_fraction_mean=float((numeric['device_age_days'][m]%1).mean()),balance_fraction_mean=float((numeric['balance_before'][m]%1).mean())))
say('Sequential numeric and timestamp structure',table(sequential)+'\n\n'+table(phases)+'\n\nPhase 0 here means ordinary newly created devices; phases 1–11 use existing band edges. Shape-related differences in ages, timestamps, whole amounts or balances are not a second generation-order leak. These statistics do not recover home-device order. Fractions and age ranges are often rounded or recipe-specific, so matching coarse decimals alone is uninformative.')
# Direct seeded RNG sequence alignment on known-device rows. Test first 256 anchors
# for normalized fractional/uniform candidates and compare with shuffled order.
numtests=[]
for name,d in frames.items():
    dn=devices[name];ix=np.flatnonzero(dn>423000);ix=ix[np.argsort(dn[ix],kind='stable')][:256]
    for col in ['sim_age_hours','device_age_days','balance_before']:
        values=pd.to_numeric(d[col]).to_numpy()[ix]
        dec=1 if col!='balance_before' else 2
        for scale in [1,10,100,1000,10000,100000]:
            best=(-1,None,None)
            for seed in smallseeds:
                for api in ['RandomState','Generator']:
                    r=np.random.RandomState(seed).random_sample(256) if api=='RandomState' else np.random.default_rng(seed).random(256)
                    h=int((np.round(r*scale,dec)==values).sum())
                    if h>best[0]:best=(h,seed,api)
            numtests.append(dict(file=name,column=col,scale=scale,seeds=len(smallseeds),apis=2,anchors=256,max_equal=best[0],best_seed=best[1],best_api=best[2]))
say('Seeded numeric-sequence probes',table(numtests)+'\n\nFresh-state RandomState/default_rng uniform draws scaled by listed constants and rounded to the CSV precision, matched in order to the first 256 creation-device rows. Tests exclude arbitrary affine offsets, nonuniform distributions and unknown skipped draws. Device creation does not imply consecutive calls to any one field RNG. A negative here rejects only this aligned simple-uniform construction, not seeded generation in general.')

say('Positives and limitations',table(positives)+'\n\nA leaderboard jump is not evidence identifying a generator leak. This audit tests explicit finite families. No test labels are read. There is no defensible label-purity-by-recovered-position claim unless a candidate actually recovers order beyond known bands. The pre-existing DV leak is used only as the anchor. Full results are summarized per family with exact seed ranges, lengths, thresholds and maximums; candidate ties at zero do not imply a preferred seed. Faker providers, unknown RNG burn-in/interleaving, arbitrary salts and shuffle composition remain untested.')
OUT.write_text('# Generator-order leak audit — 14 November 2026\n\nNew files only: `scripts/explore/41_codex_generator_leak.py` and this report. No fitting or existing-file changes. Numerical threads limited to four.\n\n'+SUMMARY+'\n\n'+ '\n'.join(sections)+f'\nRuntime: {time.time()-start:.1f} seconds. Run `C:/Users/ASUS/AppData/Local/Programs/Python/Python311/python.exe -I scripts/explore/41_codex_generator_leak.py` in a fresh output state; the script refuses to overwrite this report.\n',encoding='utf-8')
print('DONE',OUT,flush=True)


def extra_checks(root):
    """Additional finite families, callable independently without rerunning shuffles."""
    import numpy as np
    import pandas as pd
    import random,hashlib,zlib,itertools
    from pathlib import Path
    from scipy.stats import spearmanr
    root=Path(root)
    ds={k:pd.read_csv(root/f'data/raw/{k}.csv',usecols=['txn_id','device_id','sender_id','receiver_id','timestamp','amount']+(['label'] if k=='train' else []),dtype=str) for k in ['train','test']}
    xx={k:np.fromiter((int(s[2:],16) for s in d.txn_id),np.uint32) for k,d in ds.items()}
    allx=np.unique(np.r_[xx['train'],xx['test']]);st=set(map(int,allx))
    def hit(a):
        i=np.searchsorted(allx,a);return (i<len(allx))&(allx[np.minimum(i,len(allx)-1)]==a)
    def tab(rows):
        d=pd.DataFrame(rows)
        return '\n'.join(['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |']+['| '+' | '.join(str(v) for v in row)+' |' for row in d.itertuples(index=False,name=None)])
    text=['## Additional ID structure, provider and hash tests']
    text.append(f'Combined distinct IDs: {len(allx)} out of {sum(map(len,ds.values()))}; observed within-file duplicates and cross-file overlap are zero. Independent unconstrained 32-bit draws would have approximately {len(allx)*(len(allx)-1)/2**33:.1f} colliding pairs. This is evidence of an ID-uniqueness mechanism or injective construction, not recovered generation order: transaction IDs are required to be unique and rejection sampling suffices.')
    transitions=[]
    for name,a,c in [('NumericalRecipes',1664525,1013904223),('ANSI_C',1103515245,12345),('MSVC',214013,2531011),('Borland',22695477,1),('identity_increment',1,1)]:
        v=((allx.astype(np.uint64)*a+c)&0xffffffff).astype(np.uint32)
        transitions.append(dict(test=name,ids=len(allx),successor_in_set=int(hit(v).sum()),chance=len(allx)**2/2**32))
    v=allx.copy();v^=v<<13;v^=v>>17;v^=v<<5
    transitions.append(dict(test='xorshift32_13_17_5',ids=len(allx),successor_in_set=int(hit(v).sum()),chance=len(allx)**2/2**32))
    text.append(tab(transitions))
    text.append('These are seed-free necessary tests for a consecutive full-state stream. Interleaving/missing outputs can break adjacency; ordinary chance successor hits do not identify a seed.')
    # Invert 4-byte CRC32 as an affine 32-bit bijection using a GF(2) basis.
    const=zlib.crc32(bytes(4));basis={}
    for bit in range(32):
        value=zlib.crc32((1<<bit).to_bytes(4,'little'))^const;pre=1<<bit
        while value:
            pivot=value.bit_length()-1
            if pivot in basis:value^=basis[pivot][0];pre^=basis[pivot][1]
            else:basis[pivot]=(value,pre);break
    def inv_crc(a):
        rem=a^np.uint32(const);out=np.zeros(len(a),np.uint32)
        for bit in range(31,-1,-1):
            selected=((rem>>bit)&1).astype(bool);value,pre=basis[bit]
            rem[selected]^=np.uint32(value);out[selected]^=np.uint32(pre)
        assert not rem.any()
        return out
    def undo_right(a,shift):
        out=a.copy()
        for _ in range(32//shift+1):out=a^(out>>shift)
        return out
    def inv_fmix(a):
        a=undo_right(a,16);a=(a.astype(np.uint64)*pow(0xc2b2ae35,-1,2**32)).astype(np.uint32)
        a=undo_right(a,13);a=(a.astype(np.uint64)*pow(0x85ebca6b,-1,2**32)).astype(np.uint32)
        return undo_right(a,16)
    inversion=[]
    for k,a in xx.items():
        dnum=ds[k].device_id.str[2:].to_numpy(np.int64);anchors=dnum>423000
        crc=inv_crc(a)
        for label,v in [('CRC32 little endian counter',crc),('CRC32 big endian counter',crc.byteswap()),('Murmur3 fmix32 inverse',inv_fmix(a))]:
            inversion.append(dict(file=k,inverse=label,rows=len(a),decoded_below_combined_N=int((v<sum(map(len,ds.values()))).sum()),anchor_rho=float(spearmanr(dnum[anchors],v[anchors]).statistic)))
    text.append(tab(inversion))
    text.append('Inverse transforms map every ID to some integer by construction. Only compact counter support or monotone creation-device ranks would indicate recovered order; universal invertibility itself is not a positive.')
    seeds=sorted(set(range(101))|{123,321,777,999,1024,1234,1337,1729,2026,4096,12345,54321,31337,314159,271828,20260701,20261001,20261113,20261114,20261115})
    provider=[]
    for alg in ['md5','sha1','sha256']:
        for part in ['first8','last8']:
            best=(-1,None)
            for seed in seeds:
                r=random.Random(seed);vals=[]
                for _ in range(4096):
                    h=getattr(hashlib,alg)(str(r.random()).encode()).hexdigest();vals.append(int(h[:8] if part=='first8' else h[-8:],16))
                h=int(hit(np.array(vals,np.uint32)).sum())
                if h>best[0]:best=(h,seed)
            provider.append(dict(provider=alg,part=part,seeds=len(seeds),draws=4096,max_hits=best[0],best_seed=best[1],chance=4096*len(allx)/2**32))
    text.append(tab(provider))
    text.append('Faker is unavailable locally. The tested hash-provider recipes are reproduced directly from [Faker misc provider source](https://raw.githubusercontent.com/joke2k/faker/master/faker/providers/misc/__init__.py): md5/sha1/sha256 hash the string form of random.random(); uuid4 wraps random.getrandbits(128), already tested above. No claim is made to cover all providers, versions or earlier shared RNG consumption.')
    # Salt layouts absent from the first sweep; bounded prefix probe, not full N.
    salts=['42:','2026:','123:','1337:','train:','test:','TX_','txn_','transaction_','upi:']
    counters=[]
    for alg in ['md5','sha1','sha256','blake2s','blake2b']:
        for salt in salts:
            for position in ['prefix','suffix']:
                count=[0,0]
                for i in range(65536):
                    raw=salt+str(i) if position=='prefix' else str(i)+salt
                    h=getattr(hashlib,alg)(raw.encode()).hexdigest()
                    count[0]+=int(int(h[:8],16) in st);count[1]+=int(int(h[-8:],16) in st)
                counters.append(dict(hash=alg,salt=repr(salt),position=position,counters=65536,first8_hits=count[0],last8_hits=count[1],chance=65536*len(allx)/2**32))
    text.append(tab(counters))
    text.append('These 200 additional digest/salt/layout candidates test only counters 0..65535, in contrast to the full counter-domain sweep above. Large starting offsets and arbitrary salts remain unresolved.')
    # Per-file row hash confirmation, including simple field subsets and JSON.
    content=[]
    for name,d in ds.items():
        sample=d.iloc[np.linspace(0,len(d)-1,512,dtype=int)].to_dict('records')
        best=[]
        for fields in [('sender_id','receiver_id','timestamp','amount'),('timestamp','sender_id','receiver_id'),('sender_id','receiver_id'),('timestamp','amount')]:
            for sep in ['',':','|',',','_']:
                for salt in ['','TX','42:','2026:']:
                    for alg in ['md5','sha1','sha256','blake2s','blake2b']:
                        match=0
                        for row in sample:
                            raw=salt+sep.join(row[f] for f in fields)
                            match+=getattr(hashlib,alg)(raw.encode()).hexdigest()[:8]==row['txn_id'][2:].lower()
                        best.append((match,fields,sep,salt,alg))
        mx=max(best,key=lambda x:x[0]);content.append(dict(file=name,recipes=len(best),rows_per_recipe=512,max_matches=mx[0],max_recipe=str(mx[1:])))
    text.append(tab(content))
    text.append('No significant-result claim is made from these additional searches. If no compact/monotone inverse or excess stream matches are found, no train label-purity table can legitimately be attributed to recovered order.')
    # Full-length follow-up to the largest prefix hit for each broad-seed family.
    # Read the already-written summary to select those seeds without labels.
    report=(root/'docs/codex_generator_leak.md').read_text(encoding='utf-8')
    donor=np.r_[xx['train'],xx['test']];dn=np.r_[ds['train'].device_id.str[2:].to_numpy(np.int64),ds['test'].device_id.str[2:].to_numpy(np.int64)]
    order=np.argsort(donor);sortedid=donor[order];sorteddn=dn[order];follow=[]
    for api in ['Python getrandbits32','RandomState uint32 [0,2^32)','Generator uint32 [0,2^32)']:
        line=next(l for l in report.splitlines() if l.startswith('| '+api+' |'))
        seed=int(line.split('|')[2].strip());N=len(donor)
        if api.startswith('Python'):
            r=random.Random(seed);a=np.fromiter((r.getrandbits(32) for _ in range(N)),np.uint32)
        elif api.startswith('RandomState'):a=np.random.RandomState(seed).randint(0,2**32,N,dtype=np.uint32)
        else:a=np.random.default_rng(seed).integers(0,2**32,N,dtype=np.uint32)
        loc=np.searchsorted(sortedid,a);ok=(loc<N)&(sortedid[np.minimum(loc,N-1)]==a)
        anchor=ok&(sorteddn[np.minimum(loc,N-1)]>423000)
        follow.append(dict(api=api,seed=seed,draws=N,matching_ids=int(ok.sum()),chance=N*N/2**32,creation_anchors=int(anchor.sum()),rho_stream_vs_device=float(spearmanr(np.flatnonzero(anchor),sorteddn[loc[anchor]]).statistic)))
    text.append('### Full-length ID-stream partial-order follow-up')
    text.append(tab(follow))
    text.append('For each broad-seed stream family, extend its largest-prefix-hit seed to train+test length and compare stream position with creation-device number on matched anchors. Seed selection uses ID overlap only. This follows up apparent prefix outliers; it is not a full-length scan of all seeds.')
    return '\n\n'.join(text)+'\n'


with OUT.open('a',encoding='utf-8') as extra_output:
    extra_output.write(extra_checks(ROOT))
print('EXTRAS DONE',flush=True)


def numeric_coverage(root):
    """Broaden descriptive order checks to all remaining numeric raw columns."""
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr
    cols=['amount','sim_age_hours','device_age_days','sender_account_age_days','new_payee','pin_retries','login_to_txn_sec','balance_before','ip_risk_score','qr_vpa_mismatch']
    rows=[]
    for name in ['train','test']:
        d=pd.read_csv(Path(root)/f'data/raw/{name}.csv',usecols=['device_id']+cols)
        dn=d.device_id.str[2:].to_numpy(np.int64);ix=np.flatnonzero(dn>423000);ix=ix[np.argsort(dn[ix],kind='stable')]
        for c in cols:
            v=d[c].to_numpy()[ix];good=np.isfinite(v);v=v[good]
            rows.append(dict(file=name,column=c,finite_anchors=len(v),rho=float(spearmanr(dn[ix][good],v).statistic),lag1=float(np.corrcoef(v[:-1],v[1:])[0,1]) if len(np.unique(v))>1 else np.nan))
    d=pd.DataFrame(rows)
    tab='\n'.join(['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |']+['| '+' | '.join(str(v) for v in row)+' |' for row in d.itertuples(index=False,name=None)])
    return '\n## All numeric raw-column order checks\n\n'+tab+'\n\nAll ten numeric raw columns are checked for rank relationship and lag-1 structure along creation-device order. ip_risk_score is included only as a diagnostic of a known train/test-changing field, never as a fraud rule. Recipe/phase shifts and discrete support can create autocorrelation without recovering generation order. Seeded exact-uniform sequence matching above covers ages and balances only; arbitrary numeric distributions, transformations and skipped draws remain outside that finite sweep.\n'


with OUT.open('a',encoding='utf-8') as numeric_output:
    numeric_output.write(numeric_coverage(ROOT))
