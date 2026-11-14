import sys, pandas as pd, numpy as np
d = pd.read_parquet('work.parquet', columns=['txn_id','split','y_raw','dnum','timestamp'])
x = d.txn_id.str[2:].apply(lambda s: int(s,16)).values.astype(np.uint64)
print('n', len(x), 'unique', len(np.unique(x)), 'min', x.min(), 'max', x.max(), 'max/2^32', x.max()/2**32)
bits = [(int(((x>>np.uint64(b))&np.uint64(1)).mean()*1e5))/1e5 for b in range(32)]
print('bit means', bits)
for m in [3,5,7,10,16,255,256,257,1000,65521,65537]:
    c = np.bincount((x % np.uint64(m)).astype(int), minlength=m)
    print('mod', m, 'chi2/df', round(((c-c.mean())**2/c.mean()).sum()/(m-1),3))
# hex char position frequency
s = d.txn_id.str[2:]
for p in range(8):
    vc = s.str[p].value_counts()
    print('pos', p, 'min/max char freq', vc.min(), vc.max())
# sorted spacing: uniform random -> exponential spacing; check min spacing & share of tiny gaps
xs = np.sort(x); g = np.diff(xs).astype(np.int64)
print('spacing mean', g.mean(), 'min', g.min(), 'share<10', (g<10).mean(), 'expected', 1-np.exp(-10/g.mean()))
print('spacing<=1 count', (g<=1).sum(), 'expected', len(g)*(1-np.exp(-1.5/g.mean())))
# AP test: differences between random subset and all
rng = np.random.default_rng(0)
S = set(x.tolist())
A = rng.choice(x, 40, replace=False)
xi = x.astype(np.int64)
diffs = ((xi[None,:] - A.astype(np.int64)[:,None]) % (1<<32)).ravel()
u, c = np.unique(diffs, return_counts=True)
o = np.argsort(-c)[:10]
print('top pairwise differences (value,count):', [(int(u[i]), int(c[i])) for i in o])
# LCG successor tests
def test(f, name):
    y = f(x)
    hit = np.isin(y, x).sum()
    print(name, 'successor hits', int(hit), 'expected by chance', round(len(x)**2/2**32,1))
M = np.uint64(2**32-1)
test(lambda v: (v*np.uint64(1664525)+np.uint64(1013904223)) & M, 'LCG numrecipes')
test(lambda v: (v*np.uint64(1103515245)+np.uint64(12345)) & M, 'LCG glibc')
test(lambda v: (v*np.uint64(214013)+np.uint64(2531011)) & M, 'LCG msvc')
test(lambda v: (v*np.uint64(22695477)+np.uint64(1)) & M, 'LCG borland')
test(lambda v: (v*np.uint64(69069)+np.uint64(1)) & M, 'LCG 69069')
test(lambda v: (v*np.uint64(134775813)+np.uint64(1)) & M, 'LCG delphi')
test(lambda v: (v + np.uint64(0x9E3779B9)) & M, 'weyl golden')
test(lambda v: (v + np.uint64(1)) & M, '+1')
def xs32(v):
    v = v ^ ((v << np.uint64(13)) & M); v = v ^ (v >> np.uint64(17)); v = v ^ ((v << np.uint64(5)) & M); return v
test(xs32, 'xorshift32')
