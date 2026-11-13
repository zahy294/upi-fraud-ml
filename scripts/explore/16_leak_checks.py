"""Three quick checks for generator artefacts: month-boundary tails, shuffle seed, txn_id as a hash of a counter."""
import sys, hashlib, zlib
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)
tr = pd.read_csv("data/raw/train.csv", usecols=["txn_id", "timestamp", "label", "device_id", "sender_id"])
te = pd.read_csv("data/raw/test.csv", usecols=["txn_id", "timestamp", "device_id", "sender_id"])
tr["ts"] = pd.to_datetime(tr.timestamp); te["ts"] = pd.to_datetime(te.timestamp)
print("(a) train rows dated October:", int((tr.ts >= "2026-10-01").sum()), tr[tr.ts >= "2026-10-01"].label.value_counts().to_dict())
print("    test rows dated November:", int((te.ts >= "2026-11-01").sum()), "| test rows dated before October:", int((te.ts < "2026-10-01").sum()),
      "| train rows dated before July:", int((tr.ts < "2026-07-01").sum()))
sub = pd.read_csv("submissions/s1_v3.csv").set_index("id").label
nov = te[te.ts >= "2026-11-01"]
print("    current predictions for the November test rows:", sub.reindex(nov.txn_id).value_counts().to_dict())
# first/last hours of each split: label mix near the start boundary
print("    train first 2 hours of July label mix:", tr[tr.ts < "2026-07-01 02:00"].label.value_counts().head(4).to_dict())

print("\n(b) shuffle seed: does un-shuffling with a known seed reveal an ordered file?")
for name, d in (("train", tr), ("test", te)):
    n = len(d); t = d.ts.values.astype("int64") // 10**9
    dn = d.device_id.str[2:].astype(np.int64).values
    fraud = (d.label != "legit").values.astype(float) if "label" in d else None
    print(f"  {name}: corr(file position, time) = {np.corrcoef(np.arange(n), t)[0,1]:+.4f}")
    best = []
    for seed in list(range(0, 101)) + [123, 1234, 12345, 2024, 2025, 2026, 2027, 1337, 31337, 777, 999, 20261113, 314159]:
        for api, perm in (("RandomState", np.random.RandomState(seed).permutation(n)), ("default_rng", np.random.default_rng(seed).permutation(n))):
            for way, pos in (("fwd", perm), ("inv", np.argsort(perm))):
                c_t = abs(np.corrcoef(pos, t)[0, 1]); c_d = abs(np.corrcoef(pos, dn)[0, 1])
                c_f = abs(np.corrcoef(pos, fraud)[0, 1]) if fraud is not None else 0.0
                best.append((max(c_t, c_d, c_f), seed, api, way, round(c_t, 4), round(c_d, 4), round(c_f, 4)))
    best.sort(reverse=True)
    print("   top candidates (max |corr|, seed, api, direction, |corr| with time / device number / fraud):", best[:3])

print("\n(c) txn_id as a hash of a counter?")
ids = set(tr.txn_id.str[2:]) | set(te.txn_id.str[2:])
N = 2_200_000
def rate(fn):
    hit = sum(1 for i in range(N) if fn(i) in ids)
    return hit / len(ids)
tests = {
    "md5(str(i))": lambda i: hashlib.md5(str(i).encode()).hexdigest()[:8],
    "sha1(str(i))": lambda i: hashlib.sha1(str(i).encode()).hexdigest()[:8],
    "sha256(str(i))": lambda i: hashlib.sha256(str(i).encode()).hexdigest()[:8],
    "crc32(str(i))": lambda i: "%08x" % zlib.crc32(str(i).encode()),
    "md5('TX'+str(i))": lambda i: hashlib.md5(("TX" + str(i)).encode()).hexdigest()[:8],
}
print("   share of ids reproduced (chance level is about %.4f):" % (N / 16**8))
for k, fn in tests.items():
    print("   ", k, round(rate(fn), 4))
for seed in (0, 1, 42, 123, 2026):
    r = np.random.RandomState(seed).randint(0, 2**32, size=N, dtype=np.uint64)
    g = np.random.default_rng(seed).integers(0, 2**32, size=N, dtype=np.uint64)
    for nm, arr in (("RandomState.randint", r), ("default_rng.integers", g)):
        hit = len(ids & set("%08x" % v for v in arr))
        if hit / len(ids) > 0.01:
            print("   ", nm, "seed", seed, "reproduces", round(hit / len(ids), 4))
print("   (no line above for the numpy generators means none matched)")
