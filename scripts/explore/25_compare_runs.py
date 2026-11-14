"""Compare the from-scratch run (fresh clone) with the artefacts in the project folder, step by step.

Usage: python -I a06_compare_repro.py <project_root> <repro_root> [submission_name]
"""
import json
import sys
from pathlib import Path

MAIN, REP = Path(sys.argv[1]), Path(sys.argv[2])
NAME = sys.argv[3] if len(sys.argv) > 3 else "final_local"
sys.path.insert(0, str(MAIN / "src/upi"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import LABELS, macro_f1, per_class  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)

print("== 1. tables: same rows, same order, same values? ==")
bm = pd.read_parquet(MAIN / "data/processed/base_v0.parquet", columns=["txn_id", "label", "split", "receiver_type"])
br = pd.read_parquet(REP / "data/processed/base_v0.parquet", columns=["txn_id", "label", "split", "receiver_type"])
same_order = len(bm) == len(br) and (bm.txn_id.values == br.txn_id.values).all()
print("base_v0 row order identical:", same_order, "| rows", len(bm), len(br))
assert same_order, "row order differs: everything downstream must be compared by txn_id"
for f in ("base_v0", "ratios_v0", "relational_v0", "features_v1", "features_v2", "features_v3", "features_v4", "features_v5"):
    pm, pr = MAIN / f"data/processed/{f}.parquet", REP / f"data/processed/{f}.parquet"
    if not pr.exists():
        print(f"{f}: missing in the from-scratch run")
        continue
    a, b = pd.read_parquet(pm), pd.read_parquet(pr)
    if list(a.columns) != list(b.columns):
        print(f"{f}: COLUMNS DIFFER; only in project: {sorted(set(a.columns) - set(b.columns))}; only in repro: {sorted(set(b.columns) - set(a.columns))}")
    if "txn_id" in a.columns and "txn_id" in b.columns and len(a) == len(b) and not (a.txn_id.values == b.txn_id.values).all():
        b = b.set_index("txn_id").reindex(a.txn_id.values).reset_index()
    bad = []
    for c in [c for c in a.columns if c in b.columns]:
        x, z = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(z):
            xv, zv = x.values.astype("float64"), z.values.astype("float64")
            diff = ~((xv == zv) | (np.isnan(xv) & np.isnan(zv)))
            if diff.any():
                bad.append((c, int(diff.sum()), float(np.nanmax(np.abs(xv[diff] - zv[diff])))))
        else:
            d = ~((x.astype("object") == z.astype("object")) | (x.isna() & z.isna())).values
            if d.any():
                bad.append((c, int(d.sum()), float("nan")))
    print(f"{f}: {len(a.columns)} columns, {'identical' if not bad else 'DIFFERENT in ' + str(len(bad)) + ' columns: ' + str(bad[:8])}")

print("\n== 2. band edges ==")
em = json.loads((MAIN / "data/processed/band_edges.json").read_text())
er = json.loads((REP / "data/processed/band_edges.json").read_text())
print("train edges identical:", em["train"] == er["train"], "| test edges identical:", em["test"] == er["test"])
if em["test"] != er["test"]:
    print("  project:", em["test"])
    print("  repro  :", er["test"])
    print("  shift  :", (np.array(er["test"]) - np.array(em["test"])).tolist())

print("\n== 3. cleaned labels ==")
tr = (bm.split == "train").values
y_raw = bm.label[tr].map({lab: i for i, lab in enumerate(LABELS)}).values.astype(int)
ym = np.load(MAIN / "models/oof_raw/y_clean.npy")
yr = np.load(REP / "models/oof_raw/y_clean.npy")
print(f"rows changed from raw: project {int((ym != y_raw).sum())}, repro {int((yr != y_raw).sum())}; rows where the two cleanings differ: {int((ym != yr).sum())}")


def band_rules(root, pred, rows):
    F3 = pd.read_parquet(root / "data/processed/features_v3.parquet")
    cbv = pd.read_parquet(root / "data/processed/features_v5.parquet", columns=["ch_band"]).ch_band.values[rows].astype(int)
    band = F3.band.values.astype(int)[rows]
    S = np.stack([F3[f"snd_band_{k}_3h"].values[rows] for k in range(1, 12)], 1)
    R = np.stack([F3[f"rcv_band_{k}_24h"].values[rows] for k in range(1, 12)], 1)
    sb = np.where((S > 0).sum(1) == 1, S.argmax(1) + 1, 0)
    rb = np.where((R > 0).sum(1) == 1, R.argmax(1) + 1, 0)
    p2p = (bm.receiver_type.values == "P2P")[rows]
    out = np.where(band > 0, band, pred)
    free = (band == 0) & (pred > 0)
    out = np.where(free & np.isin(pred, [4, 5]) & np.isin(sb, [4, 5]), sb, out)
    out = np.where(free & ((np.isin(pred, [1, 2, 3]) & p2p & np.isin(rb, [1, 2, 3])) | (np.isin(pred, [7, 8, 11]) & ~p2p & np.isin(rb, [7, 8, 11]))), rb, out)
    out = np.where(free & (cbv > 0), cbv, out)
    return out


print("\n== 4. local score (pooled leave-one-month-out macro-F1, band rules applied) ==")
rows = []
for name, root, tag, wname in (("project oof_c3 (s5_c3)", MAIN, "oof_c3", "weights_s5_c3.npy"), ("from-scratch oof_clean", REP, "oof_clean", f"weights_{NAME}.npy")):
    oof = np.load(root / "models" / tag / "oof.npy").astype(float)
    w = np.load(root / "models" / tag / wname)
    pred = band_rules(root, (oof * w).argmax(1), tr)
    rows.append({"model": name, "vs project cleaned labels": macro_f1(ym, pred), "vs repro cleaned labels": macro_f1(yr, pred), "vs raw labels": macro_f1(y_raw, pred)})
    if root is REP:
        T = per_class(ym, pred)
        T["w"] = w
        print(T.round(3).to_string())
print(pd.DataFrame(rows).round(4).to_string(index=False))
om = np.load(MAIN / "models/oof_c3/oof.npy").astype(float)
orp = np.load(REP / "models/oof_clean/oof.npy").astype(float)
print(f"out-of-fold probabilities: mean |difference| {np.abs(om - orp).mean():.5f}, argmax agreement {(om.argmax(1) == orp.argmax(1)).mean():.5f}")

print("\n== 5. submission files ==")
new = pd.read_csv(REP / "submissions" / f"{NAME}.csv")
for other in ("s5_c3", "s3_clean_v4"):
    old = pd.read_csv(MAIN / "submissions" / f"{other}.csv")
    assert (old.id.values == new.id.values).all()
    d = old.label.values != new.label.values
    print(f"{NAME} vs {other}: {int(d.sum())} of {len(new)} rows differ ({d.mean() * 100:.3f}%); fraud predictions {int((new.label != 'legit').sum())} vs {int((old.label != 'legit').sum())}")
    if d.any():
        ct = pd.crosstab(pd.Series(old.label.values[d], name=other), pd.Series(new.label.values[d], name=NAME))
        top = ct.stack().sort_values(ascending=False).head(8)
        print("   largest changes (old -> new):", {f"{a}->{b}": int(n) for (a, b), n in top.items()})
print("predicted rows per label:", new.label.value_counts().reindex(LABELS).to_dict())
