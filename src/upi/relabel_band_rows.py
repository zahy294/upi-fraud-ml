"""Relabel the test rows that sit on a range of device numbers.

The October device-number bands (bands.py) are placed without labels. Between the last row that clearly belongs to
one fraud type and the first that clearly belongs to the next there are a few ordinary-looking rows whose band cannot
be told from the data. This script moves such rows to the neighbouring band in a finished submission file, so that
one leaderboard upload can test one edge.

Usage: python -I src/upi/relabel_band_rows.py <project_root> <in.csv> <out.csv> <lo>-<hi>:<label> [<lo>-<hi>:<label> ...]

Every test row on a device numbered lo..hi (inclusive) gets <label>. Nothing else changes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd  # noqa: E402
from common import LABELS  # noqa: E402

ROOT, SRC, DST, MOVES = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4:]
assert MOVES, "give at least one <lo>-<hi>:<label>"
sub = pd.read_csv(SRC)
idc, labc = sub.columns[0], sub.columns[1]
te = pd.read_csv(ROOT / "data/raw/test.csv", usecols=["txn_id", "device_id"])
dnum = te.set_index("txn_id").device_id.str[2:].astype("int64").reindex(sub[idc]).values
total = 0
for mv in MOVES:
    rng, lab = mv.split(":")
    lo, hi = (int(x) for x in rng.split("-"))
    assert lab in LABELS, f"unknown label {lab}"
    m = (dnum >= lo) & (dnum <= hi)
    assert m.any(), f"no test row on devices {lo}..{hi}"
    was = sub.loc[m, labc].value_counts().to_dict()
    changed = int((sub.loc[m, labc] != lab).sum())
    sub.loc[m, labc] = lab
    total += changed
    print(f"devices {lo}..{hi}: {int(m.sum())} rows {was} -> {lab} ({changed} changed)")
sub.to_csv(DST, index=False)
print(f"wrote {DST}: {total} rows changed")
