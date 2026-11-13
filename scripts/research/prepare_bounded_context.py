"""Build original EDA features with no later-month transaction context.

Reuses the original feature-building prefix verbatim and skips its plotting and
diagnostics, which assume a nonempty October test set. No source is modified.
"""
import sys
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SOURCE = Path(sys.argv[1]).resolve()
DEST = Path(sys.argv[2]).resolve()
END = pd.Timestamp(sys.argv[3])
inter = DEST / "data/interim"
inter.mkdir(parents=True, exist_ok=True)
original = SOURCE / "data/interim"
train = pd.read_parquet(original / "train.parquet")
train = train[train.timestamp < END]
train.to_parquet(inter / "train.parquet", index=False)
test = pd.read_parquet(original / "test.parquet")
test.iloc[:0].to_parquet(inter / "test.parquet", index=False)
del train, test
for name in ["accounts", "merchants"]:
    target = inter / f"{name}.parquet"
    if not target.exists():
        target.symlink_to(original / f"{name}.parquet")
source_path = REPO / "scripts/eda/04_relational.py"
text = source_path.read_text()
marker = "out = []\n# ---------- density"
assert text.count(marker) == 1
prefix = text.split(marker)[0]
sys.argv = [str(source_path), str(DEST)]
exec(compile(prefix, str(source_path), "exec"), {"__file__": str(source_path), "__name__": "__main__"})
