"""One entry point: raw CSVs -> features -> models -> submissions/<name>.csv

    python run_all.py                      # full run, skipping steps whose outputs already exist
    python run_all.py --force              # rebuild everything
    python run_all.py --seeds 5 --name final

Steps (each is a script you can also run on its own):
  1. scripts/eda/00_load_and_overview.py   raw CSV -> data/interim/*.parquet
  2. scripts/eda/04_relational.py          33 relational features
  3. src/upi/build_base.py                 base table + neighbour ratios
  4. src/upi/features_v1.py, v2, v4        recipe, session and bait-timing features
  5. src/upi/run_oof.py  (tag s1)          first model without device bands; its test probabilities place the test bands
  6. src/upi/bands.py                      device-number bands -> features_v3
  7. src/upi/run_oof.py  (tag oof_raw)     leave-one-month-out predictions on the raw labels
  8. src/upi/clean_labels.py               cleaned training labels
  9. src/upi/run_oof.py  (tag oof_clean)   final model on cleaned labels, with test predictions
 10. src/upi/make_submission.py            class weights + band rule -> submission
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--force", action="store_true")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--trees", type=int, default=150)
ap.add_argument("--name", default="final")
a = ap.parse_args()
PY = [sys.executable, "-I"]
P = ROOT / "data/processed"
M = ROOT / "models"
FEATS_NOBAND = "features_v1.parquet,features_v2.parquet,features_v4.parquet"
FEATS = "features_v1.parquet,features_v2.parquet,features_v3.parquet,features_v4.parquet"


def step(name, outputs, cmd):
    if not a.force and all(Path(o).exists() for o in outputs):
        print(f"[skip] {name}")
        return
    t0 = time.time()
    print(f"[run ] {name}: {' '.join(str(c) for c in cmd[2:])}", flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)
    print(f"[done] {name} ({time.time() - t0:.0f}s)", flush=True)


for d in ("data/interim", "data/processed", "models", "submissions", "reports/eda/tables", "reports/eda/figures"):
    (ROOT / d).mkdir(parents=True, exist_ok=True)
step("interim", [ROOT / "data/interim/train.parquet"], PY + ["scripts/eda/00_load_and_overview.py", "."])
step("relational", [P / "relational_v0.parquet"], PY + ["scripts/eda/04_relational.py", "."])
step("base", [P / "base_v0.parquet", P / "ratios_v0.parquet"], PY + ["src/upi/build_base.py", "."])
for v in ("v1", "v2", "v4"):
    step(f"features_{v}", [P / f"features_{v}.parquet"], PY + [f"src/upi/features_{v}.py", "."])
step("stage1 (no bands)", [M / "s1/test_proba.npy"],
     PY + ["src/upi/run_oof.py", ".", "s1", "--features", FEATS_NOBAND, "--trees", str(a.trees), "--test"])
step("bands", [P / "features_v3.parquet"], PY + ["src/upi/bands.py", ".", "s1"])
step("oof raw", [M / "oof_raw/oof.npy"], PY + ["src/upi/run_oof.py", ".", "oof_raw", "--features", FEATS, "--trees", str(a.trees)])
step("clean labels", [M / "oof_raw/y_clean.npy"], PY + ["src/upi/clean_labels.py", ".", "oof_raw", "--tau-unrep", "0.85"])
step("final model", [M / "oof_clean/test_proba.npy"],
     PY + ["src/upi/run_oof.py", ".", "oof_clean", "--features", FEATS, "--labels", "models/oof_raw/y_clean.npy",
           "--trees", str(a.trees), "--seeds", str(a.seeds), "--test"])
step("submission", [ROOT / "submissions" / f"{a.name}.csv"],
     PY + ["src/upi/make_submission.py", ".", "oof_clean", "--labels", "models/oof_raw/y_clean.npy", "--name", a.name])
print("finished:", ROOT / "submissions" / f"{a.name}.csv")
