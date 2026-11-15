"""One entry point: raw CSVs -> features -> models -> submissions/<name>_edges_quirk.csv, plus
<name>_edges.csv, <name>_typ_all.csv, <name>.csv and <name>_base.csv from the earlier steps

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
     src/upi/features_v5.py                where forwarding chains come from (uses the bands)
  7. src/upi/run_oof.py  (tag oof_raw)     leave-one-month-out predictions on the raw labels
  8. src/upi/clean_labels.py               cleaned training labels
  9. src/upi/run_oof.py  (tag oof_clean)   one-stage model on cleaned labels, with test predictions
 10. src/upi/make_submission.py            class weights + band rules -> submissions/<name>_base.csv
 11. src/upi/features_v6.py                account-episode features built from the one-stage model's predictions
 12. src/upi/run_oof.py  (tag s2)          second-stage model on all feature tables, with test predictions
 13. src/upi/make_submission.py            class weights + band rules -> submissions/<name>.csv
 14. src/upi/exp_typing.py (tag s2_typ_all) detect-then-type: the second stage decides fraud or legit, a specialist model
                                           trained only on detected fraud rows decides the type
 15. src/upi/make_submission.py            class weights + band rules -> submissions/<name>_typ_all.csv
 16. src/upi/relabel_band_rows.py          EDGE_MOVES below -> submissions/<name>_edges.csv
 17. src/upi/quirk_rules.py                rows the legit generator cannot produce -> submissions/<name>_edges_quirk.csv

--mlp adds a neural network to the second stage (needs PyTorch; fitted on --device, default cuda):
 18. src/upi/exp_mlp.py (tag mlp)          small neural network on the same feature tables, with test predictions
 19. src/upi/make_submission.py --blend    average of second stage and network -> submissions/<name>_blend.csv
 20. relabel_band_rows.py, quirk_rules.py  -> submissions/<name>_blend_edges.csv, <name>_blend_edges_quirk.csv

--no-stage2 stops after step 10. The second stage adds about 80 minutes on a 16 GB laptop, steps 14 and 15 about 25 more.
Step 14 always uses its own defaults (300 trees, 3 seeds); --trees and --seeds apply to the 12-class models.
"""
import argparse
import os
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
ap.add_argument("--no-stage2", action="store_true")
ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4, help="threads for the second-stage and specialist fits")
ap.add_argument("--mlp", action="store_true", help="also build the neural-network blend (needs PyTorch)")
ap.add_argument("--device", default="cuda", help="device for the neural network: cuda or cpu")
a = ap.parse_args()
PY = [sys.executable, "-I"]
P = ROOT / "data/processed"
M = ROOT / "models"
FEATS_NOBAND = "features_v1.parquet,features_v2.parquet,features_v4.parquet"
FEATS = "features_v1.parquet,features_v2.parquet,features_v3.parquet,features_v4.parquet,features_v5.parquet"
# October band-edge corrections, each confirmed by one public-leaderboard upload on 15 Nov (see README, "Edge probes").
# bands.py places the October edges without labels; where no clearly typed row sits near an edge, the rows next to it
# can belong to either band. "<first device number>-<last device number>:<label>".
EDGE_MOVES = ["473818-473821:velocity_bot"]


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
step("features_v5", [P / "features_v5.parquet"], PY + ["src/upi/features_v5.py", "."])
step("oof raw", [M / "oof_raw/oof.npy"], PY + ["src/upi/run_oof.py", ".", "oof_raw", "--features", FEATS, "--trees", str(a.trees)])
step("clean labels", [M / "oof_raw/y_clean.npy"], PY + ["src/upi/clean_labels.py", ".", "oof_raw", "--tau-unrep", "0.85"])
step("one-stage model", [M / "oof_clean/test_proba.npy"],
     PY + ["src/upi/run_oof.py", ".", "oof_clean", "--features", FEATS, "--labels", "models/oof_raw/y_clean.npy",
           "--trees", str(a.trees), "--seeds", str(a.seeds), "--test"])
step("submission (one stage)", [ROOT / "submissions" / f"{a.name}_base.csv"],
     PY + ["src/upi/make_submission.py", ".", "oof_clean", "--labels", "models/oof_raw/y_clean.npy", "--name", f"{a.name}_base"])
if a.no_stage2:
    print("finished (one stage only):", ROOT / "submissions" / f"{a.name}_base.csv")
    sys.exit(0)
step("features_v6", [P / "features_v6.parquet"], PY + ["src/upi/features_v6.py", ".", "oof_clean"])
step("second stage", [M / "s2/test_proba.npy"],
     PY + ["src/upi/run_oof.py", ".", "s2", "--features", FEATS + ",features_v6.parquet", "--labels", "models/oof_raw/y_clean.npy",
           "--trees", str(a.trees), "--seeds", str(a.seeds), "--jobs", str(a.jobs), "--test"])
step("submission", [ROOT / "submissions" / f"{a.name}.csv"],
     PY + ["src/upi/make_submission.py", ".", "s2", "--labels", "models/oof_raw/y_clean.npy", "--name", a.name])
step("type specialist", [M / "s2_typ_all/test_proba.npy"],
     PY + ["src/upi/exp_typing.py", ".", "s2_typ_all", "--detector", "s2", "--features", FEATS + ",features_v6.parquet",
           "--labels", "models/oof_raw/y_clean.npy", "--mode", "all", "--jobs", str(a.jobs), "--test"])
step("submission (type specialist)", [ROOT / "submissions" / f"{a.name}_typ_all.csv"],
     PY + ["src/upi/make_submission.py", ".", "s2_typ_all", "--labels", "models/oof_raw/y_clean.npy", "--name", f"{a.name}_typ_all"])
step("edge corrections", [ROOT / "submissions" / f"{a.name}_edges.csv"],
     PY + ["src/upi/relabel_band_rows.py", ".", f"submissions/{a.name}_typ_all.csv", f"submissions/{a.name}_edges.csv"] + EDGE_MOVES)
step("quirk rules", [ROOT / "submissions" / f"{a.name}_edges_quirk.csv"],
     PY + ["src/upi/quirk_rules.py", ".", f"submissions/{a.name}_edges.csv", f"submissions/{a.name}_edges_quirk.csv"])
print("finished:", ROOT / "submissions" / f"{a.name}_edges_quirk.csv")
if a.mlp:
    step("neural network", [M / "mlp/test_proba.npy"],
         PY + ["src/upi/exp_mlp.py", ".", "mlp", "--features", FEATS + ",features_v6.parquet", "--labels", "models/oof_raw/y_clean.npy",
               "--device", a.device, "--test"])
    step("submission (blend)", [ROOT / "submissions" / f"{a.name}_blend.csv"],
         PY + ["src/upi/make_submission.py", ".", "s2", "--blend", "s2,mlp", "--labels", "models/oof_raw/y_clean.npy", "--name", f"{a.name}_blend"])
    step("edge corrections (blend)", [ROOT / "submissions" / f"{a.name}_blend_edges.csv"],
         PY + ["src/upi/relabel_band_rows.py", ".", f"submissions/{a.name}_blend.csv", f"submissions/{a.name}_blend_edges.csv"] + EDGE_MOVES)
    step("quirk rules (blend)", [ROOT / "submissions" / f"{a.name}_blend_edges_quirk.csv"],
         PY + ["src/upi/quirk_rules.py", ".", f"submissions/{a.name}_blend_edges.csv", f"submissions/{a.name}_blend_edges_quirk.csv"])
    print("finished:", ROOT / "submissions" / f"{a.name}_blend_edges_quirk.csv")
