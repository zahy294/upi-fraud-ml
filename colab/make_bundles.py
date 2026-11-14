"""Build the archives a Colab session needs, and one notebook per plan. Run on the laptop from the project root:

    python -I colab/make_bundles.py [code] [raw] [cache] [notebooks]        (no argument = all four)

colab/dist/upi_code_<time>.zip   the code (small; rebuild and re-upload whenever the code changes)
colab/dist/upi_raw.zip           the five raw CSVs and the benchmark labels keyed by txn_id   (plan 'final')
colab/dist/upi_cache.zip         processed features and reference probabilities              (all other plans)
notebooks/colab_<plan>.ipynb     one notebook per plan; each needs its own Colab session
Upload the three archives to Google Drive, folder MyDrive/upi_colab/.
"""
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "colab/dist"
DIST.mkdir(parents=True, exist_ok=True)
what = set(sys.argv[1:]) or {"code", "raw", "cache", "notebooks"}


def add(z, f, arc=None):
    z.write(f, arc or Path(f).relative_to(ROOT).as_posix())


if "code" in what:
    out = DIST / f"upi_code_{time.strftime('%m%d_%H%M')}.zip"
    files = [ROOT / "run_all.py", ROOT / "requirements.txt", *sorted((ROOT / "src/upi").glob("*.py")), *sorted((ROOT / "colab").glob("*.py")),
             *[ROOT / "scripts/eda" / n for n in ("00_load_and_overview.py", "04_relational.py", "common.py")]]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            add(z, f)
    print(out.name, f"{out.stat().st_size / 1e3:.0f} KB, {len(files)} files")

if "raw" in what:
    base = pd.read_parquet(ROOT / "data/processed/base_v0.parquet", columns=["txn_id", "split"])
    ids = base.txn_id.values[(base.split == "train").values]
    y = np.load(ROOT / "models/oof_raw/y_clean.npy")
    assert len(ids) == len(y)
    stage = DIST / "y_clean_ids.parquet"
    pd.DataFrame({"txn_id": ids, "y": y.astype("int8")}).to_parquet(stage, index=False)
    out = DIST / "upi_raw.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted((ROOT / "data/raw").glob("*.csv")):
            add(z, f)
        add(z, stage, "bench/y_clean_ids.parquet")
    stage.unlink()
    print(out.name, f"{out.stat().st_size / 1e6:.0f} MB")

if "cache" in what:
    out = DIST / "upi_cache.zip"
    files = [ROOT / "data/processed" / f"{n}.parquet" for n in ("base_v0", "features_v1", "features_v2", "features_v3", "features_v4", "features_v5", "features_v6")]
    files += [ROOT / "data/processed/band_edges.json", ROOT / "models/oof_raw/y_clean.npy", ROOT / "models/s2_a/oof.npy",
              *sorted((ROOT / "models/oof_final_local").glob("*.npy"))]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for f in files:
            add(z, f)
    print(out.name, f"{out.stat().st_size / 1e6:.0f} MB")

if "notebooks" in what:
    PLANS = {"final": ("raw", "Whole pipeline from the raw CSVs, then the second stage. Writes the submission files. CPU-bound: pick the runtime with the most CPU cores."),
             "lgb": ("cache", "LightGBM variants: all legit rows, finer bins, wider and longer models. CPU-bound: pick the runtime with the most CPU cores and RAM."),
             "xgb": ("cache", "XGBoost on the GPU: more trees, all legit rows, deeper and shallower trees. Needs a GPU runtime."),
             "cat": ("cache", "CatBoost on the GPU. Needs a GPU runtime."),
             "styles": ("cache", "Different styles: detect-then-type specialist models and a neural network. A GPU helps the network.")}
    for plan, (data, blurb) in PLANS.items():
        setup = f'''PLAN = "{plan}"
import glob, os, shutil, zipfile
from google.colab import drive
drive.mount("/content/drive")
D = "/content/drive/MyDrive/upi_colab"
W = "/content/upi"
shutil.rmtree(W, ignore_errors=True)
os.makedirs(W)
def unzip(pattern):
    files = sorted(glob.glob(f"{{D}}/{{pattern}}"), key=os.path.getmtime)
    assert files, f"{{pattern}} not found in {{D}}: upload it to Google Drive, folder upi_colab"
    print("using", files[-1])
    zipfile.ZipFile(files[-1]).extractall(W)
unzip("upi_{data}*.zip")
unzip("upi_code*.zip")          # last, so the newest code wins
!nvidia-smi -L || echo "no GPU in this runtime"
!nproc; free -g | head -2'''
        cells = [("markdown", f"# UPI Fraud Rings on Colab: plan `{plan}`\n\n{blurb}\n\n"
                              f"Before running: `upi_code_*.zip` and `upi_{data}.zip` must be in Google Drive, folder `MyDrive/upi_colab/`. "
                              "Then **Runtime > Run all**. Results are copied to `MyDrive/upi_colab/out/" + plan + "/` after every fit. "
                              "**Disconnect the runtime when the last cell finishes**: compute units burn while it stays connected."),
                 ("code", setup),
                 ("code", '# Same library versions as the laptop. If the pinned install fails, the session continues with Colab\'s own versions.\n'
                          '!pip -q install -r /content/upi/requirements.txt catboost==1.2.10 || echo "pinned install failed: using the versions Colab ships"'),
                 ("code", "!cd /content/upi && python -I colab/driver.py $PLAN --out $D/out/$PLAN"),
                 ("code", "!cat /content/upi/results_$PLAN.md; ls -la $D/out/$PLAN")]
        nb = {"nbformat": 4, "nbformat_minor": 0, "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                                             "accelerator": "GPU" if plan in ("xgb", "cat", "styles") else "None"},
              "cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True), **({"outputs": [], "execution_count": None} if t == "code" else {})}
                        for t, s in cells]}
        (ROOT / "notebooks" / f"colab_{plan}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("notebooks:", ", ".join(f"colab_{p}.ipynb" for p in PLANS))
