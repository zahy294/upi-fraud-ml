"""Runs one plan of model fits on Colab (or any machine) and benchmarks every result on the fixed local benchmark.

    python -I colab/driver.py <plan> [--out DIR] [--smoke] [--only name1,name2]

Plans
  final    the whole pipeline from the raw CSVs (run_all.py), then the second stage; writes the submissions
  lgb      LightGBM variants the laptop is too small or slow for (all legit rows, finer bins, wider and longer models)
  xgb      XGBoost on the GPU
  cat      CatBoost on the GPU
  styles   detect-then-type specialists, and a neural network

'final' needs data/raw/*.csv. The other plans need data/processed/*.parquet and the reference probabilities
(models/s2_a, models/oof_final_local, models/oof_raw/y_clean.npy), which the cache archive provides.
Every result is scored alone and blended with the reference; results_<plan>.md is rewritten after each fit and
copied to --out together with the probabilities, so a disconnected session loses at most the fit in progress.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

ap = argparse.ArgumentParser()
ap.add_argument("plan", choices=["final", "lgb", "xgb", "cat", "styles"])
ap.add_argument("--out", default=None, help="folder that survives the session (Google Drive)")
ap.add_argument("--smoke", action="store_true", help="tiny models, to test the wiring")
ap.add_argument("--only", default=None, help="comma-separated experiment names")
a = ap.parse_args()
PY = [sys.executable, "-I"]
JOBS = str(os.cpu_count() or 4)
F5 = ",".join(f"features_v{i}.parquet" for i in (1, 2, 3, 4, 5))
F6 = F5 + ",features_v6.parquet"
FIXED = "models/oof_raw/y_clean.npy"
OUT = Path(a.out) if a.out else None
LOG = ROOT / f"colab_{a.plan}.log"
RES_MD, RES_JSON = ROOT / f"results_{a.plan}.md", ROOT / f"results_{a.plan}.json"
ROWS = []
GPU = shutil.which("nvidia-smi") is not None and subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
DEV = "cuda" if GPU else "cpu"
T0 = time.time()


def say(msg):
    line = f"[{(time.time() - T0) / 60:6.1f} min] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sh(name, cmd):
    """Run one step, streaming its output to the screen and the log. Returns minutes taken; raises if it fails."""
    t0 = time.time()
    say(f"run  {name}: {' '.join(str(c) for c in cmd[2:])}")
    with open(LOG, "a", encoding="utf-8") as f:
        p = subprocess.Popen([str(c) for c in cmd], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            f.write(line)
            if not line.startswith(("C:\\", "/usr", "  ")) or "Error" in line:      # skip warning noise on screen
                print("    " + line.rstrip(), flush=True)
        rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"{name} failed with exit code {rc}; see {LOG.name}")
    mins = (time.time() - t0) / 60
    say(f"done {name} ({mins:.1f} min)")
    return mins


_bench = None


def bench():
    global _bench
    if _bench is None:
        from bench import Bench
        ids = ROOT / "bench/y_clean_ids.parquet"
        _bench = Bench(ROOT, ids if (a.plan == "final" and ids.exists()) else None)
    return _bench


def record(name, spec, minutes=0.0, note=""):
    try:
        r = bench().score_spec(spec)
    except Exception as e:                                           # a failed score must not stop the session
        say(f"could not score {spec}: {e}")
        return None
    row = dict(name=name, spec=spec, pooled=round(r["pooled"], 5), pooled_raw=round(r["pooled_raw"], 5), nested=round(r["nested"], 5),
               nested_raw=round(r["nested_raw"], 5), minutes=round(minutes, 1), note=note)
    ROWS.append(row)
    say(f"SCORE {name:<28} pooled {row['pooled']:.5f}  nested {row['nested']:.5f}  (raw labels {row['pooled_raw']:.5f} / {row['nested_raw']:.5f})  {note}")
    lines = [f"# Results: plan `{a.plan}`{' (smoke test, tiny models)' if a.smoke else ''}", "",
             "Fixed benchmark: leave-one-month-out macro-F1 against the cleaned benchmark labels, band rules applied.",
             "Laptop reference: one-stage model 0.5596 pooled / 0.5589 nested; second stage `s2_a` 0.5627 / 0.5615.", "",
             "| name | spec | pooled | nested | pooled raw | nested raw | minutes | note |", "|---|---|---:|---:|---:|---:|---:|---|"]
    lines += [f"| {x['name']} | `{x['spec']}` | {x['pooled']:.5f} | {x['nested']:.5f} | {x['pooled_raw']:.5f} | {x['nested_raw']:.5f} | {x['minutes']} | {x['note']} |" for x in ROWS]
    RES_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    RES_JSON.write_text(json.dumps(ROWS, indent=1), encoding="utf-8")
    sync()
    return row


def sync(tags=()):
    """Copy results, log, submissions and the probabilities of `tags` to the folder that outlives the session."""
    if OUT is None:
        return
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        for f in [RES_MD, RES_JSON, LOG, ROOT / "data/processed/band_edges.json", *sorted((ROOT / "submissions").glob("*.csv"))]:
            if f.exists():
                shutil.copy2(f, OUT / f.name)
        for t in tags:
            d = ROOT / "models" / t
            if d.exists():
                (OUT / "models" / t).mkdir(parents=True, exist_ok=True)
                for f in d.glob("*.npy"):
                    shutil.copy2(f, OUT / "models" / t / f.name)
    except Exception as e:
        say(f"sync to {OUT} failed: {e}")


def tiny(args):
    return args + ["--trees", "5", "--legit-frac", "0.05"] if a.smoke else args


def experiment(name, script, args, ref=None, labels=FIXED, feats=F6):
    """Fit one variant (out-of-fold only), score it alone and blended with the reference."""
    cmd = PY + [f"src/upi/{script}", ".", name, "--features", feats, "--labels", labels, "--jobs", JOBS] + (tiny(args) if script == "run_oof.py" else args)
    try:
        mins = sh(name, cmd)
    except Exception as e:
        say(f"FAILED {name}: {e}")
        return None
    row = record(name, name, mins)
    sync([name])
    if ref and row:
        record(f"{ref} + {name}", f"{ref}+{name}", note="blend")
    return row


# ------------------------------------------------------------------ plans
def plan_final():
    seeds, trees = ("1", "5") if a.smoke else ("3", "150")
    mins = sh("pipeline", PY + ["run_all.py", "--force", "--name", "s2", "--seeds", seeds, "--trees", trees, "--jobs", JOBS])   # writes s2_base.csv, s2.csv and s2_typ_all.csv
    record("base (one stage)", "oof_clean")
    sync(["oof_clean", "s1"])
    record("second stage", "s2", mins)
    record("base + second stage", "oof_clean+s2", note="blend")
    sync(["s2"])
    record("s2_typ_all", "s2_typ_all")                               # detect-then-type, all types: fitted by run_all.py
    sync(["s2_typ_all"])
    if not a.smoke:                                                  # the per-family variant, with test predictions
        try:
            m3 = sh("s2_typ_fam", PY + ["src/upi/exp_typing.py", ".", "s2_typ_fam", "--detector", "s2", "--features", F6, "--labels", FIXED,
                                        "--mode", "family", "--jobs", JOBS, "--test"])
            sh("submission s2_typ_fam", PY + ["src/upi/make_submission.py", ".", "s2_typ_fam", "--labels", FIXED, "--name", "s2_typ_fam"])
            record("s2_typ_fam", "s2_typ_fam", m3)
            sync(["s2_typ_fam"])
        except Exception as e:
            say(f"FAILED s2_typ_fam: {e}")
    if OUT is not None and not a.smoke:                              # features and probabilities for later sessions
        say("writing the cache archive for other sessions")
        OUT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(OUT / "upi_cache_final.zip", "w", zipfile.ZIP_STORED) as z:
            for f in [*sorted((ROOT / "data/processed").glob("*.parquet")), ROOT / "data/processed/band_edges.json", ROOT / FIXED,
                      *sorted((ROOT / "models/oof_clean").glob("*.npy")), *sorted((ROOT / "models/s2").glob("*.npy"))]:
                if f.name not in ("relational_v0.parquet", "ratios_v0.parquet"):
                    z.write(f, f.relative_to(ROOT).as_posix())


EXPERIMENTS = {
    "lgb": [("lgb_ref", "run_oof.py", []),
            ("lgb_bin1023", "run_oof.py", ["--max-bin", "1023"]),
            ("lgb_leg100", "run_oof.py", ["--legit-frac", "1.0"]),
            ("lgb_wide", "run_oof.py", ["--leaves", "255", "--min-child", "10"]),
            ("lgb_slow", "run_oof.py", ["--trees", "500", "--lr", "0.02"]),
            ("lgb_big", "run_oof.py", ["--legit-frac", "0.5", "--trees", "400", "--lr", "0.03", "--leaves", "127"])],
    "xgb": [("xgb_ref", "run_oof.py", ["--model", "xgb", "--device", DEV]),
            ("xgb_600", "run_oof.py", ["--model", "xgb", "--device", DEV, "--trees", "600", "--lr", "0.03"]),
            ("xgb_leg100", "run_oof.py", ["--model", "xgb", "--device", DEV, "--trees", "600", "--lr", "0.03", "--legit-frac", "1.0"]),
            ("xgb_d10", "run_oof.py", ["--model", "xgb", "--device", DEV, "--trees", "400", "--lr", "0.04", "--depth", "10"]),
            ("xgb_d6", "run_oof.py", ["--model", "xgb", "--device", DEV, "--trees", "1000", "--lr", "0.03", "--depth", "6"]),
            ("xgb_bin1024", "run_oof.py", ["--model", "xgb", "--device", DEV, "--trees", "600", "--lr", "0.03", "--max-bin", "1024"])],
    "cat": [("cat_ref", "run_oof.py", ["--model", "cat", "--device", DEV, "--trees", "1000", "--lr", "0.08"]),
            ("cat_leg100", "run_oof.py", ["--model", "cat", "--device", DEV, "--trees", "1000", "--lr", "0.08", "--legit-frac", "1.0"]),
            ("cat_d10", "run_oof.py", ["--model", "cat", "--device", DEV, "--trees", "1000", "--lr", "0.08", "--depth", "10"]),
            ("cat_long", "run_oof.py", ["--model", "cat", "--device", DEV, "--trees", "3000", "--lr", "0.04"])],
    "styles": [("typ_fam", "exp_typing.py", ["--detector", "s2_a", "--mode", "family"]),
               ("typ_all", "exp_typing.py", ["--detector", "s2_a", "--mode", "all"]),
               ("typ_fam_big", "exp_typing.py", ["--detector", "s2_a", "--mode", "family", "--leaves", "63", "--trees", "600", "--lr", "0.02"]),
               ("typ_fam_small", "exp_typing.py", ["--detector", "s2_a", "--mode", "family", "--leaves", "7", "--trees", "200", "--lr", "0.05", "--min-child", "30"]),
               ("mlp", "exp_mlp.py", ["--device", DEV])],
}


def plan_experiments():
    record("reference: one stage (laptop)", "oof_final_local")
    ref = record("reference: second stage (laptop)", "s2_a")
    todo = EXPERIMENTS[a.plan]
    if a.only:
        todo = [e for e in todo if e[0] in a.only.split(",")]
    if a.smoke:
        todo = todo[:2]
    for name, script, args in todo:
        if not (ROOT / "src/upi" / script).exists():
            say(f"skip {name}: {script} is not in this code archive")
            continue
        experiment(name, script, args, ref="s2_a")
    blends = [r for r in ROWS if r["note"] == "blend"]
    if ref and blends:
        best = max(blends, key=lambda r: r["nested"])
        gain = best["nested"] - ref["nested"]
        say(f"best blend: {best['spec']} nested {best['nested']:.5f} ({gain:+.5f} against the reference alone)")
        if gain >= 0.001 and not a.smoke:                             # worth test predictions: refit that variant with three seeds
            name = best["spec"].split("+")[1]
            script, args = next((s, g) for n, s, g in EXPERIMENTS[a.plan] if n == name)
            extra = ["--seeds", "3", "--test"] if script != "exp_mlp.py" else ["--test"]
            try:
                sh(f"refit {name} with test predictions", PY + [f"src/upi/{script}", ".", name + "_full", "--features", F6, "--labels", FIXED, "--jobs", JOBS] + args + extra)
                record(name + "_full", name + "_full", note="three seeds, has test predictions")
                record(f"s2_a + {name}_full", f"s2_a+{name}_full", note="blend")
                sync([name + "_full"])
            except Exception as e:
                say(f"FAILED refit {name}: {e}")
        else:
            say("no variant adds 0.001 nested to the reference; nothing refitted")


say(f"plan {a.plan} | python {sys.version.split()[0]} | cpus {JOBS} | gpu {'yes' if GPU else 'no'} | smoke {a.smoke}")
try:
    import lightgbm, numpy, pandas, xgboost  # noqa: E401
    say(f"numpy {numpy.__version__}, pandas {pandas.__version__}, lightgbm {lightgbm.__version__}, xgboost {xgboost.__version__}")
except Exception as e:
    say(f"version check: {e}")
try:
    plan_final() if a.plan == "final" else plan_experiments()
finally:
    sync()
    say(f"finished plan {a.plan}. Results: {RES_MD.name}" + (f" (copied to {OUT})" if OUT else ""))
    if RES_MD.exists():
        print("\n" + RES_MD.read_text(encoding="utf-8"))
