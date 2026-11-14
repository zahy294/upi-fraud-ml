# Team v4 master integration

Integration date: 2026-11-14. User-reported public score: **0.5411** for
`v4_master_ensemble.csv`. Next target: **0.642**, not a predicted score.

- Starting `codex/target-070`: `093599fa0bf3738e984f9eb301109eea7d0c8607`.
- Friend's `siddharth/v4-master-ensemble`: `f875aa94999f2f95f0b91e478c7db0ccd61c35b6`.
- Merge into `codex/target-070`; no changes to the friend's branch, modelling
  or main. Existing project files and the local EDA image edit are preserved.
- `src/upi/features_v6.py` is byte-identical at the two Git revisions. The
  merge adds `features_stage2.py`, `scripts/run_oof_catboost.py`, and
  `scripts/build_v5_master.py` without replacing our pipeline APIs.

## Recorded v4 recipe

Blend probabilities in the fixed class order from `src/upi/common.py`:

- 0.35 `oof_clean`: LightGBM, v1–v5.
- 0.25 `oof_v6_lgb`: LightGBM with v6.
- 0.15 `oof_clean_xgb`: XGBoost, v1–v5.
- 0.10 `oof_v6_xgb`: XGBoost with v6.
- 0.15 `oof_clean_cat`: CatBoost, v1–v5.

The user's recipe fits `common.fit_weights` to pooled OOF probabilities and
`models/oof_raw/y_clean.npy` (cleaning threshold0.85), then applies the original
device/sender/receiver/chain band rules. These original rules are equivalent
to the current `_apply_rules` helper. This is **Stage1**, not the v5 Stage2
enhanced rules (which also add opposite-pair and chain-root overrides).

The imported v5 builder unconditionally loads Stage2 model arrays and creates
v5 files. It does not reproduce v4 by itself. The imported v6 module exposes
functions, not a CLI that builds `features_v6.parquet`. Exact original LGB/XGB
tree counts, seed counts and the v6 feature-generation command are not recorded
in this branch. Do not silently substitute defaults and call the result the
0.5411 baseline.

## Evaluation caveats found in source

`run_oof.py` and the CatBoost runner use leave-one-month-out fitting: July can
be predicted by models trained on August/September. Pooled OOF scores are not
forward chronological scores. Fitting decision weights and reporting their
score on the same pooled labels adds selection optimism. Cleaned labels are
pseudo-labels, not clean evaluation ground truth.

Stage2 combines OOF training probabilities and full-fit test probabilities in
one table. Its windows do not isolate prediction folds/months/splits, and
removing a row's own probability does not remove its labels' indirect influence
through a neighbour's model. Before assessing Stage2 as a prospective upgrade,
rebuild every outer fold's teachers, cleaning, bands, temporal encodings and
probability features without outer-fold labels. Do not use its quoted pooled
clean F1 as evidence for reaching0.642.

## Needed for exact reproduction

The local checkout has none of the five `models/<tag>` folders, the cleaning
array, shared `data/processed/features_v3.parquet` and `features_v5.parquet`, or
the exact submitted `v4_master_ensemble.csv`. Preserve the friend's complete
artifact bundle with its original `base_v0.parquet` row order and sample CSV.
Bare `.npy` files contain no transaction IDs; matching the final submitted CSV
is necessary, but cannot by itself certify training OOF row provenance.

Use `v4_master.py` for inventory, reproduction and bounded saved-probability
diagnostics. No fitting of models or competition uploads occurs in that tool.
The original source scripts remain available for reference; don't execute the
v5 builder expecting the v4 submission.

## Tuning sequence

1. Reproduce the exact submitted v4 labels; report SHA256 and ID-aligned
   differences. Stop tuning if the reproduction differs.
2. Compare the fixed blend, equal blend, five leave-one-member-out blends,
   and raw-label versus cleaned-label decision-weight fitting. Fit weights on
   Sep1–15, choose on Sep16–23 using original labels, freeze, and report Sep24–30
   with per-class precision/recall/F1/support. These are saved-OOF diagnostics
   subject to the upstream caveats, not certified leakage-free validation.
3. Refit decision weights on Sep1–23 for the frozen candidate, then report its
   later assessment once. The diagnostic tool does not generate a new tuned
   test submission; review results and rebuild fold-safe validation first.
4. After evidence identifies a useful direction, change one member's features
   or training settings at a time. User performs all model training locally.
   Retain the verified v4 file as the public baseline.

All implementation commits and pushes stay on `codex/target-070`.
