# Final checks, 14 Nov night

One-off scripts behind the "Update, 14 Nov, midnight" box in `docs/IMPLEMENTATION_PLAN.md`. They are exploratory and unpolished.

Run from an empty working folder, as `python -I <script> <project_root> [...]`:

- `01_build_raw_all.py <project_root> <out_folder>` writes `raw_all.parquet` (train and test rows with ID numbers and device class).
- `02_build_work_table.py <project_root>` reads `raw_all.parquet` from the current folder and writes `work.parquet` (adds out-of-fold predictions of `models/s2_a`, cleaned labels and the test predictions of `submissions/final_s2_v7edges.csv`).
- `03` to `11` and `14` read `work.parquet` from the current folder; `09`, `10`, `11` and `14` also take `<project_root>`.
- `12_band_singles_flagged_share.py <run_folder>` needs a folder holding `models/s1` with test probabilities (a from-scratch `run_all.py` run).
- `13_weight_sensitivity_and_blends.py <run_folder> <out_folder> "spec|multiplier|name" ...` scores blends and fraud-weight multipliers; a name of `-` writes no file.
- `15_trap_columns_in_october.py` and `16_account_history_across_months.py` (added 15 Nov) read `work.parquet` from the current folder and take no arguments.
- `17_october_edge_uncertainty.py` and `18_october_edge_rows.py <out_folder>` (added 15 Nov) run from a from-scratch run folder (they need `data/processed`, `models/s1` and `out_real/s2_typ_all.csv`, the 0.5472 file). They list, per October band edge, the rows whose band the data cannot decide; the edge probes in the main README come from them.
- `19_three_observations.py`, `20_qr_field_rule.py`, `21_qr_field_by_band.py` and `22_october_rare_states.py` (added 15 Nov) read `work.parquet` from the current folder. They test payment type, hour and amount format on all ordinary-looking rows, and establish that `qr_vpa_mismatch` is filled exactly when a merchant is paid by QR in legit rows; `src/upi/quirk_rules.py` uses that.
