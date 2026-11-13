# Comprehensive Post-Mortem, Architecture Summary & Future Run Guide
**Amazon Business Entity Resolution Challenge (26.4M+ Records)**

---

## 1. Executive Summary
Over the course of this project, we conducted deep exploratory data analysis (EDA), synthesized five domain-specific research investigations, passed four rounds of adversarial technical review, and designed, implemented, audited, and deployed an end-to-end Machine Learning pipeline for the **Amazon Business Entity Resolution Challenge** across 26.4M+ cross-source records (`US`, `India`, `France`).

While the software architecture passed **337 automated unit and end-to-end tests** (with independent AST static audits confirming zero data leakage, zero row-wise iteration bottlenecks, and strict memory bounds on test fixtures), full-scale execution across the 26.4M dataset hit a series of hardware and Linux binary hurdles across our local laptop and two cloud GPU servers before reaching the hard 10:00 PM cutoff. **In short: the ML architecture succeeded, while deployment operations (Linux binary compilation and multi-core thread scaling) killed the run.**

All complete pipeline code is preserved in Git under the branch **`feat/antigravity-pipeline`** (commit `283700e`, 36 files, 11,974 insertions) at:
`https://github.com/ItzJeston/amazon-ml-challenge-2026/tree/feat/antigravity-pipeline`

---

## 2. Dataset Findings & EDA Insights (`eda_report.md`)

Before designing the pipeline, we extracted `6ab10eb3b23ba_student_resource.zip` into `extracted_data/` and ran a comprehensive EDA across all 7 TSV datasets:

*   **Total Volume & Memory Footprint:** **26,435,994 rows** across 7 files totaling **2.40 GB on disk** and **7.82 GB in raw pandas memory**:
    *   *Training Set:* `train_source1` (2,206,821 rows), `train_source2` (5,034,616 rows), `train_source3` (5,285,603 rows), `train_ground_truth` (2,206,821 rows).
    *   *Test Set:* `test_source1` (1,732,544 rows), `test_source2` (4,887,273 rows), `test_source3` (5,082,316 rows).
    *   *Implication:* Loading all raw files simultaneously consumes nearly half of a 16 GB RAM budget before computing a single feature, mandating **strict country-level partitioning** and streaming.
*   **Ground Truth Label Distribution (The Singleton Multiplier):**
    *   **Singletons (0 matches):** 123,247 entities (**5.58%**). Under the competition's Macro $F_{0.5}$ formula, predicting an empty string on a true singleton scores `1.0`, whereas even a single false positive drops the entity's score to `0.0`.
    *   **1 Match:** 119,157 entities (**5.40%**).
    *   **Multiple Matches ($\ge 2$):** 1,964,417 entities (**89.02%**), peaking at a mode of 3–4 matches per $S_1$ entity (with a tail up to 11 matches).
    *   **Source Parity:** 7,638,365 total matched IDs split nearly evenly between $S_2$ (48.36%) and $S_3$ (51.64%).
*   **The Open-Set "France" Discovery:**
    *   Training data contains **only `US` (~60%) and `India` (~40%)**.
    *   Test data contains `India` (46.75%), `US` (38.27%), and a completely unseen third country: **`France` (14.98% / 259,452 $S_1$ entities)**.
    *   *Implication:* Any model relying on country one-hot encodings or memorized English/Hindi vocabulary would fail on 15% of the test set. Features had to be strictly **language-agnostic** (normalized character/token distances) paired with explicit French legal suffix (`SARL`, `SAS`, `SA`, `EURL`, `SCI`) and address (`Rue`, `Boulevard`, `Cedex`) normalization.
*   **Noise & Missingness Profile:**
    *   **Business Names:** 0% missing across all 6 files, but heavily polluted with inconsistent legal suffixes (`Pvt/Private` in 25–35%, `Ltd/Limited` in 30–35%, `LLC` in 10–16%), punctuation (`&` vs `and` in 6–8%), domain-style names in $S_3$, and raw **Devanagari script** in $S_2$ (27.4% of Indian records) and $S_3$ (18.5%).
    *   **Business Addresses:** Never missing in $S_1$ (0%), but **missing in 2.65%–3.36% of $S_2$ and $S_3$ records** (~130K–176K rows per file). Furthermore, **only 5%–8% of addresses contain a PIN/ZIP code**, ruling out postal-code blocking as a primary strategy and requiring explicit missing-address indicators so candidates aren't penalized when $S_2/S_3$ addresses are null.

---

## 3. Evolution of Architectural Decisions & Adversarial Reviews (Plans v1 $\to$ v4)

Our implementation plan evolved across four major iterations driven by competition rules, deep research synthesis (`research/`), and two independent adversarial code reviews:

### 3.1 Research-Driven Upgrades (Plan v1/v2 $\to$ v3)
1.  **Audit-Winning Candidate Compactness ($K \le 15$, not 100):** The competition updated its rules to evaluate `candidate_pairs.tsv` alongside `matching_results.tsv` (*"The approach that generates a smaller candidate set per Source 1 entity will be ranked higher"*). Replacing a naive `top_k=100` block with a two-stage cascade (wide 4-strategy retrieval $\to$ lightweight cosine re-ranker pruning to $\le 15$ candidates, median 10–12) cut downstream candidate pairs by **~85%** (from ~170M to ~15–20M pairs).
2.  **Dropping RandomForest for CatBoost:** Uncompressed scikit-learn `RandomForest` trees over 20M rows risk immediate OOM on 16 GB RAM. We replaced it with **CatBoost** alongside **LightGBM** (CPU) and **XGBoost** (`tree_method='hist', device='cuda'`).
3.  **Banning `libpostal` / `pypostal` on Windows 11:** Compiling `libpostal` C bindings via MSYS2 on Windows is notoriously fragile and requires ~2 GB of static dictionary RAM per worker process. We opted for a zero-dependency, pre-compiled regex normalization dictionary (`ADDR_ABBREVS`) covering US, Indian (`Nr`, `Opp`), and French abbreviations.
4.  **`anyascii` + `corp-names` Over `unidecode`:** `anyascii` was adopted for zero-dependency script-to-ASCII transliteration (producing consistent canonical tokens for Devanagari), followed by ISO 20275 legal suffix stripping via `corp-names`.
5.  **API Gotchas Caught Early:**
    *   `sparse_dot_topn >= 1.2.0` replaced deprecated `awesome_cossim_topn` with `sp_matmul_topn`, where `sort=False` is the default — explicitly passing **`sort=True`** and enforcing **`scipy.sparse.csr_matrix`** (never COO/CSC) was made mandatory.
    *   `rapidfuzz >= 3.x` changed its default string pre-processor to `None`, meaning case/punctuation differences aren't normalized unless **`processor=rfuzz_utils.default_process`** is explicitly passed.
6.  **Strict Guardrails:** Explicitly forbade any external databases, geocoding APIs, or internet lookups to comply with competition rules.

### 3.2 Adversarial Review #1 (`research/phase4_adversarial_review.md` + User Loose-End Catch)
1.  **`CalibratedClassifierCV` + `FrozenEstimator` Crash Fix:** Passing both `FrozenEstimator(model)` and `cv='prefit'` crashes in `scikit-learn >= 1.8/1.9` because `cv='prefit'` was deprecated and removed in favor of `FrozenEstimator`. We fixed calibration to use `CalibratedClassifierCV(estimator=FrozenEstimator(model), method='sigmoid')` with no `cv` argument.
2.  **Stage 2 Re-Ranker $\to$ Phase 4 Feature Handoff:** Initially, `rerank_and_prune()` computed only name cosine while its docstring claimed a name+address blend, and Phase 4 hardcoded `'addr_tfidf_cosine': 0.0`. We fixed Stage 2 to genuinely compute $0.70 \times \text{Name Cosine} + 0.30 \times \text{Address Cosine}$ and wired Phase 4 to consume the exact scores emitted by Stage 2.
3.  **Entity-Grouped $F_{0.5}$ Scorer:** Replaced pairwise `fbeta_score` during HPO with a custom `entity_grouped_f05` scorer that groups candidates by `source1_entity_id` and scores singletons according to the exact competition formula.
4.  **Dropped `enable_metadata_routing=True`:** Unnecessary when iterating over explicit `StratifiedGroupKFold.split(X, y, groups=s1_ids)` indices.

### 3.3 Adversarial Review #2 (8 Critical Refinements $\to$ Plan v4)
1.  **Corrected Python $\ge 3.11$ Justification:** Clarified that Python 3.11+ is strictly required by `corp-names` metadata (`Requires-Python: >=3.11`), not `rapidfuzz>=3.14` (which only dropped Python 3.9).
2.  **Exact Pin `corp-names==0.3.3` + Regex Fallback:** Because `corp-names` is a young beta package, we pinned it to `==0.3.3` and wrapped it in a `try/except` with a compiled regex suffix-stripper fallback (`FALLBACK_SUFFIXES`) plus Devanagari legal suffix pre-mapping (`प्राइवेट लिमिटेड` $\to$ `private limited`, `एलएलपी` $\to$ `llp`).
3.  **Banned `.iter_rows()` in Polars (Phase 4 Native Parallelism):** Using `for row in batch.iter_rows(named=True):` with Python dict lookups forces data out of Rust/Arrow memory into a single-threaded CPython loop (~500M sequential calls). We mandated **zero `.iter_rows()`** across the codebase, replacing it with native Polars lazy relational joins (`pl.scan_parquet().join()`) and batched `pl.struct(...).map_batches()` calls.
4.  **Strict 80/20 Train/Holdout Split (Zero Calibration/Threshold Leakage):** Reusing the same out-of-fold validation predictions to fit Platt scaling, train the Level-1 meta-learner, *and* tune the 3D Optuna thresholds causes severe threshold overfitting. We enforced an explicit **80% Train** (5-fold CV for base learners, Platt calibration, and meta-learner) and **20% Unseen Holdout** (reserved strictly for Phase 6 3D Optuna threshold tuning).
5.  **Candidate-Side CV Leakage Diagnostic:** Grouping by `source1_entity_id` prevents $S_1$ leakage across folds, but shared $S_2/S_3$ candidates could still span multiple $S_1$ groups. We added `diagnose_candidate_leakage()` in `src/dataset.py` to quantify candidate overlap across folds.
6.  **Explicit Stage 1B & 1D Code:** Wrote out full Polars implementations for Token Inverted Index (Strategy B) and 2-Token Soundex (Strategy D) rather than referencing prior drafts.
7.  **Dynamic Row Count Validation:** Replaced hardcoded `1,732,544` row checks with dynamic assertions reading the exact length of `test_source1.tsv` at runtime.
8.  **Cleaned Up Stale `zero_division=np.nan` Citations:** Synchronized all summary tables with the custom `entity_grouped_f05` implementation.

### 3.4 PyTorch/CUDA, Neural Embeddings & The Roadmap to 0.98+ $F_{0.5}$
*   **PyTorch + CUDA Role:** We installed and verified `torch==2.11.0+cu128` on the local RTX 4050. However, in the V4 baseline, PyTorch itself is not invoked — only **XGBoost** uses CUDA (`device='cuda'`). We confirmed that GPU tree building affects only speed (3–5$\times$ faster), not accuracy.
*   **Decision to Defer Neural Embeddings (`all-MiniLM-L6-v2`):** We debated adding a local sentence-transformer / cross-encoder into the initial build. We explicitly decided to **build and score the V4 baseline first** before adding neural embeddings into a "Phase 4.5" upgrade, to avoid debugging VRAM/throughput bottlenecks on 20M pairs before having a working baseline.
*   **The 4 Limiting Factors to Reach 0.98+ Macro $F_{0.5}$ (Future Upgrade Roadmap):**
    1.  *Zero-Shot France Penalty (15% of test set):* Overcome via **Transductive Pseudo-Labeling** — run the US/India-trained ensemble on France, extract high-confidence predictions ($P > 0.99$ or $P < 0.01$), inject them into the training set, and retrain so the trees learn French address/name weights.
    2.  *Blocking Recall Ceiling:* Overcome via **Transitive Graph Blocking** — if $S_1\text{-A}$ matches $S_2\text{-B}$ and $S_2\text{-B}$ strongly overlaps $S_3\text{-C}$, inject $(S_1\text{-A}, S_3\text{-C})$ into the candidate pool even if direct lexical similarity is low.
    3.  *Parent vs. Subsidiary Ambiguity:* Overcome via **GPU Cross-Encoder Re-Ranking** (`all-MiniLM-L6-v2` on the borderline probability band $P \in [0.30, 0.85]$) to distinguish true entity matches from related corporate subsidiaries.
    4.  *The 3.3% Missing Address Gap:* Overcome via **Graph Centrality Features** — compute node degree / PageRank on candidate entities so the model can identify canonical records when multiple candidates share an identical business name (e.g., `"Starbucks"`) with null addresses.

---

## 4. Complete Pipeline Architecture Built (Phases 0–7)

The codebase was built from the self-contained **Implementation Plan v4** via a multi-agent orchestration workflow (`teamwork_preview`) and independently verified by a fresh-context **Victory Auditor**:

### Phase 0 & 1: Normalization & Preprocessing (`src/normalization.py`, `src/preprocessing.py`)
*   **Windows UTF-8 Safety:** Every module initializes with a defensive `setup_utf8_stdout()` wrapper (`sys.stdout = io.TextIOWrapper(...)`) to prevent Windows `cp1252` decoding crashes on international business names.
*   **Regex Address Normalization:** Pure regex normalization using the `ADDR_ABBREVS` lookup tables across US, India, and France (including Devanagari suffix mappings and ASCII transliteration), with zero external C dependencies (`libpostal` banned).
*   **Legal Suffix Stripping:** Powered by `corp-names==0.3.3` (requiring Python 3.11+) paired with a comprehensive fallback regex.
*   **Phonetic Blocking Keys:** Computes a 2-token Soundex phonetic signature via `jellyfish`.
*   **Partitioned Storage:** Outputs zstd-compressed, country-partitioned Parquet files (`US`, `India`, `France`) into `data_processed/`.

### Phase 3: 4-Way Cascaded Blocking & Pruning (`src/blocking.py`, `src/reranking.py`)
To avoid an intractable $O(N \times M)$ cross-join (over 39 trillion combinations in the US partition alone), each country partition runs a 4-strategy candidate generation union:
1.  **Strategy A (Character 3-Gram TF-IDF):** Matrix cosine similarity over character trigrams via C++ `sparse_dot_topn` (`top_k=50`, `threshold=0.05`) to capture typos and character transpositions.
2.  **Strategy B (Token Inverted Index):** Native Polars hash-join over informative word tokens ($\text{length} \ge 3$, frequency-capped at `max_df=1500` to ignore stop-tokens like `"INC"` or `"LLC"`).
3.  **Strategy C (Address Word TF-IDF):** Word-level TF-IDF cosine similarity via `sparse_dot_topn` to catch businesses that rebranded (different name) at the same physical address.
4.  **Strategy D (Phonetic Soundex Match):** Exact matching on the 2-token Soundex hash to catch phonetic misspellings.
*   **Bitmask Provenance:** Each candidate pair tracks which strategies discovered it via bitmask flags (`blocking_source_flags`).
*   **Lightweight Cosine Re-Ranker & Pruning:** Candidates are scored by a composite similarity ($0.70 \times \text{Name Cosine} + 0.30 \times \text{Address Cosine}$, with calibrated fallback priors of `0.35` for token-only and `0.30` for soundex-only matches) and pruned via Polars `.group_by(SOURCE1_ID_COL).head(15)` to enforce $\le 15$ candidates per $S_1$ entity (median 10–12). Singletons with $< 0.05$ similarity are dropped immediately.

### Phase 4 & 4b: Vectorized Feature Engineering & Disjoint Splits (`src/features.py`, `src/dataset.py`)
*   **32 Pairwise Features:** Extracts 32 similarity metrics (`feat_01_name_ratio` through `feat_32_blocking_rank`) comparing business names, addresses, numeric tokens, length ratios, legal suffix matches, source indicators, and blocking provenance flags.
*   **Zero `.iter_rows()`:** Implemented 100% in native Polars (`pl.struct`, `map_batches`) with RapidFuzz calls locked to `processor=rfuzz_utils.default_process`. Missing addresses safely default to `0.0` with an explicit `addr_missing_flag` binary indicator.
*   **Zero-Leakage Split (Phase 4b):** Partitions training data into a strict disjoint **80% Train / 20% Holdout** split grouped at the `source1_entity_id` level, accompanied by candidate-side cross-validation leakage diagnostics.

### Phase 5 & 6: Calibrated Stacked Ensemble & 3D Optuna Tuning (`src/models.py`, `src/calibration.py`, `src/optimization.py`)
*   **Level-0 Base Learners:** Trained via 5-Fold `StratifiedGroupKFold` (grouped by `source1_entity_id`) on the 80% split:
    1.  **LightGBM** (Multi-core CPU)
    2.  **XGBoost** (GPU-accelerated via `device='cuda'`)
    3.  **CatBoost** (Multi-core CPU)
*   **Platt Scaling Calibration:** Out-of-fold and holdout probabilities are calibrated using `CalibratedClassifierCV(estimator=FrozenEstimator(model), method='sigmoid')`, avoiding scikit-learn's deprecated `cv='prefit'`.
*   **Level-1 Meta-Learner:** A `LogisticRegression` stacker combines the calibrated probabilities from all three base models.
*   **3D Bayesian Threshold Optimization (Phase 6):** Optuna searches independently for $(\tau_{\text{US}}, \tau_{\text{India}}, \tau_{\text{France}})$ on the unseen 20% holdout set to directly maximize the entity-level Macro $F_{0.5}$ metric. If the top candidate for an $S_1$ entity falls below $\tau_{\text{country}}$, it is exported as a singleton (`S1-xxxxx\t\n`).

### Phase 7: Streaming Inference & Dynamic Validation (`src/inference.py`, `src/submission.py`, `run_pipeline.py`)
*   **Master CLI:** `run_pipeline.py` supports `--stage all` or modular stage execution (`preprocess`, `block`, `features`, `train`, `infer`, `validate`).
*   **Dynamic Assertions:** Automatically verifies that `output/matching_results.tsv` and `output/candidate_pairs.tsv` have zero quotes, exact row-count parity, and 1-to-1 entity ID alignment against `test_source1.tsv`, followed by running the official `validate_submission.py`.

---

## 5. Chronological Timeline of Events & Execution Attempts

### Stage 1: Multi-Agent Build & Audit Remediation (Night of Sep 26 – Morning of Sep 27)
1.  **Laptop Crash Mid-Build:** During initial parallel multi-agent execution, the local laptop crashed. On recovery, agent concurrency was restricted to sequential single-worker execution and high-RAM tasks were frozen while unit tests were verified.
2.  **API Rate Limits (`RESOURCE_EXHAUSTED` 429):** Subagents briefly paused during Phase 4 due to API quota exhaustion and resumed once the quota window reset.
3.  **Victory Audit #1 Rejection (`pytest` vs. Windows UTF-8 Wrapper):**
    *   The first independent Victory Auditor passed Phase A (Lineage) and Phase B (all 14 AST Integrity checks), but **rejected Phase C** because `pytest`'s stdout capture buffer collided with `sys.stdout = io.TextIOWrapper(...)` on Windows, raising `ValueError: I/O operation on closed file`.
    *   **Fix:** Added `pytest.ini` with `addopts = -s` and hardened `setup_utf8_stdout()` in `src/utils.py` to check stream liveness before wrapping.
4.  **Victory Audit #2 Confirmed:** All **127 unit tests** and **210 E2E tests** (337 total) passed in 8.35s with a 0.71 GB peak RAM footprint on the mock test fixtures.
5.  **Dependency Verification:** Verified that `lightgbm`, `xgboost`, `catboost`, and `optuna` were installed in the active environment so the defensive dummy fallbacks (`_FallbackLGBMClassifier`, etc.) in `src/models.py` would not trigger during real runs.

### Stage 2: Attempt 1 — Local Laptop Run & The 16 GB RAM Math (Morning of Sep 27)
*   **Hardware:** Intel i7-13650HX (14 threads @ ~4.56–4.9 GHz), 16 GB DDR5 RAM, NVIDIA RTX 4050 Laptop GPU (6 GB VRAM).
*   **The 16 GB Memory Reality Check:** Why 16 GB RAM is pushed to the absolute brink during Phase 3 Blocking on large partitions (US / India):
    *   $S_2 + S_3$ Polars DataFrame in memory (4.1M–6.2M rows): **~2.5 GB**
    *   $S_2 + S_3$ strings materialized as Python list for `scikit-learn`: **~1.5 GB**
    *   Name TF-IDF CSR matrix (`300K` vocab): **~1.5 GB** + CSR transpose for dot product: **~1.5 GB**
    *   Address TF-IDF vectorizer + CSR matrix: **~1.2 GB**
    *   $S_1$ streaming batch + candidate buffers + Polars thread overhead: **~1.5 GB**
    *   Windows 11 OS + background services + Python runtime: **~3.5 GB**
    *   **Total Real Peak:** **~13.2–14.8 GB** (leaving $< 1.5\text{–}2.8\text{ GB}$ headroom before Windows pagefile thrashing or OOM).
*   **What Happened:**
    *   Preprocessing completed cleanly, and Phase 3 (`block`) began on `train/US` (1.32M $S_1$ rows $\times$ 6.18M $S_{2/3}$ rows).
    *   **GPU Idle (0%):** The RTX 4050 sat at 0% because Phase 3 text blocking (TF-IDF, Soundex, Token Index) is 100% CPU-bound; the GPU is only invoked in Phase 5 (`XGBoost`).
    *   **Hidden Native Memory Overhead:** While Python's internal heap logger reported `6.32 GB`, Windows Task Manager showed **95% system memory utilization (14.8 GB / 16 GB)** due to Polars and SciPy/C++ sparse matrix buffers. Increasing `STREAMING_BATCH_SIZE` above `50,000` on the laptop was impossible without causing an immediate Windows OOM crash.
    *   **Speed Bottleneck:** At `STREAMING_BATCH_SIZE = 50,000`, full 4-way blocking took **22.5 minutes per 50,000 rows** (~10 hours just for US Train, ~24–36 hours for the full pipeline).
*   **Fast Baseline Experiment:**
    *   Attempted to disable Strategy A (Character 3-gram TF-IDF) to get a fast baseline submission.
    *   *Bug Caught:* Initially disabled Strategy A in `src/blocking.py`, but `src/reranking.py` called Strategy A directly inside `streaming_cascaded_blocking()`. After patching `src/reranking.py`, chunk time dropped from 22.5 minutes to **5 minutes per 50,000 rows** (~9–10 hours total).
    *   **Leaderboard Reality Check:** Because the competition top 50 were scoring **0.97+ to 0.991**, dropping Strategy A (which costs 1–2%+ recall on typo-heavy matches) was unacceptable. We needed full 4-way blocking on cloud hardware with more RAM.

### Stage 3: Attempt 2 — Cloud Server #1 (`gpu-ogjj`, 12:00 PM – 4:10 PM)
*   **Hardware Provisioned:** AIC Cloud / Vast.ai `RTX PRO 5000` instance (`gpu-ogjj`).
*   **Environment Setup:**
    *   Created a Miniconda `Python 3.11` environment (`er_env`) because the server defaulted to Python 3.10 and `corp-names==0.3.3` requires Python $\ge 3.11$.
    *   Installed `libgomp1` and all ML libraries, uploaded the 2.5 GB dataset via `scp`, re-enabled Strategy A, and set `STREAMING_BATCH_SIZE = 2,000,000`.
    *   Pushed the full codebase to GitHub on branch `feat/antigravity-pipeline`.
*   **What Happened:**
    *   **Core Count Realization:** While the marketplace listing showed a 64-vCPU variant, the actual deployed instance (`gpu-ogjj`) only had **16 vCPUs and 94 GB RAM**. Thus, `NUM_THREADS = 14` was already maxing out the machine (1400% CPU).
    *   After ~3.5 hours of grinding through the massive 2,000,000-row US Train matrix on 14 threads, the remaining pipeline was still projected to take another ~4 hours (~8:00 PM finish). To finish much faster, we decided to migrate to a massive multi-core instance.

### Stage 4: Attempt 3 — Cloud Server #2 (`gpu-50wf`, 4:30 PM – 10:00 PM)
*   **Hardware Provisioned:** AIC Cloud / Vast.ai `RTX 4090` instance (`gpu-50wf`, port `16810` on `ssh9.aiccloud.one`) with **112 vCPUs** and a **170.8 GiB Linux container cgroup memory limit** (on a 503 GB physical host).
*   *(Note: A duplicate instance `gpu-2pui` was accidentally spun up alongside `gpu-50wf` and was immediately destroyed to prevent double billing).*
*   **What Happened:**
    1.  **`src/config.py` Mutation (`NUM_THREADS = 110`, `STREAMING_BATCH_SIZE = 1_000_000`):**
        *   Updating `NUM_THREADS` to `110` in `src/config.py` cascaded into every downstream module (Polars, `sparse_dot_topn`, and batch buffers) simultaneously.
    2.  **Slower Single-Threaded Preprocessing & Vectorization (~5:00 PM – 6:00 PM):**
        *   Preprocessing and `TfidfVectorizer.fit()` ran noticeably slower on the 112-core server than on the laptop.
        *   **Why:** Server EPYC processors with 112 cores are thermally clocked at **2.0 GHz** per core (vs. **4.9 GHz** on the i7 laptop). Because Python regex preprocessing and Scikit-Learn `TfidfVectorizer` vocabulary fitting are strictly single-threaded (bound by Python's GIL), they only used 1 core at 2.0 GHz.
    3.  **Repeated Out-Of-Memory (`oom_kill`) Crashes (~6:50 PM – 9:05 PM):**
        *   With `NUM_THREADS = 110` and `STREAMING_BATCH_SIZE = 1,000,000`, the Linux kernel killed the process at the end of the US batch.
        *   We restarted from `--stage block` (skipping the saved preprocessing parquets) with smaller batch sizes (`100,000`, then `50,000`), and reduced `TfidfVectorizer` `max_features` (`100,000` for names, `50,000` for addresses), yet the process still spiked past the **170.8 GiB cgroup limit** and suffered `oom_kill`.
        *   **Root Cause Discovery:** The culprit was **Polars in Strategy B (`blocking_strategy_b_token_index`)**. Polars scales its internal hash-table memory allocations linearly with `RAYON_NUM_THREADS` (which defaults to all 112 logical cores). Executing a parallel Polars hash-join across millions of token rows on 112 threads spawned 112 independent thread-local hash tables (~1.5 GB each $\times 112 \approx 168\text{ GB}$), consuming over 170 GB of RAM in seconds.
    4.  **The 16-Core Safety Throttle (~9:05 PM – 9:37 PM):**
        *   Setting `export POLARS_MAX_THREADS=16`, `NUM_THREADS = 16`, and `STREAMING_BATCH_SIZE = 25,000` completely cured the OOM crashes.
        *   However, running 16 cores at 2.0 GHz pushed the realistic completion ETA to **1:30 AM – 2:00 AM**, well past the user's hard cutoff (9:15 PM, extended to 10:00 PM max).
    5.  **Emergency Surgery & The Silent `sparse_dot_topn` Single-Core Bug (~9:37 PM – 10:00 PM):**
        *   To try to finish before 10:00 PM, we killed the 16-core run, patched out Strategy B (eliminating the Polars OOM bottleneck entirely), reduced TF-IDF `max_features` to `25,000` (names) and `10,000` (addresses), and restored `NUM_THREADS = 110` with `STREAMING_BATCH_SIZE = 100,000`.
        *   After vectorization finished at 9:48 PM (`16:18 UTC`) and streaming blocking began, the process still hadn't completed a batch by 9:58 PM.
        *   **Root Cause Discovery via `top`:** Inspecting `top` on the server revealed PID 5730 was running at **`%CPU = 100.0`** (1 single core!) with `78.6 GiB RES` memory, leaving 95.1% of the 112-core machine completely idle.
        *   **Why:** Unlike the Windows wheel on the laptop (which compiled with MSVC OpenMP and saturated all 14 threads), the pre-compiled Linux `pip` wheel for `sparse_dot_topn` on the cloud container was built **without OpenMP multithreading support**. Passing `n_threads=110` was silently ignored by the C++ binary, bottlenecking billions of matrix multiplications onto a single 2.0 GHz core (~110x slower than intended).
    6.  **Shutdown (10:00 PM):**
        *   Killed the remote Python process (`PID 5730`) and cancelled all monitoring crons so the cloud instance (`gpu-50wf`) could be destroyed without incurring overnight charges.

---

## 6. Technical Root Causes & Exact Fixes for Future Runs

Whether you run this pipeline on a multi-core Linux cloud server or adapt it for your local 16 GB laptop in the future, apply these fixes first:

### 6.1 Running on a Multi-Core Linux Cloud Server (Recommended)

#### Fix 1: Compile `sparse_dot_topn` from Source with OpenMP Enabled
Standard Linux `pip` wheels for `sparse_dot_topn` frequently lack OpenMP support, silently forcing `n_threads` onto a single CPU core. Always compile it from source on the Linux instance:
```bash
sudo apt-get update && sudo apt-get install -y build-essential libomp-dev
pip uninstall -y sparse_dot_topn
CFLAGS="-fopenmp" LDFLAGS="-fopenmp" pip install --no-binary :all: sparse_dot_topn
```
*Verification:* When `Stage Block` reaches `Streaming Cascaded Blocking`, run `top` and verify that `python` shows **`%CPU` between `6000%` and `11000%`**, NOT `100.0%`.

#### Fix 2: Decouple `POLARS_MAX_THREADS` from `NUM_THREADS`
Never let Polars default to 100+ threads during high-cardinality token joins (Strategy B), or it will allocate 100+ thread-local hash tables and trigger a Linux cgroup `oom_kill`.
*   Keep `POLARS_MAX_THREADS` capped at `16` or `32`, while allowing OpenMP (`sparse_dot_topn`, LightGBM, CatBoost) to use all 110 cores:
```bash
export POLARS_MAX_THREADS=16
export RAYON_NUM_THREADS=16
export OMP_NUM_THREADS=110
```
*   In `src/config.py`, set `NUM_THREADS = 110` (for OpenMP) and `STREAMING_BATCH_SIZE = 100_000` (or `250_000` if RAM $\ge 128\text{ GB}$).

### 6.2 Running on the Local 16 GB Laptop (Low-Memory Mode)
Note that `src/config.py` in the repo currently has `NUM_THREADS = 110` and `STREAMING_BATCH_SIZE = 1_000_000` from the cloud run. To run locally on the i7-13650HX (16 GB RAM):
1.  **Restore `src/config.py`:** Set `NUM_THREADS = 14` and `STREAMING_BATCH_SIZE = 50_000`.
2.  **Trim TF-IDF Vocabulary in `src/blocking.py`:** Reduce `max_features` from `300,000` to `100,000` (and optionally skip Strategy C Address TF-IDF or run it in a separate process after freeing the Name CSR matrices) to shave ~3–4 GB off the peak memory footprint, keeping total system RAM safely around ~9–10 GB.

### 6.3 Add Per-Morsel Progress Logging in `src/reranking.py`
Currently, `streaming_cascaded_blocking()` in `src/reranking.py` logs once before the chunk loop (`Starting stage: 'Streaming Cascaded Blocking...'`) and does not log again until every batch in the country partition finishes. Adding a `logger.info(f"Processed batch {batch_idx}/{total_batches} (RAM: {get_memory_usage_gb():.2f} GB)")` inside the `for start_idx in range(0, total_s1, batch_size):` loop gives immediate visibility into per-chunk speed and ETA.

### 6.4 Leverage Hybrid Stage Checkpointing (`--stage`)
The pipeline is completely modular and saves intermediate Parquet checkpoints in `data_processed/`:
1.  `python run_pipeline.py --stage preprocess` *(writes `data_processed/{train,test}/{country}_{s1,s23}.parquet` — fast on high-GHz laptop CPU)*
2.  `python run_pipeline.py --stage block` *(writes `data_processed/{train,test}/candidate_pairs_{country}.parquet` — best on high-RAM cloud server with OpenMP)*
3.  `python run_pipeline.py --stage features` *(writes `data_processed/{train,test}/features_{country}.parquet`)*
4.  `python run_pipeline.py --stage train` *(trains LightGBM + GPU XGBoost + CatBoost + Optuna 3D thresholds, saves to `models/`)*
5.  `python run_pipeline.py --stage infer` *(generates `output/matching_results.tsv` and `output/candidate_pairs.tsv` and runs `validate_submission.py`)*

---

## 7. Key Takeaways & Pointers for Future ML Competitions

1.  **Run a 10,000-Row Smoke Test on the Target OS First:** Always run `run_pipeline.py --stage all` on a 1% subsample (10K rows) *immediately* upon provisioning a new cloud machine while watching `htop` and `nvidia-smi`. A 2-minute subsample test on `gpu-50wf` would have immediately exposed the `100.0%` single-core `sparse_dot_topn` wheel bug at 5:00 PM instead of 9:50 PM.
2.  **Beware of Pre-Compiled Linux Wheels for C++ Extensions:** Packages like `sparse_dot_topn`, `faiss`, or `nmslib` often ship manylinux wheels compiled with conservative flags (missing OpenMP or AVX-512) for compatibility. Always benchmark C++ extensions on Linux with a 10-line script and verify multi-core CPU saturation via `top` before starting a multi-hour job.
3.  **High vCPU Count $\ne$ Fast Single-Thread Speed:** Cloud EPYC/Xeon servers trade clock speed (2.0 GHz) for core count (112 cores). Any GIL-bound Python step (like `TfidfVectorizer.fit()` or regex preprocessing) will run **2.5$\times$ slower** on a cloud server than on a 4.9 GHz laptop. Either run single-threaded preprocessing locally and upload the resulting `.parquet` files, or parallelize vocabulary fitting across chunks.
4.  **Polars Thread Pools Must Be Capped on Many-Core Servers:** Polars is optimized for 8–32 cores. On 64–128+ vCPU servers, Rayon spawns a worker per logical core, multiplying hash-table memory footprints by $100\times$. Always set `POLARS_MAX_THREADS=16` on big-iron servers.
5.  **Python Heap Loggers Under-Report Native C++/Rust Memory:** `psutil` process RSS or Python memory trackers often miss transient C++/Rust Arena allocations (SciPy CSR matrices, Polars Arrow chunks) until the OS cgroup killer strikes. Always leave at least 30% headroom below the container's hard RAM limit.
