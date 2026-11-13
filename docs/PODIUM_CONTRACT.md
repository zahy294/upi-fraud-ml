# UPI implementation contract: codex/target-070

Card 1 starts at merge commit 77a7fad (target starting SHA
7ae1941900737d34eb0139e0e4295e2e3e597ecc, modelling source SHA
d4ecf6caef06c08fc43b965f14fbabb998a3a34a).
All implementation commits remain on codex/target-070. No push until Card 3.
The modelling branch is a read-only source; it is not modified.

## Fixed classes

Indices 0..11:
legit, mule_account, velocity_bot, layering_chain, sim_swap,
account_takeover, phishing_collect, qr_swap, refund_abuse, fake_cashback,
loan_app_extortion, merchant_collusion.
features_v6 checks common.LABELS on import.

## Time, complete context and IDs

Base columns required by the combined feature construction:
txn_id,t,split,month,label,sender_id,receiver_id,receiver_type,device_id,
txn_type,amount,balance_before,new_payee,hour,merchant_category.
Preserve all inherited row and relational columns.
t is the existing EDA loader's integer seconds coordinate; this card does
not reinterpret it as Unix time or change its origin. Dates use the original
timestamp when available and YYYY-MM calendar month, never today's year.
The permitted source does not reveal the EDA epoch. A later runner must
inspect the allowed loader before translating calendar dates into t.
Frozen TE accepts timestamp, or t from the same base convention for both
reference and query; it also checks calendar-month order.

Build base and all label-free features over complete train+test context
before cleaning or legitimate-row sampling. Sort by (t,txn_id) deterministically.
ID-keyed feature packs require unique, non-null txn_id, exact base ID sets,
one-to-one merges and explicit restoration to base order. Receivers use
(receiver_type,receiver_id), keeping merchants separate from P2P accounts.
No caller may silently slice feature packs before complete-context construction.

build_tables(frame,relational) -> (base,ratios) is pure; main(argv=None)
preserves python -I src/upi/build_base.py ROOT.
The internal ratios table carries txn_id as a named parquet index rather than
a numeric predictor column: existing features_v1 iterates all ratio columns.
The builder verifies that ratio IDs match base order before writing. v1's
unchanged reader only checks length; Card 2/3 must additionally verify the
ratio index equals base.txn_id before running that legacy reader.
Sequence ratios retain adjacent-row semantics, including same-time adjacency;
the new backward v6 counts exclude all simultaneous events.
Onward account matches now explicitly exclude merchant receivers. Infinite
ratios caused by zero denominators become NaN.

## Pack dependencies

Existing EDA interim and relational_v0 artifacts -> build_base ->
base_v0.parquet and ratios_v0.parquet -> features_v1 and features_v2.
features_v4 reads base directly.
features_v3.parquet is produced by bands.py, not a features_v3.py script.
features_v5 requires base, v1 and v3 (including ch_band propagation).
Label-free v6 requires base plus packs["v4"].
Do not import unchanged v1/v2/v4/v5 scripts as modules: their legacy entry
points still execute top-level code. Invoke them externally or refactor in
a separately authorized card.

## Events

Events(group,tt,vals=None,mask=None,*,values=None).
count/sum(qg,qt,lo,hi) use closed endpoints. prev_dt/next_dt(qg,qt,within)
exclude equal timestamps. previous_gap,next_gap,previous_value,next_value
retain the older target API.
Integer group/time vectors and matching boolean masks are validated.
Structured (group,time) keys avoid composite multiplication overflow and
cross-group queries. Negative query groups never match. Empty pools yield
zero counts/sums and NaN gaps. Sum without values raises ValueError.
Out-of-range int64 time bounds/differences fail explicitly.
No data access or CLI execution occurs on module import.

## Cleaning API and provenance

clean_training_labels(base,teacher_proba,bands,sessions,fit_ids,
                      *,tau_unrep=0.85) -> dataframe.
teacher_proba must be in exact base order, shape (N,12), with either a valid
normalized probability vector or all NaN per row. Arrays have no intrinsic IDs:
the caller must align saved teacher IDs to base IDs before passing them.
bands and sessions are full base-ID-keyed frames; sessions requires S_flag.

Only fit_ids labels are read and only those rows can vote. fit_ids must be
unique known training IDs. Return columns txn_id,y_raw,y_clean,reason,
teacher_available, in base order for the selected fit rows; no rows are dropped.
Reasons: unchanged,unreported,event_type,band.

Detected = teacher fraud probability >=0.5 OR band>0.
Per-class raw-label votes: same sender +/-7d; eligible receiver +/-1d
(P2P or S_flag); same device +/-10min; reverse P2P pair +/-7d.
Subtract self in the first three channels. Additive multi-link votes are
intentional. Require winning votes>=1 and share>=0.6.
No-band original legit with teacher fraud>=0.85 becomes accepted event type,
else teacher fraud argmax. No-band detected original fraud changes only to
a different accepted event type. Direct bands take precedence. Original fraud
labels disagreeing only with teacher type remain unchanged.
All-NaN teacher rows cannot trigger teacher-dependent changes; band evidence
remains usable. No trap columns enter cleaning.

Compatibility CLI:
python -I src/upi/clean_labels.py ROOT TAG [--tau-unrep 0.85].
Reads legacy train-order oof.npy and writes y_clean.npy plus ID-keyed
cleaning_rows.parquet. This CLI uses all training IDs and does not establish
OOF provenance itself. Card 2 must use the pure function per outer fold.
Diagnostics are not held-out performance claims.

## Band API

build_bands(base,teacher_proba,f1,fit_ids) -> (frame,diagnostics).
f1 must carry txn_id and snd_drain_kind for the full base.
Fit labels and anchor-density teacher evidence are restricted to fit_ids.
All rows from the original train generation, including September, use
the resulting train edges. Test has separate edges inferred without labels.
Base device IDs require a nondigit prefix and integer suffix.

Preserved: ordered 11-segment dynamic programming; shared bot pool
420001..423000; sibling mule/bot and SIM/takeover scores; Gaussian test
boundary prior with sd=0.02*test fraud width. Ordinary blocks require 40
consecutive observed device numbers. Block endpoints use label-free context.
Mule/SIM density uses unique anchored device numbers divided by band width,
clipped to [1e-6,1-1e-6], not transaction counts.

Missing ordinary blocks, invalid widths, absent fraud fitting support,
undefined anchor segments, invalid probabilities or missing test-block teacher
support raise descriptive ValueError. Empty non-anchor DP segments remain
possible as in the inherited algorithm and are counted in diagnostics.
No-test contexts return test=null. Empty base/fold cannot fit edges and raises.
No amount-band hard overrides are introduced.

Output: txn_id,band,dev_ext_legit,snd_band_any_3h,rcv_band_any_24h and
snd_band_{k}_3h,rcv_band_{k}_24h for k=1..11; numeric values float32.
Diagnostics include edges, ordered labels, bot pool, densities, unique
anchor counts, fit/teacher support and empty-segment count.

Compatibility CLI: python -I src/upi/bands.py ROOT TAG.
Legacy train/test probability arrays must already match base split order.
Writes features_v3.parquet and band_edges.json.
Card 2 must isolate outer-fold and full-refit outputs and rebuild dependent
v5 ch_band for each context. Do not reuse all-label-fitted bands in validation.

## V6 APIs and columns

build_label_free_v6(base,packs) -> full-ID feature frame; never reads label.
For snd,rcv,dev: v6_{entity}_n_{10m|1h|24h} and
v6_{entity}_amt_sum_{10m|1h|24h}, windows [t-w,t).
Integer-second implementation uses closed [-w,-1].
Sender/device velocity: Nshort=count[t-600,t),
Nbackground=count[t-21600,t-600), mu=Nbackground*600/21000,
v6_{entity}_velocity_z=clip((Nshort-mu)/sqrt(mu+1),-20,20).
v6_dev_unique_senders_24h uses a sorted sliding counter, excluding all rows
at the query timestamp and null senders.
v6_bait_delay_fraction=bait_prev_dt_h/
(bait_prev_dt_h+pair_tiny_prev_dt_h+1/3600).
v6_bait_recency_density=bait_cnt_before_1d/(1+bait_prev_dt_h).
v6_bait_prev_missing and v6_pair_tiny_prev_missing explicitly mark absence.
Missing ratios stay NaN; infinities raise. There are 25 new numeric columns,
all float32, plus txn_id.

build_temporal_te(query,reference,*,alpha=50.0,expanding_months=True)
returns txn_id plus 39 float32 columns.
Both frames need unique txn_id, month,sender_id,receiver_id,receiver_type,
device_id; reference needs original label. Frozen mode additionally needs
timestamp or same-coordinate t in each frame.
Expanding mode uses only strictly earlier calendar-month reference labels.
Frozen mode asserts maximum reference time < minimum query time and checks
calendar-month order. Caller restricts reference to the outer fitting fold.
For each entity and class k:
pi_k=(N_k+1)/(N+12);
v6_te_{entity}_{k}=(n_entity_k+alpha*pi_k)/(n_entity+alpha);
v6_te_{entity}_log_support=log1p(n_entity).
No reference gives uniform 1/12. Null entities do not aggregate labels.
Future labels are not mapped/validated until eligible. Unknown eligible
labels raise. No cleaned labels enter this encoding.

assemble_features(base,packs) -> (X,schema).
Includes all supplied v1..v5/v6/TE packs and base predictors selected by
common.DROP/CATS; also excludes common.TRAPS and cleaning provenance.
No new trap features. Exact equal duplicate columns are retained once;
conflicting duplicates raise. Unknown nonnumeric predictors raise rather than
silently dropping them. CATS are sorted-string integer codes; missing/unknown
category=-1. Schema contains ordered predictor columns, category mappings,
sentinel, class order and row IDs.
apply_feature_schema(raw_predictor_frame,schema) reuses persisted mappings
for later predictions; never independently factorize validation/test categories.
Persist schema once and keep the same base/pack construction across stages.
NaN numeric values are retained; infinite assembled predictors raise.

## Verification

No model training, CV, dependency installation, data/model/notebook reads.
Run:
rtk proxy python -m py_compile src/upi/events.py src/upi/build_base.py src/upi/clean_labels.py src/upi/bands.py src/upi/features_v6.py tests/test_podium_features.py
rtk proxy python tests/test_podium_features.py

Synthetic fixtures have at most 42 rows each. The test-block fixture reduces
only its mocked ordinary-block run length to two; production always uses 40.
Tests cover event boundaries/empty pools/group isolation, large IDs, ID
alignment, rolling-vs-brute-force distinct counts, feature label invariance,
future-label TE invariance, sibling-label preservation, held-out cleaning and
band-label invariance, test-block separation, schema reuse and trap exclusions.
Syntax and synthetic success do not establish end-to-end Colab compatibility
or any leaderboard improvement.

Local environment limitation: an isolated-mode (-I) CLI help probe cannot
import NumPy because dependencies are installed in the Windows user site.
Normal-interpreter synthetic checks work. No packages were installed to alter
this environment. Colab must install its declared dependencies in its active
environment before using the inherited -I command form.

## Card 2 training and decision APIs (codex/target-070)

Card 2 starts at e3b117f and changes only fit_submit.py, make_submission.py,
configs/podium.json, tests/test_podium_predictions.py and this contract.
No training has been executed locally. The modelling branch remains unchanged.

### External entry point

python src/upi/fit_submit.py ROOT --config configs/podium.json --stage all
  [--profiles control,fusion,temporal,ensemble] [--device cpu|gpu|cuda]
  [--allow-cpu-fallback] [--output-dir PATH] [--tag NAME]

Stages: bootstrap, validate, refit, all. For separate commands, use identical
config, profile selection and device options at every stage. Default profiles:
ensemble only. All comparisons can be requested explicitly; no profile is
automatically chosen using the assessment block. This CLI replaces the legacy
fit_submit --trees/--seeds interface; inherited run_all commands must be
updated by Card 3 rather than assumed compatible.

Output root defaults to ROOT/runs/podium_v6. Explicit output_dir can point
to local Colab disk; Card 3 handles copying completed artifacts to Drive.
Card 2 refuses existing-run/bootstrap/validation provenance mismatches but does not
implement resume or atomic stage recovery. Card 3 must add those guards.

### Calendar and splitting

Original base.timestamp is required. Never reinterpret t as Unix seconds.
Naive timestamps retain their calendar dates; timezone-aware timestamps are
converted to configured Asia/Kolkata before calendar selection.
The year is inferred from July-August training timestamps.
Boundaries are half-open intervals configured as month-day strings:
fit [07-01,09-01); early_stop [09-01,09-16);
calibration [09-16,09-24); assessment [09-24,10-01).
All supervised partitions contain training rows only. Missing, overlapping
or nonchronological required partitions raise. Actual dated training tails
are reported separately, not folded back into September. Full September
contains only actual September rows. September has already informed project
development; assessment is not claimed to be untouched project data.

### Teacher and context construction

Bootstrap reads base_v0, ratios_v0, v1,v2,v4 only. The ratio named txn_id index
must equal base txn_id order. Teacher features exclude bands, TE, cleaning
labels and the common trap exclusions; teacher is fixed at 500 rounds.
July1-14 has unavailable forward probabilities; July15-31 is predicted
from July1-14, August from July, September from July-August. Later training
tails are predicted from earlier calendar months. There is no backwards OOF.
Separate test teachers fit July-August for validation bands and all training
rows for final-refit bands. Saved predictions have explicit IDs and provenance.
The bootstrap stage prepares both contexts; validation uses only its eligible
teacher, fitting labels, boundaries and history.

For each context: build_bands -> rebuild_chain_features -> clean_training_labels
-> v6/TE as requested. v5 is reconstructed in fit_submit without importing
the top-level legacy script. Its nearest incoming/outgoing payment, strict
time matching, 90%-100% amount ratio, 7200-second window and eight-hop walk
match the inspected v5 source. Unambiguous ch_band is recomputed from each
context's bands; shared data/processed/features_v3/v5 are never overwritten.

build_context_te(base,fit_mask) produces training encodings from strictly
earlier eligible calendar months. Non-fitting queries use a frozen eligible
reference for their query month. An October training tail is excluded from
all October test encodings; eligible October tails may enter November
encodings. This avoids passing a future-dated reference to the frozen TE API.

### Models and profiles

train_member(X_fit,y_fit,sample_weight,X_stop,y_stop,*,spec,device,
fixed_rounds=None) -> (model,rounds).
LightGBM is imported only here. CPU default; gpu/cuda are explicit choices.
Unavailable backend raises unless allow_cpu_fallback was explicitly enabled.
Fallback records the actual CPU backend; no claim of A100 acceleration.
Card 3 must perform its requested tiny external backend probe.
Fixed-round refits require no stop/eval set and disable early stopping.
Observed classes are explicitly remapped and stored as model.podium_classes_;
align_proba(proba,classes) restores fixed 12-column order and normalizes.
Stop sets containing a class absent from fitting data raise a clear error.
No local backend fit or LightGBM integration run was performed.

A: seed17,leaves63,min_child20,colsample0.50,weight0.50.
B: seed43,leaves31,min_child60,colsample0.75,weight0.30.
C: seed101,leaves95,min_child40,colsample0.65,weight0.20.
All multiclass, lr0.05, bagging0.8/freq1, L2=5; maximum2000 rounds,
early stopping100 on original-label stop multi_logloss.
Legit sampling rate0.25, seed2026; ID-hashed draws keep the same rows across
members/profiles. Keep all cleaned-fraud rows; retained legit weight4 and
fraud weight1. Teacher fits use all eligible raw labels with unit weights.
Final refit uses each member's validation-selected round count.
Categorical feature names and integer mappings are persisted and reused.

control: base+v1..v5, member A.
fusion: control+label-free v6, member A.
temporal: fusion+TE, member A.
ensemble: temporal with weighted A/B/C probabilities.
Single-member profiles normalize A's weight to1.
Control is a fold-safe comparison, not a reproduction of public0.5400.

### Band rules and calibration

apply_band_rules(pred,meta,f3,f5) aligns frames by txn_id.
Direct band first. Non-direct-band rows whose ORIGINAL argmax is fraud can
receive sender drain, receiver fan-in/shop, then chain corrections, in
inherited order. Chain never overrides direct band or promotes original legit.

event_groups uses priority direct_band > chain > merchant_session > drain
> bait > other. Conditions: band>0; ch_len>1; S_flag==1; Ds_n>=2;
is_tiny_p2p_pay==1 or finite bait_prev_dt_h.

fit_offsets(proba,y_raw,groups,rule_context,*,config) requires original
September16-23 timestamps. rule_context can be {meta,f3,f5} or the prepared
ID/array context from prepare_rule_context. The caller must align probability
IDs before passing arrays. predict_labels(proba,groups,offsets,rule_context)
uses log(max(P,1e-9))+global+conditional, then band rules.

Global legit offset stays0. Coordinate steps0.4,0.2,0.1,0.05, at most3 passes
per step, bounds+/-1.5. Objective: original-label 12-class macro-F1 minus
0.002*mean(global[1:]**2); ties prefer smaller norm. Traces are persisted.
Conditional candidates +/-0.25,+/-0.125,0 are shrunk by n/(n+2000);
require group n>=200 and affected true-class support>=30.
No direct-band or legit-class conditional offset. Additional penalty
0.01*mean(conditional**2). Enable only if Sep16-19-fitted conditional
parameters beat global-only parameters on Sep20-23. Refit on Sep16-23 only
after that gate. Otherwise conditional offsets stay empty.
No assessment-dependent threshold fitting or automatic retuning.
Both selected and global-only assessment metrics are recorded.
No cleaned-label agreement is used as a headline validation score.

### Artifacts and metrics

bootstrap/ contains schema, ID-keyed forward_teacher.npz (warm-up NaN rows
explicitly allowed), validation_test_teacher.npz, refit_test_teacher.npz,
teacher model text files, model metadata and manifest.
contexts/validation and contexts/refit contain full-ID band/chain frames,
cleaning rows and diagnostics.
Each PROFILE/validation contains schema, unknown-category counts, model
text/JSON, member probabilities, validation_proba.npz, per-partition
probabilities, meta.parquet, matching f3/f5, groups.npy, offsets.json,
metrics.json, per_class_metrics.csv and manifest.
PROFILE/refit contains corresponding models/schema/metadata,
test_proba.npz, test meta/f3/f5/groups, submission.csv and submission_metrics.json.
inputs.json records content hashes, dataset year, config and source hashes.
Manifest records configuration, class order, feature-schema hash, training
ID hashes, rounds, backends and split boundaries.
Card 3 adds commit SHA, full dependency versions, timings, logs and copies.

Probability artifacts are pickle-free NPZ: ids,proba,classes,provenance.
load_probability_artifact rejects ID order, class order, shape, invalid
probabilities and (when supplied) provenance mismatches. Never blend two
arrays without matching these identities. Member metadata also records
observed-class mapping and actual backend.

evaluate_predictions returns all12 precision/recall/F1/support/predicted
counts, confusion matrix, macro-F1, fraud rate and empty-subset flag.
Validation reports early stop, calibration, assessment, full September,
tail, direct-band and no-direct-band subsets. Empty subsets have zero
macro-F1 and empty=true, avoiding a claimed measurement on no examples.
Reports include band-only and offset changes and cleaning reason counts.
Saved predictions are named by partition, not falsely called OOF.

write_submission(test_ids,pred,sample,path) enforces unique non-null IDs,
equal sample/prediction sets, exact sample order, columns id,label, valid
labels and no missing values. It reports absent predicted classes instead
of forcing them to occur. SHA256 and distributions are returned.
Standalone rerender:
python src/upi/make_submission.py ROOT/runs/podium_v6/ensemble --sample ROOT/data/raw/sample_submission.csv
This reads persisted test identities/context and validation offsets, not
test labels or assessment-driven parameters. No fraud-rate matching.

### Card 2 checks

rtk proxy python -m py_compile src/upi/fit_submit.py src/upi/make_submission.py tests/test_podium_predictions.py
rtk proxy python tests/test_podium_predictions.py
Fixtures have <=40 rows. Mock estimators only record arguments/return
fabricated probabilities; no real LightGBM fit occurs. Temporary synthetic
files exercise serialization, validation/refit artifact shapes and sample
mapping. Any tiny printed score in this suite is a synthetic fixture score,
not competition performance. Real training, runtime, backend compatibility
and leaderboard improvement remain externally unverified.
