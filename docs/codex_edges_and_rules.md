# Test band uncertainty and collect-family rules — 14 November 2026

Parts 1 and 2 completed. No predictive models fitted. Only class-weight calibration on saved probabilities is fitted, as required by the specified benchmark. New files: scripts/explore/44_codex_test_edges.py, scripts/explore/45_codex_collect_rule.py, and this report.

**Fixed benchmark reproduced first.** October training tails are assigned to September, exactly as common.load_xy. Pooled final_local uses its saved weights_final_local.npy, reproducing 0.5597/0.5113; recalibrating against the fixed target gives 0.559606/0.511298 because its original calibration labels differ on 69 rows. s2_a pooled weights and all nested weights use the fixed target. Nested means ONLY two-month weight calibration, not fully nested upstream feature/model construction. Fast calibration is checked against the existing implementation on 5,000 rows.

| model | pooled_clean | pooled_raw | nested_clean | nested_raw |
| --- | --- | --- | --- | --- |
| oof_final_local | 0.559686 | 0.511344 | 0.558918 | 0.510583 |
| s2_a | 0.562701 | 0.513932 | 0.561506 | 0.512919 |

**Part 1 — uncertainty model.** Each numbered device is one observation, including unused numbers. Occupancy probabilities use train-band Beta(0.5,0.5) posterior means; shape emissions use occupied-device mean stage-1 probabilities, fraud threshold 0.5 and 15 categories (ordinary, classes 1–11, quarter-drain, .35-drain, ambiguous drain), with Dirichlet(0.5) smoothing. Thus sibling-shaped anchors retain their empirical generator-band likelihood. Anchor evidence is conditional on occupancy; multiplying the two does not count absence twice. Width prior: independent Gaussian penalties on ordered inner cuts, sd=2% of block width, fixed endpoints, strictly positive widths. Exact forward/backward dynamic programming marginalizes every other cut. These are 95% equal-tail **conditional credible intervals**, not calibrated frequentist guarantees: emissions, occupancy, sibling mix and prior strength are plug-in assumptions. Anchor-only and occupancy-only use a uniform ordered-cut prior.

Occupancy estimates by class 1–11: [0.5328, 0.3524, 0.4514, 0.7078, 0.2626, 0.5623, 0.338, 0.4887, 0.5551, 0.5, 0.4578]. Native bot bursts use the separate 420001–423000 device pool. Halving their frequency does not imply halving the new-device bot-band width; the bot-band anchor counts by category are [167, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]. We therefore retain the width prior and separately show the stronger half-entire-bot-band sensitivity. Test labels are never used.

Modes [95% CI] by evidence. Main disagreements: bot/layering anchors prefer 473839 versus occupancy 473818 and width prior 473826; SIM/takeover occupancy prefers 473988 versus anchors 473971 and prior 473975; phishing/QR anchors prefer 474360 versus prior 474334; occupancy gives very weak tail localization (loan/collusion mode 474550 versus anchors 474529). Half-entire-bot-width prior leaves all joint-MAP cuts unchanged. Half-bot column is joint MAP under that sensitivity prior.

| edge | between | current | anchors | occupancy | prior | combined | half_bot_MAP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | mule_account/velocity_bot | 473750 | 473750 [473735,473807] | 473757 [473740,473773] | 473742 [473706,473778] | 473751 [473749,473762] | 473751 |
| 2 | velocity_bot/layering_chain | 473826 | 473839 [473826,473839] | 473818 [473773,473886] | 473826 [473790,473862] | 473839 [473823,473839] | 473839 |
| 3 | layering_chain/sim_swap | 473920 | 473918 [473916,473920] | 473920 [473879,473948] | 473906 [473870,473941] | 473920 [473916,473920] | 473920 |
| 4 | sim_swap/account_takeover | 473969 | 473971 [473968,474003] | 473988 [473980,473994] | 473975 [473939,474011] | 473988 [473977,473992] | 473988 |
| 5 | account_takeover/phishing_collect | 474280 | 474281 [474280,474285] | 474276 [474133,474294] | 474267 [474230,474302] | 474281 [474280,474285] | 474281 |
| 6 | phishing_collect/qr_swap | 474360 | 474360 [474360,474372] | 474362 [474146,474426] | 474334 [474299,474370] | 474360 [474360,474369] | 474360 |
| 7 | qr_swap/refund_abuse | 474459 | 474461 [474457,474462] | 474443 [474275,474458] | 474448 [474412,474478] | 474460 [474456,474462] | 474460 |
| 8 | refund_abuse/fake_cashback | 474496 | 474496 [474496,474505] | 474457 [474311,474504] | 474483 [474454,474510] | 474496 [474496,474505] | 474496 |
| 9 | fake_cashback/loan_app_extortion | 474509 | 474512 [474510,474517] | 474509 [474350,474541] | 474508 [474481,474533] | 474512 [474509,474517] | 474512 |
| 10 | loan_app_extortion/merchant_collusion | 474526 | 474529 [474520,474546] | 474550 [474420,474551] | 474531 [474504,474550] | 474527 [474520,474543] | 474527 |

Relative log marginal profiles (position: log density relative to its maximum; selected quantiles plus current and modal positions). Each evidence-only profile integrates other cuts under uniform ordering constraints; the prior column is the ordered width-prior marginal. Runnable posterior() returns every position.

| edge | combined | anchors | occupancy | prior |
| --- | --- | --- | --- | --- |
| 1 | 473743:-4.13; 473749:-2.35; 473750:-0.02; 473751:0.00; 473752:-0.35; 473756:-1.05; 473758:-1.42; 473762:-2.93; 473767:-4.89 | 473750:0.00; 473760:-1.47; 473792:-2.47 | 473746:-1.58; 473750:-0.67; 473755:-0.83; 473757:0.00; 473762:-1.63 | 473719:-0.81; 473742:0.00; 473750:-0.08; 473766:-0.82 |
| 2 | 473816:-5.10; 473823:-3.34; 473826:-2.71; 473829:-2.00; 473833:-1.16; 473836:-0.51; 473838:-0.24; 473839:0.00; 473844:-4.75 | 473826:-3.69; 473831:-1.58; 473836:-0.53; 473839:0.00 | 473798:-1.66; 473818:0.00; 473822:-0.99; 473826:-0.74; 473862:-1.80 | 473803:-0.80; 473826:0.00; 473850:-0.83 |
| 3 | 473916:-1.93; 473917:-1.33; 473918:-1.18; 473919:-0.59; 473920:0.00; 473921:-4.64 | 473916:-0.63; 473918:0.00; 473920:-0.00 | 473896:-3.25; 473920:0.00; 473929:-1.89 | 473882:-0.85; 473906:0.00; 473920:-0.29; 473929:-0.79 |
| 4 | 473969:-8.09; 473974:-4.04; 473977:-3.73; 473979:-1.61; 473983:-2.29; 473988:0.00; 473991:-1.36; 473992:-2.33; 473994:-4.29 | 473969:-0.15; 473971:0.00; 473972:-0.37; 473987:-3.53 | 473969:-10.90; 473984:-2.05; 473988:0.00; 473991:-0.86 | 473952:-0.77; 473969:-0.05; 473975:0.00; 473998:-0.80 |
| 5 | 474277:-4.16; 474280:-0.48; 474281:0.00; 474283:-1.45; 474285:-2.91 | 474280:-0.00; 474281:0.00; 474283:-1.12 | 474263:-2.46; 474276:0.00; 474277:-0.77; 474280:-1.78; 474289:-0.97 | 474243:-0.82; 474266:-0.00; 474267:0.00; 474280:-0.27; 474290:-0.82 |
| 6 | 474360:0.00; 474362:-0.27; 474363:-0.77; 474366:-2.28; 474369:-3.10; 474371:-4.14 | 474360:0.00; 474365:-0.20; 474371:-0.41 | 474279:-3.11; 474357:-0.66; 474360:-0.08; 474362:0.00; 474404:-2.55 | 474311:-0.79; 474334:0.00; 474358:-0.85; 474360:-0.99 |
| 7 | 474455:-3.24; 474456:-3.00; 474459:-0.23; 474460:0.00; 474462:-0.07 | 474459:-0.12; 474461:0.00; 474462:0.00 | 474293:-1.88; 474380:-0.27; 474443:0.00; 474448:-0.11; 474459:-2.04 | 474424:-0.89; 474447:-0.00; 474448:0.00; 474459:-0.22; 474468:-0.76 |
| 8 | 474495:-2.83; 474496:0.00; 474497:-0.18; 474499:-0.54; 474502:-0.54; 474504:-0.94; 474505:-1.14 | 474496:0.00; 474500:-0.06; 474504:-0.13 | 474371:-2.33; 474454:-0.17; 474457:0.00; 474488:-0.70; 474496:-1.57 | 474464:-0.81; 474483:0.00; 474496:-0.39; 474501:-0.77 |
| 9 | 474508:-2.64; 474509:-2.65; 474512:0.00; 474513:-0.02; 474514:-0.15; 474515:-0.28; 474517:-0.56 | 474509:-2.79; 474512:0.00; 474514:-0.11; 474517:-0.11 | 474405:-2.62; 474502:-0.41; 474509:0.00; 474531:-0.70 | 474490:-0.84; 474508:0.00; 474509:-0.00; 474525:-0.81 |
| 10 | 474520:-0.21; 474521:-0.27; 474522:-0.34; 474525:-0.20; 474526:-0.10; 474527:0.00; 474528:-0.08; 474529:-0.17; 474543:-3.45; 474550:-3.72 | 474521:-0.04; 474525:-0.02; 474526:-0.01; 474529:0.00 | 474499:-2.00; 474526:-0.73; 474533:-0.46; 474548:-0.21; 474550:0.00 | 474513:-0.79; 474526:-0.06; 474530:-0.00; 474531:0.00; 474546:-0.41 |

**Blind train validation.** Reverse prior uses the CURRENT test proportions, as bands.py does. Existing procedure reproduced 72/3231 wrong band rows; combined cuts give 28/3231. It is not independent calibration: train edges supplied emission/occupancy estimates, and test proportions were themselves regularized by train edges. Bootstrap: 20 multinomial resamples of test device-position evidence (including absent numbers), refit test cuts, then use their proportions for train inference. Coverage below is conditional and cannot remove that circularity.

| edge | truth | legacy_error | combined_error | CI | covered | bootstrap_MAP_range | bootstrap_CI_coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 423656 | 23 | 23 | 423656:423720 | True | 423656:423679 | 20/20 |
| 2 | 424153 | -2 | 2 | 424138:424155 | True | 424155:424155 | 20/20 |
| 3 | 424625 | 5 | 5 | 424624:424630 | True | 424630:424630 | 20/20 |
| 4 | 425033 | -32 | 0 | 425024:425036 | True | 425033:425033 | 20/20 |
| 5 | 426763 | 10 | 6 | 426764:426770 | False | 426769:426769 | 0/20 |
| 6 | 427163 | 7 | 5 | 427165:427170 | False | 427168:427168 | 0/20 |
| 7 | 427844 | 9 | 8 | 427844:427853 | True | 427852:427852 | 20/20 |
| 8 | 428065 | 6 | 4 | 428067:428071 | False | 428069:428071 | 0/20 |
| 9 | 428200 | 0 | -10 | 428190:428212 | True | 428190:428190 | 20/20 |
| 10 | 428302 | -1 | 8 | 428296:428318 | True | 428310:428310 | 20/20 |

**Policies, frozen model probabilities and fixed labels.** All altered band counts and chain bands are rebuilt in memory before applying make_submission-equivalent event rules. Trust uses combined blind cuts. Fallback clears uncertain devices as evidence sources and forces their own final result back to weighted model type. Neighbour takes an adjacent band for an exact model-type match, or a uniquely compatible sibling shape; overlap of sibling families does not force a flip. Confidence intervals define the ambiguous rows; both endpoints included. Model features/probabilities are not recomputed, so these are decision-layer sensitivities, not retrained scores.

| model | policy | clean | raw | changed_vs_oracle |
| --- | --- | --- | --- | --- |
| oof_final_local | oracle/current | 0.559686 | 0.511344 | 0 |
| oof_final_local | legacy blind | 0.557682 | 0.509578 | 95 |
| oof_final_local | combined blind/trust | 0.558574 | 0.510397 | 50 |
| oof_final_local | combined blind/model fallback | 0.559243 | 0.511010 | 33 |
| oof_final_local | combined blind/neighbour | 0.559256 | 0.511023 | 32 |
| s2_a | oracle/current | 0.562701 | 0.513932 | 0 |
| s2_a | legacy blind | 0.560696 | 0.512165 | 94 |
| s2_a | combined blind/trust | 0.561585 | 0.512982 | 50 |
| s2_a | combined blind/model fallback | 0.562011 | 0.513405 | 53 |
| s2_a | combined blind/neighbour | 0.562271 | 0.513611 | 32 |

**Recommendation.** Statistical joint-MAP test cuts: [473632, 473751, 473839, 473920, 473988, 474281, 474360, 474460, 474496, 474512, 474527, 474552]. Recommended deployment cuts: [473632, 473751, 473839, 473920, 473988, 474281, 474360, 474460, 474496, 474512, 474527, 474552]. Keep the trust policy as the conservative recommendation: neighbour correction wins the measured policy comparison, but would also override true generator bands for some sibling-shaped events; its improvement has not been checked on independent blocks. New cuts are recommended only if trust improves blind-train score over legacy on BOTH saved models; otherwise retain current cuts. Blind-score gains: [0.000892, 0.00089]. This is a provisional, reused-validation recommendation, not a proven test improvement. No edges/submission file was written. Recheck intervals with independent blocks before treating their nominal coverage as calibrated.

Test devices changed by the recommended cuts; event counts attribute JOINT changed predictions on other devices through sender/receiver/chain links and can overlap across devices. Total final_local test predictions changed: 41. s2_a has no test probabilities.

| device | old | new | rows | direct_pred_changes | event_rule_changes |
| --- | --- | --- | --- | --- | --- |
| 473750 | 2 | 1 | 1 | 1 | 0 |
| 473826 | 3 | 2 | 1 | 1 | 0 |
| 473828 | 3 | 2 | 1 | 1 | 0 |
| 473829 | 3 | 2 | 1 | 1 | 0 |
| 473830 | 3 | 2 | 1 | 1 | 0 |
| 473833 | 3 | 2 | 1 | 1 | 0 |
| 473835 | 3 | 2 | 1 | 1 | 0 |
| 473838 | 3 | 2 | 1 | 1 | 0 |
| 473969 | 5 | 4 | 2 | 2 | 0 |
| 473970 | 5 | 4 | 1 | 1 | 0 |
| 473971 | 5 | 4 | 3 | 3 | 0 |
| 473972 | 5 | 4 | 2 | 2 | 0 |
| 473973 | 5 | 4 | 3 | 3 | 0 |
| 473974 | 5 | 4 | 2 | 2 | 0 |
| 473976 | 5 | 4 | 2 | 2 | 0 |
| 473977 | 5 | 4 | 1 | 1 | 0 |
| 473978 | 5 | 4 | 2 | 2 | 0 |
| 473980 | 5 | 4 | 2 | 2 | 0 |
| 473981 | 5 | 4 | 4 | 4 | 0 |
| 473983 | 5 | 4 | 2 | 2 | 0 |
| 473984 | 5 | 4 | 2 | 2 | 0 |
| 473986 | 5 | 4 | 2 | 2 | 0 |
| 473987 | 5 | 4 | 1 | 1 | 0 |
| 474511 | 10 | 9 | 1 | 1 | 0 |
| 474526 | 11 | 10 | 1 | 1 | 0 |

**Part 2 - collect-family audit.**

**Leak review (source and no-fit measurements).** `src/upi/features_v6.py:85-92` subtracts the query month from label counts, so its direct label lookup excludes its own label and ALL same-month labels. Reconstructing all 55 label columns below checks the saved table. Holding the detector, bands and donor labels outside the query month fixed, arbitrary replacement of every same-month label has exactly zero direct effect. This is conditional safety, not end-to-end safety. `features_v6.py:46-51` admits foreign-month OOF probabilities from models trained on the query month and test probabilities trained on every month; these affect both flag selection and near/far evidence. Own-month bands also encode label-derived train edges (`bands.py:75-85`), and cleaned donor labels come from a global event vote (`clean_labels.py:62-89`), so indirect label paths remain.

**More serious outer-fold leak:** `run_oof.py:38` loads one global feature table before splitting at lines 65-67. For held-out July, an August training row may use a July donor label even though the July validation row excludes July donors. Removing held-out labels from ALL outer-training rows changes the counts shown below. This can inflate the second-stage gain; nested class weights do not fix it. No-fit test performed: mask each held-out month from label donors and count changed training rows; separately count collect queries exposed to foreign/test probability donors, with band donors separated. Recompute/freeze donor features per OUTER fold and inner-cross-fit upstream predictions for a clean future evaluation. A new fit would be required to estimate the score inflation; these exposure counts cannot quantify it. A stored final predictor, if available, could be scored on scrubbed validation features without fitting as a sensitivity test, but saved OOF arrays alone cannot do that.

Leak table: slash triples are July/August/September. `outer_train_changed` counts available training rows BEFORE legit downsampling with at least one held-out cleaned-fraud-label donor; `collect_unsafe_any_far` is collect-query count with foreign/test non-band probability donors at any distance / >24h; own-month band column excludes self for same-role links. All 55 far-probability columns are reconstructed on collect queries, with maximum float32 rounding error reported. Count reconstruction checks near/far endpoints and self subtraction. **Self-exclusion bug:** 1 flagged self-transfer retains its own near evidence in s2r/r2s because `has_self=False` at lines 66/72; replace the constant with `me = flag & (qkey == pkey) & (qkey >= 0)`. Direct label counts still exclude its month.

| link | lab_mismatch_rows | lab_max_error | near_far_count_error | far_probability_max_error | own_month_perturb | outer_train_changed | collect_unsafe_any_far | collect_ownmonth_band |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| snd | 0 | 0.000000 | 0/0 | 4.75e-07 | 0/0/0 | 39491/37985/38221 | 612/612; 577/576; 576/576 | 122/61/73 |
| rcv | 0 | 0.000000 | 0/0 | 8.48e-07 | 0/0/0 | 295935/301281/326797 | 671/671; 624/624; 624/624 | 40/28/36 |
| dev | 0 | 0.000000 | 0/0 | 4.17e-07 | 0/0/0 | 32466/30657/30934 | 591/591; 556/555; 557/557 | 0/0/0 |
| s2r | 0 | 0.000000 | 0/0 | 2.38e-07 | 0/0/0 | 14185/13547/13511 | 303/302; 234/232; 260/258 | 15/24/13 |
| r2s | 0 | 0.000000 | 0/0 | 7.75e-07 | 0/0/0 | 9304/9044/9073 | 416/413; 417/414; 427/426 | 173/121/115 |

**Train/test support.** Selection: final_local fraud probability >=0.5 OR band>0, AND its band-adjusted weighted prediction in {6,9,10}. This common detector is used for BOTH rule baselines, as for features_v6; no threshold fitted. Month tails use the benchmark September assignment. Counts:

| month | rows | detected | collect | detected_clean_legit |
| --- | --- | --- | --- | --- |
| 2026-07 | 527065 | 8867 | 1722 | 200 |
| 2026-08 | 526852 | 8515 | 1599 | 160 |
| 2026-09 | 510143 | 8386 | 1594 | 176 |
| test | 260582 | 4010 | 822 | NA |

Link distributions include zero-support rows. q fields are median/p90/p99; far_prob_q sums all eleven fraud-probability columns (fraud mass, not an average), followed by individual collect-family label/probability column quantiles. Label counts here follow v6 and exclude cleaned-legit donors. Mean label count per donor month divides train by 2 and test by 3. Far columns use all four months and exclude <=24h, whereas labels use only other training months at ANY distance: they are not comparable exposure windows. October has roughly half the rows but an extra labeled donor month. Nonetheless sender label coverage falls from 37.5-39.6% to 22.7%, device coverage from about 32% to 17.2%; sender count per donor month falls from 0.260-0.297 to 0.221. Receiver coverage instead rises to 43.3% from 34.9-36.9%. Thus a single density correction is inadequate; include donor exposure, count and share separately.

| month | link | n | lab_coverage | lab_q | lab_mean_per_donor_month | far_coverage | far_n_q | far_prob_q | lab6_q | far_p6_q | lab9_q | far_p9_q | lab10_q | far_p10_q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-07 | snd | 1722 | 0.385598 | 0.00/1.00/5.00 | 0.294715 | 0.650407 | 1.00/3.00/9.00 | 0.98/3.00/8.81 | 0.00/1.00/1.00 | 0.00/0.87/1.69 | 0.00/0.00/1.00 | 0.00/0.09/1.84 | 0.00/0.00/1.00 | 0.00/1.48/4.98 |
| 2026-08 | snd | 1599 | 0.395872 | 0.00/1.00/5.00 | 0.297061 | 0.642276 | 1.00/3.00/8.00 | 0.98/2.99/8.00 | 0.00/1.00/2.00 | 0.00/0.88/1.79 | 0.00/0.00/1.00 | 0.00/0.08/0.42 | 0.00/0.00/2.00 | 0.00/1.97/4.92 |
| 2026-09 | snd | 1594 | 0.374529 | 0.00/1.00/5.00 | 0.260038 | 0.604141 | 1.00/3.00/7.00 | 0.98/2.99/6.97 | 0.00/1.00/2.00 | 0.00/0.88/1.83 | 0.00/0.00/1.00 | 0.00/0.08/1.01 | 0.00/0.00/1.00 | 0.00/1.95/4.90 |
| test | snd | 822 | 0.227494 | 0.00/1.00/13.58 | 0.220600 | 0.619221 | 1.00/3.00/14.37 | 0.98/3.00/14.32 | 0.00/0.00/1.00 | 0.00/0.87/1.70 | 0.00/0.00/1.00 | 0.00/0.08/1.78 | 0.00/0.00/0.00 | 0.00/1.91/3.99 |
| 2026-07 | rcv | 1722 | 0.351916 | 0.00/2.00/4.00 | 0.371370 | 0.487224 | 0.00/2.00/8.79 | 0.00/2.00/8.79 | 0.00/1.00/2.00 | 0.00/0.88/2.55 | 0.00/0.00/1.00 | 0.00/0.10/1.04 | 0.00/0.00/2.00 | 0.00/0.06/2.10 |
| 2026-08 | rcv | 1599 | 0.348968 | 0.00/2.00/4.00 | 0.348968 | 0.499687 | 0.00/2.00/5.00 | 0.00/2.00/4.97 | 0.00/1.00/2.00 | 0.00/0.89/2.42 | 0.00/0.00/1.00 | 0.00/0.11/1.08 | 0.00/0.00/2.00 | 0.00/0.08/2.78 |
| 2026-09 | rcv | 1594 | 0.368883 | 0.00/2.00/4.07 | 0.410289 | 0.474279 | 0.00/2.00/8.07 | 0.00/2.00/8.02 | 0.00/1.00/2.00 | 0.00/0.88/2.26 | 0.00/0.00/1.00 | 0.00/0.09/1.01 | 0.00/0.00/2.00 | 0.00/0.06/2.13 |
| test | rcv | 822 | 0.433090 | 0.00/2.00/4.00 | 0.284672 | 0.475669 | 0.00/2.00/5.00 | 0.00/2.00/4.95 | 0.00/1.00/2.00 | 0.00/0.88/2.08 | 0.00/0.00/1.00 | 0.00/0.11/1.00 | 0.00/0.00/3.00 | 0.00/0.05/2.92 |
| 2026-07 | dev | 1722 | 0.321138 | 0.00/1.00/5.00 | 0.230546 | 0.569686 | 1.00/3.00/7.00 | 0.96/2.96/6.53 | 0.00/1.00/1.00 | 0.00/0.87/1.67 | 0.00/0.00/1.00 | 0.00/0.09/1.80 | 0.00/0.00/1.00 | 0.00/0.99/4.98 |
| 2026-08 | dev | 1599 | 0.324578 | 0.00/1.00/4.00 | 0.221388 | 0.574109 | 1.00/3.00/7.00 | 0.96/2.98/6.87 | 0.00/1.00/2.00 | 0.00/0.87/1.58 | 0.00/0.00/1.00 | 0.00/0.08/0.30 | 0.00/0.00/2.00 | 0.00/1.97/4.91 |
| 2026-09 | dev | 1594 | 0.321205 | 0.00/1.00/3.00 | 0.202008 | 0.543915 | 1.00/3.00/6.00 | 0.94/2.97/5.60 | 0.00/1.00/1.00 | 0.00/0.87/1.81 | 0.00/0.00/1.00 | 0.00/0.08/0.83 | 0.00/0.00/1.00 | 0.00/1.93/4.90 |
| test | dev | 822 | 0.171533 | 0.00/1.00/7.79 | 0.134631 | 0.542579 | 1.00/3.00/8.79 | 0.94/2.97/8.58 | 0.00/0.00/1.00 | 0.00/0.87/1.70 | 0.00/0.00/1.00 | 0.00/0.08/1.61 | 0.00/0.00/0.00 | 0.00/1.76/3.99 |
| 2026-07 | s2r | 1722 | 0.160859 | 0.00/1.00/2.79 | 0.111789 | 0.222416 | 0.00/2.00/3.00 | 0.00/1.96/3.00 | 0.00/0.00/2.00 | 0.00/0.56/1.81 | 0.00/0.00/1.00 | 0.00/0.05/1.06 | 0.00/0.00/0.00 | 0.00/0.01/0.07 |
| 2026-08 | s2r | 1599 | 0.126954 | 0.00/1.00/2.00 | 0.087867 | 0.202627 | 0.00/1.00/3.02 | 0.00/1.00/3.01 | 0.00/0.00/1.00 | 0.00/0.39/1.71 | 0.00/0.00/1.00 | 0.00/0.03/0.89 | 0.00/0.00/0.00 | 0.00/0.01/0.05 |
| 2026-09 | s2r | 1594 | 0.144918 | 0.00/1.00/2.00 | 0.106964 | 0.205144 | 0.00/2.00/4.00 | 0.00/1.70/3.89 | 0.00/0.00/1.00 | 0.00/0.60/1.66 | 0.00/0.00/1.00 | 0.00/0.05/1.70 | 0.00/0.00/0.00 | 0.00/0.01/0.06 |
| test | s2r | 822 | 0.163017 | 0.00/1.00/3.00 | 0.087997 | 0.206813 | 0.00/2.00/4.00 | 0.00/1.95/3.98 | 0.00/0.00/1.00 | 0.00/0.19/1.61 | 0.00/0.00/1.00 | 0.00/0.05/1.03 | 0.00/0.00/0.00 | 0.00/0.01/0.05 |
| 2026-07 | r2s | 1722 | 0.282811 | 0.00/2.00/7.00 | 0.278455 | 0.430894 | 0.00/2.00/9.00 | 0.00/2.00/8.94 | 0.00/0.00/1.00 | 0.00/0.04/1.69 | 0.00/0.00/2.00 | 0.00/0.01/1.90 | 0.00/0.00/0.00 | 0.00/0.00/0.01 |
| 2026-08 | r2s | 1599 | 0.328330 | 0.00/2.00/7.02 | 0.316448 | 0.430894 | 0.00/3.00/9.00 | 0.00/2.97/8.99 | 0.00/0.00/2.00 | 0.00/0.11/1.84 | 0.00/0.00/2.00 | 0.00/0.02/1.89 | 0.00/0.00/0.00 | 0.00/0.00/0.01 |
| 2026-09 | r2s | 1594 | 0.318695 | 0.00/2.00/7.00 | 0.306462 | 0.436637 | 0.00/3.00/9.00 | 0.00/2.97/8.98 | 0.00/0.00/2.00 | 0.00/0.05/1.81 | 0.00/0.00/1.00 | 0.00/0.01/1.91 | 0.00/0.00/0.00 | 0.00/0.00/0.01 |
| test | r2s | 822 | 0.383212 | 0.00/2.00/9.00 | 0.296026 | 0.435523 | 0.00/3.00/10.00 | 0.00/2.87/9.96 | 0.00/0.00/2.00 | 0.00/0.12/1.82 | 0.00/0.00/1.79 | 0.00/0.02/1.90 | 0.00/0.00/0.00 | 0.00/0.00/0.02 |

**Rule definition and full scan.** Start from each model after the existing band rules. A query must pass the COMMON detector above and currently predict 6, 9 or 10. Count all detected TRAIN donor rows on the same sender (snd) or device (dev), excluding the query training month; test uses all three months. Include cleaned label 0 in the denominator, unlike v6. T must be the unique modal label, T in {6,9,10}, total donor rows >=m and votes(T)/total >=s; ties abstain. Then replace the prediction with T, including a band row if it qualifies (band changes reported). No raw label, trap or test label enters the rule. No probability/weight recalibration after the rule. This is cross-month voting using globally cleaned labels and already-generated OOF detectors; it inherits their indirect paths, so it is NOT a fully nested feature/model evaluation.

Baselines and scan score cells are pooled/July/August/September macro-F1. P=pooled weight calibration; N=two-month weight calibration applied to the third. Move cells have four groups in that same order, each R/W/X = wrong-to-right / right-to-wrong / wrong-to-wrong. Raw and cleaned assessments share predictions. Scanned 18 settings per model (36 model-settings); selecting the best on these same months adds selection optimism. No significance claim.

| model | mode | clean | raw |
| --- | --- | --- | --- |
| oof_final_local | pooled | 0.559686/0.557417/0.553611/0.564869 | 0.511344/0.511026/0.506996/0.513280 |
| oof_final_local | nested | 0.558918/0.555141/0.553138/0.564901 | 0.510583/0.508809/0.506485/0.513327 |
| s2_a | pooled | 0.562701/0.559609/0.558007/0.567176 | 0.513932/0.512476/0.510350/0.516082 |
| s2_a | nested | 0.561506/0.558497/0.558208/0.564338 | 0.512919/0.511366/0.510596/0.513809 |

| model | link | m | s | P_clean | P_raw | P_moves_clean | P_moves_raw | N_clean | N_raw | N_moves_clean | N_moves_raw | keep |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oof_final_local | snd | 1 | 0.600000 | 0.561188/0.558781/0.555725/0.565874 | 0.513445/0.512922/0.509298/0.515364 | 90/66/10; 32/29/3; 30/16/3; 28/21/4 | 90/46/30; 31/22/11; 28/13/8; 31/11/11 | 0.560423/0.556512/0.555314/0.565843 | 0.512686/0.510712/0.508849/0.515350 | 90/66/10; 32/29/3; 31/16/3; 27/21/4 | 90/46/30; 31/22/11; 29/13/8; 30/11/11 | True |
| oof_final_local | snd | 1 | 0.800000 | 0.561378/0.559053/0.555954/0.565946 | 0.513603/0.513095/0.509536/0.515438 | 89/58/9; 32/25/3; 30/14/2; 27/19/4 | 89/40/27; 31/20/9; 28/11/7; 30/9/11 | 0.560613/0.556782/0.555543/0.565916 | 0.512845/0.510885/0.509086/0.515425 | 89/58/9; 32/25/3; 31/14/2; 26/19/4 | 89/40/27; 31/20/9; 29/11/7; 29/9/11 | True |
| oof_final_local | snd | 1 | 1.000000 | 0.561405/0.559053/0.556035/0.565946 | 0.513630/0.513095/0.509614/0.515438 | 89/57/9; 32/25/3; 30/13/2; 27/19/4 | 89/39/27; 31/20/9; 28/10/7; 30/9/11 | 0.560640/0.556782/0.555624/0.565916 | 0.512871/0.510885/0.509164/0.515425 | 89/57/9; 32/25/3; 31/13/2; 26/19/4 | 89/39/27; 31/20/9; 29/10/7; 29/9/11 | True |
| oof_final_local | snd | 2 | 0.600000 | 0.559263/0.556959/0.553093/0.564582 | 0.511038/0.510663/0.506614/0.513106 | 3/17/1; 0/7/0; 0/5/1; 3/5/0 | 3/13/5; 0/5/2; 0/4/2; 3/4/1 | 0.558495/0.554684/0.552621/0.564611 | 0.510278/0.508447/0.506103/0.513151 | 3/17/1; 0/7/0; 0/5/1; 3/5/0 | 3/13/5; 0/5/2; 0/4/2; 3/4/1 | False |
| oof_final_local | snd | 2 | 0.800000 | 0.559454/0.557227/0.553327/0.564659 | 0.511199/0.510833/0.506856/0.513187 | 2/9/0; 0/3/0; 0/3/0; 2/3/0 | 2/7/2; 0/3/0; 0/2/1; 2/2/1 | 0.558687/0.554952/0.552854/0.564689 | 0.510438/0.508617/0.506344/0.513233 | 2/9/0; 0/3/0; 0/3/0; 2/3/0 | 2/7/2; 0/3/0; 0/2/1; 2/2/1 | False |
| oof_final_local | snd | 2 | 1.000000 | 0.559481/0.557227/0.553407/0.564659 | 0.511225/0.510833/0.506933/0.513187 | 2/8/0; 0/3/0; 0/2/0; 2/3/0 | 2/6/2; 0/3/0; 0/1/1; 2/2/1 | 0.558714/0.554952/0.552934/0.564689 | 0.510464/0.508617/0.506422/0.513233 | 2/8/0; 0/3/0; 0/2/0; 2/3/0 | 2/6/2; 0/3/0; 0/1/1; 2/2/1 | False |
| oof_final_local | snd | 3 | 0.600000 | 0.559440/0.557149/0.553218/0.564793 | 0.511131/0.510856/0.506600/0.513200 | 1/10/1; 0/4/0; 0/4/1; 1/2/0 | 1/8/3; 0/2/2; 0/4/1; 1/2/0 | 0.558673/0.554873/0.552745/0.564824 | 0.510371/0.508639/0.506089/0.513246 | 1/10/1; 0/4/0; 0/4/1; 1/2/0 | 1/8/3; 0/2/2; 0/4/1; 1/2/0 | False |
| oof_final_local | snd | 3 | 0.800000 | 0.559632/0.557417/0.553451/0.564869 | 0.511292/0.511026/0.506841/0.513280 | 0/2/0; 0/0/0; 0/2/0; 0/0/0 | 0/2/0; 0/0/0; 0/2/0; 0/0/0 | 0.558864/0.555141/0.552978/0.564901 | 0.510531/0.508809/0.506330/0.513327 | 0/2/0; 0/0/0; 0/2/0; 0/0/0 | 0/2/0; 0/0/0; 0/2/0; 0/0/0 | False |
| oof_final_local | snd | 3 | 1.000000 | 0.559659/0.557417/0.553531/0.564869 | 0.511318/0.511026/0.506919/0.513280 | 0/1/0; 0/0/0; 0/1/0; 0/0/0 | 0/1/0; 0/0/0; 0/1/0; 0/0/0 | 0.558891/0.555141/0.553058/0.564901 | 0.510557/0.508809/0.506407/0.513327 | 0/1/0; 0/0/0; 0/1/0; 0/0/0 | 0/1/0; 0/0/0; 0/1/0; 0/0/0 | False |
| oof_final_local | dev | 1 | 0.600000 | 0.561047/0.558600/0.555356/0.565988 | 0.513313/0.512781/0.508918/0.515482 | 89/68/9; 31/29/2; 29/18/3; 29/21/4 | 89/47/30; 30/21/11; 27/15/8; 32/11/11 | 0.560281/0.556330/0.554946/0.565957 | 0.512555/0.510570/0.508470/0.515469 | 89/68/9; 31/29/2; 30/18/3; 28/21/4 | 89/47/30; 30/21/11; 28/15/8; 31/11/11 | True |
| oof_final_local | dev | 1 | 0.800000 | 0.561214/0.558679/0.555725/0.566059 | 0.513467/0.512808/0.509298/0.515556 | 88/62/9; 31/28/2; 29/15/3; 28/19/4 | 88/42/29; 30/21/10; 27/12/8; 31/9/11 | 0.560449/0.556408/0.555314/0.566029 | 0.512709/0.510597/0.508849/0.515544 | 88/62/9; 31/28/2; 30/15/3; 27/19/4 | 88/42/29; 30/21/10; 28/12/8; 30/9/11 | True |
| oof_final_local | dev | 1 | 1.000000 | 0.561242/0.558679/0.555806/0.566059 | 0.513494/0.512808/0.509377/0.515556 | 88/61/9; 31/28/2; 29/14/3; 28/19/4 | 88/41/29; 30/21/10; 27/11/8; 31/9/11 | 0.560476/0.556408/0.555395/0.566029 | 0.512735/0.510597/0.508927/0.515544 | 88/61/9; 31/28/2; 30/14/3; 27/19/4 | 88/41/29; 30/21/10; 28/11/8; 30/9/11 | True |
| oof_final_local | dev | 2 | 0.600000 | 0.559185/0.556959/0.552719/0.564698 | 0.510957/0.510663/0.506228/0.513228 | 3/19/1; 0/7/0; 0/8/1; 3/4/0 | 3/15/5; 0/5/2; 0/7/2; 3/3/1 | 0.558417/0.554684/0.552248/0.564728 | 0.510197/0.508447/0.505719/0.513273 | 3/19/1; 0/7/0; 0/8/1; 3/4/0 | 3/15/5; 0/5/2; 0/7/2; 3/3/1 | False |
| oof_final_local | dev | 2 | 0.800000 | 0.559355/0.557037/0.553093/0.564774 | 0.511115/0.510690/0.506614/0.513308 | 2/13/1; 0/6/0; 0/5/1; 2/2/0 | 2/10/4; 0/5/1; 0/4/2; 2/1/1 | 0.558588/0.554762/0.552621/0.564805 | 0.510354/0.508474/0.506103/0.513354 | 2/13/1; 0/6/0; 0/5/1; 2/2/0 | 2/10/4; 0/5/1; 0/4/2; 2/1/1 | False |
| oof_final_local | dev | 2 | 1.000000 | 0.559382/0.557037/0.553173/0.564774 | 0.511141/0.510690/0.506691/0.513308 | 2/12/1; 0/6/0; 0/4/1; 2/2/0 | 2/9/4; 0/5/1; 0/3/2; 2/1/1 | 0.558615/0.554762/0.552701/0.564805 | 0.510380/0.508474/0.506181/0.513354 | 2/12/1; 0/6/0; 0/4/1; 2/2/0 | 2/9/4; 0/5/1; 0/3/2; 2/1/1 | False |
| oof_final_local | dev | 3 | 0.600000 | 0.559461/0.557339/0.553078/0.564793 | 0.511134/0.510999/0.506457/0.513200 | 1/8/0; 0/1/0; 0/5/0; 1/2/0 | 1/7/1; 0/0/1; 0/5/0; 1/2/0 | 0.558694/0.555063/0.552606/0.564824 | 0.510374/0.508782/0.505947/0.513246 | 1/8/0; 0/1/0; 0/5/0; 1/2/0 | 1/7/1; 0/0/1; 0/5/0; 1/2/0 | False |
| oof_final_local | dev | 3 | 0.800000 | 0.559632/0.557417/0.553451/0.564869 | 0.511292/0.511026/0.506841/0.513280 | 0/2/0; 0/0/0; 0/2/0; 0/0/0 | 0/2/0; 0/0/0; 0/2/0; 0/0/0 | 0.558864/0.555141/0.552978/0.564901 | 0.510531/0.508809/0.506330/0.513327 | 0/2/0; 0/0/0; 0/2/0; 0/0/0 | 0/2/0; 0/0/0; 0/2/0; 0/0/0 | False |
| oof_final_local | dev | 3 | 1.000000 | 0.559659/0.557417/0.553531/0.564869 | 0.511318/0.511026/0.506919/0.513280 | 0/1/0; 0/0/0; 0/1/0; 0/0/0 | 0/1/0; 0/0/0; 0/1/0; 0/0/0 | 0.558891/0.555141/0.553058/0.564901 | 0.510557/0.508809/0.506407/0.513327 | 0/1/0; 0/0/0; 0/1/0; 0/0/0 | 0/1/0; 0/0/0; 0/1/0; 0/0/0 | False |
| s2_a | snd | 1 | 0.600000 | 0.562005/0.558935/0.557307/0.566455 | 0.513365/0.511957/0.509789/0.515453 | 21/44/2; 10/23/0; 5/11/1; 6/10/1 | 18/37/12; 8/19/6; 4/9/4; 6/9/2 | 0.560765/0.557602/0.557443/0.563794 | 0.512267/0.510563/0.509905/0.513360 | 18/41/2; 8/22/0; 4/10/1; 6/9/1 | 15/36/10; 6/19/5; 3/9/3; 6/8/2 | False |
| s2_a | snd | 1 | 0.800000 | 0.562171/0.559125/0.557539/0.566534 | 0.513519/0.512101/0.510030/0.515534 | 20/37/1; 10/20/0; 5/9/0; 5/8/1 | 17/31/10; 8/17/5; 4/7/3; 5/7/2 | 0.560983/0.557792/0.557673/0.564027 | 0.512475/0.510707/0.510144/0.513603 | 18/34/1; 8/19/0; 4/8/0; 6/7/1 | 15/30/8; 6/17/4; 3/7/2; 6/6/2 | False |
| s2_a | snd | 1 | 1.000000 | 0.562171/0.559125/0.557539/0.566534 | 0.513519/0.512101/0.510030/0.515534 | 20/37/1; 10/20/0; 5/9/0; 5/8/1 | 17/31/10; 8/17/5; 4/7/3; 5/7/2 | 0.560983/0.557792/0.557673/0.564027 | 0.512475/0.510707/0.510144/0.513603 | 18/34/1; 8/19/0; 4/8/0; 6/7/1 | 15/30/8; 6/17/4; 3/7/2; 6/6/2 | False |
| s2_a | snd | 2 | 0.600000 | 0.562434/0.559345/0.557654/0.567000 | 0.513721/0.512124/0.510127/0.516030 | 4/13/1; 1/6/0; 0/3/1; 3/4/0 | 3/10/5; 0/5/2; 0/2/2; 3/3/1 | 0.561187/0.558232/0.557858/0.564012 | 0.512654/0.511013/0.510374/0.513599 | 3/13/1; 1/6/0; 0/3/1; 2/4/0 | 2/10/5; 0/5/2; 0/2/2; 2/3/1 | False |
| s2_a | snd | 2 | 0.800000 | 0.562601/0.559534/0.557884/0.567079 | 0.513875/0.512268/0.510366/0.516111 | 3/6/0; 1/3/0; 0/1/0; 2/2/0 | 2/4/3; 0/3/1; 0/0/1; 2/1/1 | 0.561405/0.558423/0.558086/0.564244 | 0.512861/0.511157/0.510611/0.513841 | 3/6/0; 1/3/0; 0/1/0; 2/2/0 | 2/4/3; 0/3/1; 0/0/1; 2/1/1 | False |
| s2_a | snd | 2 | 1.000000 | 0.562601/0.559534/0.557884/0.567079 | 0.513875/0.512268/0.510366/0.516111 | 3/6/0; 1/3/0; 0/1/0; 2/2/0 | 2/4/3; 0/3/1; 0/0/1; 2/1/1 | 0.561405/0.558423/0.558086/0.564244 | 0.512861/0.511157/0.510611/0.513841 | 3/6/0; 1/3/0; 0/1/0; 2/2/0 | 2/4/3; 0/3/1; 0/0/1; 2/1/1 | False |
| s2_a | snd | 3 | 0.600000 | 0.562535/0.559419/0.557778/0.567098 | 0.513779/0.512332/0.510112/0.516001 | 1/7/1; 0/3/0; 0/2/1; 1/2/0 | 1/6/2; 0/2/1; 0/2/1; 1/2/0 | 0.561288/0.558306/0.557980/0.564106 | 0.512711/0.511222/0.510359/0.513568 | 0/7/1; 0/3/0; 0/2/1; 0/2/0 | 0/6/2; 0/2/1; 0/2/1; 0/2/0 | False |
| s2_a | snd | 3 | 0.800000 | 0.562701/0.559609/0.558007/0.567176 | 0.513932/0.512476/0.510350/0.516082 | 0/0/0; 0/0/0; 0/0/0; 0/0/0 | 0/0/0; 0/0/0; 0/0/0; 0/0/0 | 0.561506/0.558497/0.558208/0.564338 | 0.512919/0.511366/0.510596/0.513809 | 0/0/0; 0/0/0; 0/0/0; 0/0/0 | 0/0/0; 0/0/0; 0/0/0; 0/0/0 | False |
| s2_a | snd | 3 | 1.000000 | 0.562701/0.559609/0.558007/0.567176 | 0.513932/0.512476/0.510350/0.516082 | 0/0/0; 0/0/0; 0/0/0; 0/0/0 | 0/0/0; 0/0/0; 0/0/0; 0/0/0 | 0.561506/0.558497/0.558208/0.564338 | 0.512919/0.511366/0.510596/0.513809 | 0/0/0; 0/0/0; 0/0/0; 0/0/0 | 0/0/0; 0/0/0; 0/0/0; 0/0/0 | False |
| s2_a | dev | 1 | 0.600000 | 0.561850/0.558718/0.556932/0.566572 | 0.513204/0.511729/0.509402/0.515575 | 20/46/2; 9/23/0; 4/13/1; 7/10/1 | 17/39/12; 7/19/6; 3/11/4; 7/9/2 | 0.560611/0.557381/0.557070/0.563911 | 0.512105/0.510330/0.509522/0.513482 | 17/43/2; 7/22/0; 3/12/1; 7/9/1 | 14/38/10; 5/19/5; 2/11/3; 7/8/2 | False |
| s2_a | dev | 1 | 0.800000 | 0.561997/0.558718/0.557307/0.566651 | 0.513356/0.511729/0.509789/0.515656 | 19/41/2; 9/23/0; 4/10/1; 6/8/1 | 16/34/12; 7/19/6; 3/8/4; 6/7/2 | 0.560809/0.557381/0.557443/0.564144 | 0.512311/0.510330/0.509905/0.513725 | 17/38/2; 7/22/0; 3/9/1; 7/7/1 | 14/33/10; 5/19/5; 2/8/3; 7/6/2 | False |
| s2_a | dev | 1 | 1.000000 | 0.561997/0.558718/0.557307/0.566651 | 0.513356/0.511729/0.509789/0.515656 | 19/41/2; 9/23/0; 4/10/1; 6/8/1 | 16/34/12; 7/19/6; 3/8/4; 6/7/2 | 0.560809/0.557381/0.557443/0.564144 | 0.512311/0.510330/0.509905/0.513725 | 17/38/2; 7/22/0; 3/9/1; 7/7/1 | 14/33/10; 5/19/5; 2/8/3; 7/6/2 | False |
| s2_a | dev | 2 | 0.600000 | 0.562316/0.559345/0.557282/0.567000 | 0.513599/0.512124/0.509745/0.516030 | 4/16/1; 1/6/0; 0/6/1; 3/4/0 | 3/13/5; 0/5/2; 0/5/2; 3/3/1 | 0.561069/0.558232/0.557488/0.564012 | 0.512531/0.511013/0.509994/0.513599 | 3/16/1; 1/6/0; 0/6/1; 2/4/0 | 2/13/5; 0/5/2; 0/5/2; 2/3/1 | False |
| s2_a | dev | 2 | 0.800000 | 0.562462/0.559345/0.557654/0.567079 | 0.513749/0.512124/0.510127/0.516111 | 3/11/1; 1/6/0; 0/3/1; 2/2/0 | 2/8/5; 0/5/2; 0/2/2; 2/1/1 | 0.561266/0.558232/0.557858/0.564244 | 0.512735/0.511013/0.510374/0.513841 | 3/11/1; 1/6/0; 0/3/1; 2/2/0 | 2/8/5; 0/5/2; 0/2/2; 2/1/1 | False |
| s2_a | dev | 2 | 1.000000 | 0.562462/0.559345/0.557654/0.567079 | 0.513749/0.512124/0.510127/0.516111 | 3/11/1; 1/6/0; 0/3/1; 2/2/0 | 2/8/5; 0/5/2; 0/2/2; 2/1/1 | 0.561266/0.558232/0.557858/0.564244 | 0.512735/0.511013/0.510374/0.513841 | 3/11/1; 1/6/0; 0/3/1; 2/2/0 | 2/8/5; 0/5/2; 0/2/2; 2/1/1 | False |
| s2_a | dev | 3 | 0.600000 | 0.562555/0.559609/0.557637/0.567098 | 0.513782/0.512476/0.509969/0.516001 | 1/5/0; 0/0/0; 0/3/0; 1/2/0 | 1/5/0; 0/0/0; 0/3/0; 1/2/0 | 0.561309/0.558497/0.557841/0.564106 | 0.512715/0.511366/0.510217/0.513568 | 0/5/0; 0/0/0; 0/3/0; 0/2/0 | 0/5/0; 0/0/0; 0/3/0; 0/2/0 | False |
| s2_a | dev | 3 | 0.800000 | 0.562701/0.559609/0.558007/0.567176 | 0.513932/0.512476/0.510350/0.516082 | 0/0/0; 0/0/0; 0/0/0; 0/0/0 | 0/0/0; 0/0/0; 0/0/0; 0/0/0 | 0.561506/0.558497/0.558208/0.564338 | 0.512919/0.511366/0.510596/0.513809 | 0/0/0; 0/0/0; 0/0/0; 0/0/0 | 0/0/0; 0/0/0; 0/0/0; 0/0/0 | False |
| s2_a | dev | 3 | 1.000000 | 0.562701/0.559609/0.558007/0.567176 | 0.513932/0.512476/0.510350/0.516082 | 0/0/0; 0/0/0; 0/0/0; 0/0/0 | 0/0/0; 0/0/0; 0/0/0; 0/0/0 | 0.561506/0.558497/0.558208/0.564338 | 0.512919/0.511366/0.510596/0.513809 | 0/0/0; 0/0/0; 0/0/0; 0/0/0 | 0/0/0; 0/0/0; 0/0/0; 0/0/0 | False |

**Verdict.** Select the highest nested pooled gain among settings meeting BOTH gain >=0.001 and no negative cleaned-label month; if none qualify, show the highest-gain rejected setting. The winning final_local sender rule changes 155 train rows: 89 corrections, 57 regressions and 9 wrong-to-wrong under cleaned labels (raw: 89/39/27). It gains 0.001722 nested, not 0.010; almost all its useful support is single-donor history, and demanding >=2 donors makes the sender rule harmful. On s2_a the same unanimous rule corrects 18 but damages 34 nested rows, with 1 wrong-to-wrong; all positive-change settings fail. The best rejected settings make no changes. This supports keeping the explicit rule only on final_local under the requested benchmark, not adding it to s2_a or interpreting s2_a as a leakage-free improvement.

| model | setting | nested_clean | nested_gain | band_rows_changed | verdict |
| --- | --- | --- | --- | --- | --- |
| oof_final_local | snd,m=1,s=1.0 | 0.560640/0.556782/0.555624/0.565916 | 0.001722/0.001641/0.002486/0.001015 | 4 | KEEP |
| s2_a | snd,m=3,s=0.8 | 0.561506/0.558497/0.558208/0.564338 | 0.000000/0.000000/0.000000/0.000000 | 0 | DROP |

Test changes for the selected final_local setting (counterfactual if verdict is DROP), with current edges/band rules unchanged to isolate the history rule:

| model | from_type | to_type | rows |
| --- | --- | --- | --- |
| oof_final_local | phishing_collect | fake_cashback | 4 |
| oof_final_local | fake_cashback | phishing_collect | 7 |

`models/s2_a` contains OOF probabilities only: its selected rule has **no computable test transition count** without test probabilities. Transferring final_local predictions would not be an s2_a result. No model fitted and no submission written.

**Reproduce.** Run `C:/Users/ASUS/AppData/Local/Programs/Python/Python311/python.exe -I scripts/explore/44_codex_test_edges.py` then the same executable with `-I scripts/explore/45_codex_collect_rule.py`. Both cap numerical threads at 4 and write only their sections of this new report. Source artifacts read: base_v0, features_v1/v3/v5/v6, band_edges.json, s1 probabilities, final_local probabilities/saved weights, s2_a OOF, and oof_raw/y_clean.npy. `44` also exposes the complete discrete posterior arrays in `posterior()`. No files under data/, models/ or submissions/ are written.
