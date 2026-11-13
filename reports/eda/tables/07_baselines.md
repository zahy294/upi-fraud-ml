## Oracle ceiling (Sep): perfectly label every on-pattern row, call the rest legit
macro-F1 = **0.4543**
|  | precision | recall | f1 | support |
|---|---|---|---|---|
| legit | 0.973 | 1 | 0.986 | 4.91e+05 |
| mule_account | 1 | 0.261 | 0.414 | 3.37e+03 |
| velocity_bot | 1 | 0.347 | 0.515 | 2.66e+03 |
| layering_chain | 1 | 0.217 | 0.357 | 2.18e+03 |
| sim_swap | 1 | 0.262 | 0.415 | 1.27e+03 |
| account_takeover | 1 | 0.252 | 0.403 | 1.28e+03 |
| phishing_collect | 1 | 0.208 | 0.345 | 2.16e+03 |
| qr_swap | 1 | 0.248 | 0.398 | 2.35e+03 |
| refund_abuse | 1 | 0.303 | 0.465 | 1.08e+03 |
| fake_cashback | 1 | 0.236 | 0.382 | 831 |
| loan_app_extortion | 1 | 0.24 | 0.388 | 645 |
| merchant_collusion | 1 | 0.238 | 0.384 | 841 |

## Hand-written rules only (no model), Sep
macro-F1 = **0.3922**
|  | precision | recall | f1 | support |
|---|---|---|---|---|
| legit | 0.974 | 0.995 | 0.984 | 4.91e+05 |
| mule_account | 0.576 | 0.209 | 0.307 | 3.37e+03 |
| velocity_bot | 0.846 | 0.339 | 0.484 | 2.66e+03 |
| layering_chain | 0.726 | 0.217 | 0.334 | 2.18e+03 |
| sim_swap | 0.302 | 0.238 | 0.266 | 1.27e+03 |
| account_takeover | 0.294 | 0.229 | 0.258 | 1.28e+03 |
| phishing_collect | 0.556 | 0.206 | 0.301 | 2.16e+03 |
| qr_swap | 0.457 | 0.248 | 0.322 | 2.35e+03 |
| refund_abuse | 0.815 | 0.303 | 0.442 | 1.08e+03 |
| fake_cashback | 0.558 | 0.236 | 0.332 | 831 |
| loan_app_extortion | 0.82 | 0.233 | 0.362 | 645 |
| merchant_collusion | 0.466 | 0.238 | 0.315 | 841 |

Rule predictions on test (counts): {'legit': 256601, 'mule_account': 695, 'qr_swap': 583, 'sim_swap': 511, 'account_takeover': 504, 'phishing_collect': 456, 'layering_chain': 314, 'velocity_bot': 270, 'merchant_collusion': 197, 'refund_abuse': 190, 'fake_cashback': 172, 'loan_app_extortion': 89}

## Do ip_risk_score / app_version still separate rule-hit rows from the rest in October?
('oldest app ver' = 3.x in Sep, 5.0 in Oct)
| set | rule-hit rows | ip mean | rule hit | ip mean | no hit | ip>0.4 % | rule hit | ip>0.4 % | no hit | oldest app ver % | hit | oldest app ver % | no hit |
|---|---|---|---|---|---|---|---|
| Sep | 8372 | 0.369 | 0.201 | 42.55 | 7.148 | 52.04 | 15.08 |
| Oct (test) | 3981 | 0.232 | 0.22 | 6.431 | 3.967 | 34.41 | 33.45 |

Oct app_version share by rule hit (row %):
| row_0 | 5.0 | 5.1 | 5.2 |
|---|---|---|---|
| False | 33.45 | 33.44 | 33.11 |
| True | 34.41 | 32.43 | 33.16 |

## LightGBM ablation (train Jul-Aug, validate Sep; macro-F1 with best shared fraud up-weight k)
- A: single-row features incl. ip_risk_score & app_version: Sep as-is: **0.3487** (k=8); Sep with Oct-like traps: **0.3404** (k=12)
- B: single-row features, traps removed: Sep as-is: **0.3391** (k=8)
- C: B + relational features: Sep as-is: **0.3910** (k=3)
- D: C + traps: Sep as-is: **0.4022** (k=5); Sep with Oct-like traps: **0.4037** (k=3)

### Model C per-class (Sep, k=3)
|  | precision | recall | f1 | support |
|---|---|---|---|---|
| legit | 0.975 | 0.995 | 0.985 | 4.91e+05 |
| mule_account | 0.647 | 0.247 | 0.358 | 3.37e+03 |
| velocity_bot | 0.758 | 0.286 | 0.415 | 2.66e+03 |
| layering_chain | 0.728 | 0.35 | 0.473 | 2.18e+03 |
| sim_swap | 0.425 | 0.192 | 0.264 | 1.27e+03 |
| account_takeover | 0.338 | 0.197 | 0.249 | 1.28e+03 |
| phishing_collect | 0.603 | 0.291 | 0.393 | 2.16e+03 |
| qr_swap | 0.751 | 0.312 | 0.441 | 2.35e+03 |
| refund_abuse | 0.43 | 0.285 | 0.342 | 1.08e+03 |
| fake_cashback | 0.429 | 0.202 | 0.275 | 831 |
| loan_app_extortion | 0.332 | 0.186 | 0.239 | 645 |
| merchant_collusion | 0.378 | 0.196 | 0.258 | 841 |

### Model C top features by gain %
|  | gain_% |
|---|---|
| rcv_cnt_24h | 12.5 |
| amount | 11.9 |
| login_to_txn_sec | 9.9 |
| mer_age_days | 5.52 |
| device_age_days | 5.24 |
| amt_to_balance | 4.67 |
| balance_before | 4.4 |
| rcv_cnt_1h | 4.09 |
| rcv_round500_cnt_7d | 3.61 |
| snd_out_cnt_10m | 3.06 |
| sim_age_hours | 2.8 |
| rcv_cnt_total | 2.75 |
| amt_is_integer | 2.65 |
| pair_cnt_total | 2.54 |
| pair_share_of_rcv | 2.47 |
| sender_account_age_days | 2.33 |
| rcv_newpayee_share_day | 2.31 |
| hour | 2.24 |
| pair_cnt_7d | 1.73 |
| rcv_uniq_snd_total | 1.65 |
| weekday | 1.53 |
| snd_out_cnt_24h | 1.52 |
| snd_collect_cnt_7d | 1.18 |
| rcv_acct_age_days | 0.99 |
| snd_refund_cnt_total | 0.8 |