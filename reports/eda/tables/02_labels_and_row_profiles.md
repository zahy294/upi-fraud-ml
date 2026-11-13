## Label distribution
| label | count | pct | pct_of_fraud | pct_2026-07 | pct_2026-08 | pct_2026-09 |
|---|---|---|---|---|---|---|
| legit | 1.506e+06 | 96.31 |  | 96.3 | 96.33 | 96.34 |
| mule_account | 1.026e+04 | 0.656 | 17.8 | 0.649 | 0.654 | 0.66 |
| velocity_bot | 8106 | 0.518 | 14.1 | 0.538 | 0.496 | 0.521 |
| layering_chain | 6822 | 0.436 | 11.8 | 0.441 | 0.439 | 0.428 |
| sim_swap | 4072 | 0.26 | 7.1 | 0.268 | 0.264 | 0.248 |
| account_takeover | 4049 | 0.259 | 7 | 0.272 | 0.253 | 0.251 |
| phishing_collect | 6697 | 0.428 | 11.6 | 0.433 | 0.428 | 0.424 |
| qr_swap | 7264 | 0.464 | 12.6 | 0.459 | 0.472 | 0.461 |
| refund_abuse | 3305 | 0.211 | 5.7 | 0.198 | 0.212 | 0.211 |
| fake_cashback | 2532 | 0.162 | 4.4 | 0.168 | 0.154 | 0.163 |
| loan_app_extortion | 1991 | 0.127 | 3.5 | 0.129 | 0.125 | 0.126 |
| merchant_collusion | 2549 | 0.163 | 4.4 | 0.148 | 0.174 | 0.165 |

Daily rows: train median 16968, test median 8364. Fraud rows/day in train: median 614, min 3, max 718.

## Numeric medians by label
| label | amount | balance_before | amt_to_balance | sim_age_hours | device_age_days | sender_account_age_days | acct_age_from_opened | login_to_txn_sec | pin_retries | ip_risk_score |
|---|---|---|---|---|---|---|---|---|---|---|
| legit | 460 | 6274 | 0.07641 | 6214 | 220.5 | 590 | 589 | 45 | 0 | 0.18 |
| mule_account | 873.7 | 6802 | 0.1281 | 6313 | 218.5 | 591 | 591 | 46 | 0 | 0.215 |
| velocity_bot | 356 | 6350 | 0.06055 | 6332 | 94.55 | 589 | 589 | 27 | 0 | 0.233 |
| layering_chain | 1336 | 1.068e+04 | 0.2195 | 6124 | 218.8 | 574 | 574 | 45 | 0 | 0.238 |
| sim_swap | 628.9 | 4395 | 0.2241 | 3733 | 191.6 | 597.5 | 597 | 30 | 0 | 0.227 |
| account_takeover | 636 | 4594 | 0.2255 | 5439 | 143.6 | 589 | 590 | 40 | 0 | 0.232 |
| phishing_collect | 768.8 | 7013 | 0.1127 | 6006 | 219.6 | 595 | 595 | 44 | 0 | 0.232 |
| qr_swap | 550 | 6506 | 0.08893 | 6517 | 224.4 | 589 | 589 | 44 | 0 | 0.2295 |
| refund_abuse | 1017 | 6240 | 0.1379 | 6176 | 218.5 | 573 | 573 | 46 | 0 | 0.23 |
| fake_cashback | 465.4 | 7292 | 0.07725 | 6299 | 225.2 | 605.5 | 605.5 | 43 | 0 | 0.2275 |
| loan_app_extortion | 925.5 | 7915 | 0.143 | 5864 | 220 | 582 | 582 | 45 | 0 | 0.216 |
| merchant_collusion | 745.9 | 6731 | 0.11 | 5999 | 220.3 | 562 | 562 | 45 | 0 | 0.214 |

## Means (binary features => rates) by label
| label | new_payee | qr_vpa_mismatch | away_from_home | home_device | is_night | is_late_evening | rcv_acct_age_days | mer_age_days | mer_cat_matches_txn | mer_city_eq_device_city |
|---|---|---|---|---|---|---|---|---|---|---|
| legit | 0.227 | 0.00325 | 0.068 | 0.921 | 0.0318 | 0.2 | 889 | 967 | 1 | 0.0158 |
| mule_account | 0.395 | 0.00376 | 0.066 | 0.924 | 0.0894 | 0.209 | 868 | 958 | 1 | 0.0155 |
| velocity_bot | 0.254 | 0.005 | 0.0693 | 0.599 | 0.0544 | 0.185 | 904 | 982 | 1 | 0.0164 |
| layering_chain | 0.326 | 0.00613 | 0.0676 | 0.92 | 0.0726 | 0.214 | 895 | 978 | 1 | 0.0133 |
| sim_swap | 0.498 | 0.00465 | 0.153 | 0.84 | 0.225 | 0.162 | 889 | 957 | 1 | 0.0149 |
| account_takeover | 0.519 | 0.00557 | 0.272 | 0.708 | 0.107 | 0.198 | 880 | 952 | 1 | 0.0106 |
| phishing_collect | 0.45 | 0.00703 | 0.0751 | 0.922 | 0.0321 | 0.204 | 878 | 953 | 1 | 0.0177 |
| qr_swap | 0.226 | 0.3 | 0.0687 | 0.922 | 0.0479 | 0.205 | 886 | 1.02e+03 | 1 | 0.0176 |
| refund_abuse | 0.229 | 0.00517 | 0.069 | 0.926 | 0.12 | 0.189 | 920 | 1.04e+03 | 1 | 0.0174 |
| fake_cashback | 0.342 | 0.00542 | 0.0624 | 0.918 | 0.0466 | 0.209 | 923 | 961 | 1 | 0.0216 |
| loan_app_extortion | 0.455 | 0.0122 | 0.0884 | 0.901 | 0.0261 | 0.405 | 881 | 1.09e+03 | 1 | 0.0173 |
| merchant_collusion | 0.244 | 0.0316 | 0.0632 | 0.925 | 0.0843 | 0.189 | 908 | 980 | 1 | 0.0157 |

## p10 by label
| label | amount | sim_age_hours | device_age_days | login_to_txn_sec | amt_to_balance | rcv_acct_age_days |
|---|---|---|---|---|---|---|
| legit | 91.56 | 906.8 | 31.2 | 14 | 0.01508 | 197 |
| mule_account | 125.4 | 914.1 | 29.1 | 14 | 0.01998 | 181 |
| velocity_bot | 80 | 890 | 12.9 | 3 | 0.01014 | 200.3 |
| layering_chain | 133.1 | 890.5 | 32 | 14 | 0.02262 | 190 |
| sim_swap | 131.4 | 12.5 | 3.3 | 9 | 0.02132 | 189 |
| account_takeover | 132.7 | 35.18 | 0.6 | 11 | 0.02077 | 184 |
| phishing_collect | 10 | 857.8 | 30.1 | 14 | 0.00403 | 199.5 |
| qr_swap | 100 | 963.4 | 32.16 | 14 | 0.01548 | 198 |
| refund_abuse | 130 | 945 | 31.34 | 14 | 0.02243 | 189.2 |
| fake_cashback | 5 | 939.9 | 31.31 | 14 | 0.0008065 | 203 |
| loan_app_extortion | 120 | 891.2 | 31 | 14 | 0.01738 | 172.1 |
| merchant_collusion | 103.9 | 895.7 | 34.9 | 14 | 0.0183 | 192.8 |

## p90 by label
| label | amount | sim_age_hours | device_age_days | login_to_txn_sec | amt_to_balance | ip_risk_score |
|---|---|---|---|---|---|---|
| legit | 2379 | 2.07e+04 | 634.7 | 142 | 0.3644 | 0.369 |
| mule_account | 4640 | 2.074e+04 | 624.6 | 143 | 0.9058 | 0.559 |
| velocity_bot | 2173 | 2.112e+04 | 529.3 | 119 | 0.3592 | 0.607 |
| layering_chain | 1.185e+05 | 2.052e+04 | 629.5 | 137 | 0.9626 | 0.6109 |
| sim_swap | 4579 | 1.839e+04 | 598.7 | 120 | 0.9506 | 0.579 |
| account_takeover | 4377 | 2.014e+04 | 556.5 | 140 | 0.8919 | 0.5952 |
| phishing_collect | 7966 | 2.036e+04 | 631.5 | 138 | 0.9091 | 0.5914 |
| qr_swap | 3054 | 2.111e+04 | 639.1 | 140.7 | 0.4681 | 0.5977 |
| refund_abuse | 4644 | 2.17e+04 | 615.8 | 160 | 0.9091 | 0.6056 |
| fake_cashback | 8597 | 2.035e+04 | 640.4 | 135.9 | 0.9064 | 0.5588 |
| loan_app_extortion | 1.048e+04 | 2.038e+04 | 621.8 | 139 | 0.9091 | 0.551 |
| merchant_collusion | 5000 | 2.027e+04 | 637.8 | 144 | 0.7936 | 0.559 |

## txn_type mix by label (row %)
| label | AUTOPAY | COLLECT | PAY | QR | REFUND |
|---|---|---|---|---|---|
| legit | 3.2 | 3.2 | 57.1 | 35.8 | 0.7 |
| mule_account | 2.1 | 2.4 | 69.9 | 25.2 | 0.4 |
| velocity_bot | 1.8 | 2 | 73.7 | 22.1 | 0.4 |
| layering_chain | 1.9 | 2.1 | 74.5 | 21.1 | 0.5 |
| sim_swap | 1.7 | 2.4 | 73 | 22.5 | 0.4 |
| account_takeover | 1.7 | 2 | 73.4 | 22.4 | 0.5 |
| phishing_collect | 2.2 | 27.2 | 46.6 | 23.5 | 0.5 |
| qr_swap | 1.9 | 2.1 | 40.6 | 55 | 0.4 |
| refund_abuse | 2.1 | 2.1 | 41.1 | 26 | 28.7 |
| fake_cashback | 2 | 13.4 | 60.3 | 24 | 0.4 |
| loan_app_extortion | 2.3 | 5.7 | 65.1 | 26.5 | 0.4 |
| merchant_collusion | 2.4 | 3 | 55.1 | 38.9 | 0.6 |

## receiver_type mix by label (row %)
| label | MERCHANT | P2P |
|---|---|---|
| legit | 64 | 36 |
| mule_account | 44 | 56 |
| velocity_bot | 73.8 | 26.2 |
| layering_chain | 37.5 | 62.5 |
| sim_swap | 39.7 | 60.3 |
| account_takeover | 39.7 | 60.3 |
| phishing_collect | 39.6 | 60.4 |
| qr_swap | 53.2 | 46.8 |
| refund_abuse | 76.6 | 23.4 |
| fake_cashback | 42 | 58 |
| loan_app_extortion | 69.8 | 30.2 |
| merchant_collusion | 69.8 | 30.2 |

## merchant_category mix by label (row %)
| label | NA | ecommerce | education | electronics | food | fuel | gaming | grocery | lending | other | pharmacy | travel | utilities |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| legit | 36 | 14.2 | 1.4 | 2.2 | 7.9 | 6.9 | 2.3 | 12.6 | 0.9 | 2.3 | 4 | 4 | 5.3 |
| mule_account | 56 | 9.8 | 0.8 | 1.6 | 5.2 | 4.7 | 1.7 | 8.8 | 0.6 | 1.5 | 2.6 | 2.9 | 3.9 |
| velocity_bot | 26.1 | 38.7 | 0.9 | 1.7 | 5.1 | 4 | 6.1 | 7.5 | 0.5 | 1.5 | 2.4 | 2.5 | 3 |
| layering_chain | 62.5 | 8.4 | 0.7 | 1.2 | 4.6 | 3.8 | 1.5 | 7.7 | 0.6 | 1.1 | 2.5 | 2 | 3.2 |
| sim_swap | 60.2 | 9.2 | 0.5 | 1.4 | 4.6 | 4.2 | 1.4 | 8.3 | 0.5 | 1.5 | 2.6 | 2.4 | 3.1 |
| account_takeover | 60.1 | 8.7 | 0.9 | 1.4 | 5.4 | 4.6 | 1.2 | 7.8 | 0.5 | 1.3 | 2.7 | 2.3 | 2.9 |
| phishing_collect | 60.3 | 9.4 | 0.7 | 1.2 | 4.5 | 4.6 | 1.6 | 7.5 | 0.4 | 1.4 | 2.4 | 2.6 | 3.4 |
| qr_swap | 22.6 | 9.9 | 0.8 | 2.1 | 15 | 9.5 | 2.2 | 21.3 | 0.9 | 2.4 | 6.1 | 3 | 4.1 |
| refund_abuse | 23.3 | 31.6 | 0.8 | 8.1 | 6 | 4.9 | 1.8 | 9.6 | 1.2 | 1.8 | 3.3 | 2.4 | 5.3 |
| fake_cashback | 57.7 | 9.8 | 1.1 | 1.7 | 5.6 | 4.3 | 1.3 | 7.5 | 0.5 | 1.7 | 3 | 2.4 | 3.3 |
| loan_app_extortion | 30.2 | 10 | 0.8 | 1.6 | 6.6 | 4.5 | 2 | 9.6 | 23.8 | 1.7 | 2.5 | 3.6 | 3.2 |
| merchant_collusion | 28.3 | 16.5 | 1.3 | 1.7 | 8.1 | 9.3 | 3 | 14.1 | 1.7 | 4.3 | 4.8 | 2.3 | 4.7 |

## app mix by label (row %)
| label | APP_A | APP_B | APP_C | APP_D | APP_E |
|---|---|---|---|---|---|
| legit | 33.9 | 29.9 | 20.2 | 10 | 6 |
| mule_account | 34 | 29.6 | 19.9 | 10.2 | 6.3 |
| velocity_bot | 33.8 | 33 | 17.8 | 9.7 | 5.8 |
| layering_chain | 33.5 | 30.9 | 20.1 | 9.9 | 5.6 |
| sim_swap | 34.4 | 29.4 | 18.9 | 10.9 | 6.4 |
| account_takeover | 34.9 | 28.2 | 20.3 | 9.8 | 6.7 |
| phishing_collect | 34.8 | 30.7 | 18.9 | 9.6 | 5.9 |
| qr_swap | 34 | 30.1 | 19.7 | 9.9 | 6.2 |
| refund_abuse | 34.5 | 29.8 | 19.5 | 10.1 | 6.1 |
| fake_cashback | 33.1 | 31.4 | 18.7 | 11.1 | 5.7 |
| loan_app_extortion | 35.3 | 29.2 | 18.9 | 10.6 | 5.9 |
| merchant_collusion | 32.4 | 30.5 | 20.8 | 10.6 | 5.6 |

## app_major mix by label (row %)
| label | 3 | 4 |
|---|---|---|
| legit | 15.1 | 84.9 |
| mule_account | 29.9 | 70.1 |
| velocity_bot | 33.8 | 66.2 |
| layering_chain | 34.5 | 65.5 |
| sim_swap | 33.7 | 66.3 |
| account_takeover | 31.6 | 68.4 |
| phishing_collect | 34 | 66 |
| qr_swap | 32.9 | 67.1 |
| refund_abuse | 33.2 | 66.8 |
| fake_cashback | 30.4 | 69.6 |
| loan_app_extortion | 30.1 | 69.9 |
| merchant_collusion | 29.1 | 70.9 |

## kyc_level mix by label (row %)
| label | full | min |
|---|---|---|
| legit | 82.2 | 17.8 |
| mule_account | 82 | 18 |
| velocity_bot | 80.5 | 19.5 |
| layering_chain | 82.7 | 17.3 |
| sim_swap | 82.3 | 17.7 |
| account_takeover | 82.8 | 17.2 |
| phishing_collect | 81.6 | 18.4 |
| qr_swap | 82.4 | 17.6 |
| refund_abuse | 82 | 18 |
| fake_cashback | 81.1 | 18.9 |
| loan_app_extortion | 81.3 | 18.7 |
| merchant_collusion | 83.2 | 16.8 |

## os mix by label (row %)
| label | android | ios |
|---|---|---|
| legit | 86.1 | 13.9 |
| mule_account | 86.2 | 13.8 |
| velocity_bot | 86.2 | 13.8 |
| layering_chain | 85.4 | 14.6 |
| sim_swap | 86.5 | 13.5 |
| account_takeover | 86.6 | 13.4 |
| phishing_collect | 86.5 | 13.5 |
| qr_swap | 85.8 | 14.2 |
| refund_abuse | 87.4 | 12.6 |
| fake_cashback | 86 | 14 |
| loan_app_extortion | 85.4 | 14.6 |
| merchant_collusion | 85.9 | 14.1 |

## pin_retries mix by label (row %)
| label | 0 | 1 | 2 |
|---|---|---|---|
| legit | 90 | 8 | 2 |
| mule_account | 90 | 8 | 2 |
| velocity_bot | 90.6 | 7.8 | 1.7 |
| layering_chain | 89.1 | 8.7 | 2.3 |
| sim_swap | 84.4 | 11.7 | 4 |
| account_takeover | 76.5 | 16 | 7.4 |
| phishing_collect | 89.9 | 8 | 2.1 |
| qr_swap | 89.9 | 8 | 2 |
| refund_abuse | 90.2 | 8 | 1.8 |
| fake_cashback | 89.8 | 8.6 | 1.5 |
| loan_app_extortion | 90.4 | 7.6 | 2 |
| merchant_collusion | 90.8 | 7.2 | 2 |

## qr_vpa_mismatch value by txn_type (-1 = blank), row %
| txn_type | -1.0 | 0.0 | 1.0 |
|---|---|---|---|
| AUTOPAY | 100 | 0 | 0 |
| COLLECT | 98.37 | 1.62 | 0.01 |
| PAY | 99.3 | 0.7 | 0 |
| QR | 0.12 | 99.33 | 0.54 |
| REFUND | 96.58 | 3.41 | 0.01 |

## qr_vpa_mismatch counts by label (-1 = blank)
| label | -1.0 | 0.0 | 1.0 |
|---|---|---|---|
| legit | 966274 | 538381 | 1757 |
| mule_account | 6534 | 3713 | 14 |
| velocity_bot | 5104 | 2987 | 15 |
| layering_chain | 4375 | 2432 | 15 |
| sim_swap | 2568 | 1497 | 7 |
| account_takeover | 2613 | 1428 | 8 |
| phishing_collect | 4279 | 2401 | 17 |
| qr_swap | 3394 | 2710 | 1160 |
| refund_abuse | 2144 | 1155 | 6 |
| fake_cashback | 1610 | 917 | 5 |
| loan_app_extortion | 1253 | 729 | 9 |
| merchant_collusion | 1630 | 890 | 29 |

## hour share train vs test (all rows)
|  | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train | 1.25 | 0.64 | 0.4 | 0.27 | 0.28 | 0.51 | 1.22 | 2.41 | 4.27 | 5.44 | 6.01 | 6.06 | 6.62 | 6.02 | 5.42 | 5.41 | 6.01 | 6.61 | 7.27 | 7.87 | 7.27 | 6.05 | 4.23 | 2.45 |
| test | 1.22 | 0.67 | 0.39 | 0.27 | 0.28 | 0.53 | 1.25 | 2.38 | 4.28 | 5.44 | 6.08 | 6 | 6.6 | 5.96 | 5.51 | 5.46 | 6.07 | 6.66 | 7.29 | 7.82 | 7.19 | 5.99 | 4.27 | 2.4 |

## Amount patterns by label (% of rows)
| label | amt_is_integer | amt_round_100 | amt_round_500 | amt_round_1000 | amt_just_under_1000s | amt_ends_99 | amt_tiny_le10 |
|---|---|---|---|---|---|---|---|
| legit | 26.02 | 2.44 | 0.32 | 0.1 | 2.24 | 0.71 | 0.16 |
| mule_account | 49.28 | 4.43 | 0.87 | 0.35 | 3.13 | 0.68 | 0.09 |
| velocity_bot | 54.92 | 2.47 | 0.41 | 0.19 | 1.59 | 0.89 | 0.15 |
| layering_chain | 57.11 | 3.05 | 0.43 | 0.18 | 3.33 | 0.81 | 0.13 |
| sim_swap | 53.68 | 13.06 | 11.89 | 0.1 | 2.87 | 0.93 | 0.12 |
| account_takeover | 54.58 | 13.02 | 11.26 | 0.2 | 3.11 | 0.89 | 0.15 |
| phishing_collect | 55.22 | 1.85 | 0.42 | 0.15 | 24.71 | 14.51 | 10.02 |
| qr_swap | 34.77 | 13.67 | 11.66 | 8.88 | 1.98 | 0.61 | 0.23 |
| refund_abuse | 27.53 | 11.77 | 10.65 | 8.59 | 3.06 | 0.76 | 0.15 |
| fake_cashback | 51.54 | 1.9 | 0.51 | 0.2 | 8.25 | 4.46 | 16.59 |
| loan_app_extortion | 48.77 | 2.06 | 0.55 | 0.25 | 6.33 | 2.56 | 0.2 |
| merchant_collusion | 45.82 | 23.3 | 21.62 | 17.54 | 1.96 | 0.75 | 0.27 |
| TEST (all rows) | 26.84 | 2.69 | 0.52 | 0.2 | 2.61 | 0.82 | 0.19 |

## Most common amounts by label (top 12, with count)
- **legit** (n=1506412): 80x7020, 90x7018, 100x6998, 110x6982, 70x6823, 130x6689, 120x6676, 140x6633, 60x6550, 150x6479, 160x6272, 50x6022
- **mule_account** (n=10261): 120x37, 110x36, 50x35, 130x35, 80x33, 60x31, 210x30, 70x30, 170x29, 200x27, 160x27, 360x26
- **velocity_bot** (n=8106): 100x35, 70x32, 40x31, 180x31, 220x29, 50x28, 80x28, 200x28, 110x28, 240x28, 130x27, 60x26
- **layering_chain** (n=6822): 220x24, 110x23, 50x21, 130x21, 60x20, 180x20, 30x19, 210x19, 260x18, 120x18, 100x18, 200x17
- **sim_swap** (n=4072): 500x478, 70x20, 140x20, 130x15, 150x15, 110x14, 90x13, 80x13, 160x11, 60x11, 360x10, 220x10
- **account_takeover** (n=4049): 500x444, 130x17, 200x15, 40x15, 70x15, 100x13, 310x13, 180x13, 50x12, 160x11, 260x11, 90x11
- **phishing_collect** (n=6697): 1999x324, 4999x321, 2x181, 9999x180, 5x174, 1x161, 10x151, 1950x117, 1990x115, 4950x107, 24999x93, 4990x89
- **qr_swap** (n=7264): 2500x187, 2000x171, 1000x168, 5000x159, 10000x147, 50x37, 180x36, 80x33, 100x32, 60x31, 120x30, 130x29
- **refund_abuse** (n=3305): 1000x81, 2000x68, 5000x68, 10000x67, 2500x62, 230x15, 180x14, 130x13, 50x12, 90x12, 170x12, 110x11
- **fake_cashback** (n=2532): 5x114, 10x112, 1x97, 2x96, 4999x39, 1999x30, 9999x23, 1950x15, 190x12, 1990x12, 120x12, 50x10
- **loan_app_extortion** (n=1991): 4999x17, 1999x10, 1950x9, 4990x9, 80x9, 150x8, 430x7, 100x7, 160x7, 70x7, 9999x7, 330x6
- **merchant_collusion** (n=2549): 10000x124, 1000x116, 2000x105, 5000x101, 2500x96, 30x14, 160x12, 50x11, 290x11, 110x11, 150x11, 170x10

## Fraud rate by kyc_level / os (train)
|  | fraud_% |
|---|---|
| full | 3.68 |
| min | 3.73 |
| android | 3.69 |
| ios | 3.67 |

Fraud rate by device_city: min 3.43% (Varanasi), max 4.23% (Jaipur), median 3.71%
Fraud rate by sender bank_code: min 3.48%, max 4.01% (B27)