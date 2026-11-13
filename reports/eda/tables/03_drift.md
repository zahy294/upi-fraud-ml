## Sep (train) vs Oct (test) marginal drift, all rows
PSI < 0.1 stable, 0.1-0.25 moderate, > 0.25 major.
| feature | sep_mean | oct_mean | sep_median | oct_median | sep_p90 | oct_p90 | KS | PSI |
|---|---|---|---|---|---|---|---|---|
| ip_risk_score | 0.2034 | 0.2204 | 0.181 | 0.211 | 0.376 | 0.346 | 0.1587 | 0.2763 |
| mer_age_days | 1001 | 1028 | 693 | 719 | 2119 | 2147 | 0.1327 | 0.09587 |
| rcv_acct_age_days | 924.4 | 953.3 | 623 | 649 | 2046 | 2068 | 0.03908 | 0.01706 |
| sender_account_age_days | 921 | 950.7 | 620 | 651 | 2031 | 2058 | 0.0368 | 0.01526 |
| amount | 1235 | 1434 | 463.7 | 541.2 | 2447 | 2878 | 0.04943 | 0.01406 |
| amt_to_balance | 0.1443 | 0.1599 | 0.07716 | 0.08972 | 0.3719 | 0.3986 | 0.04687 | 0.01268 |
| device_age_days | 299.6 | 307.4 | 230.7 | 241.3 | 646 | 658.3 | 0.02003 | 0.002331 |
| qr_vpa_mismatch | 0.005464 | 0.004053 | 0 | 0 | 0 | 0 | 0.001411 | 0.0004234 |
| balance_before | 1.128e+04 | 1.146e+04 | 6290 | 6399 | 2.488e+04 | 2.532e+04 | 0.007189 | 0.0003828 |
| amt_just_under_1000s | 0.02358 | 0.02608 | 0 | 0 | 0 | 0 | 0.002496 | 0.0002575 |
| amt_tiny_le10 | 0.002269 | 0.001865 | 0 | 0 | 0 | 0 | 0.0004036 | 7.922e-05 |
| login_to_txn_sec | 66.93 | 66.9 | 45 | 45 | 142 | 142 | 0.001638 | 6.143e-05 |
| sim_age_hours | 8964 | 8917 | 6217 | 6183 | 2.068e+04 | 2.059e+04 | 0.00238 | 3.591e-05 |
| amt_round_100 | 0.02618 | 0.02695 | 0 | 0 | 0 | 0 | 0.0007668 | 2.274e-05 |
| amt_round_500 | 0.00492 | 0.005219 | 0 | 0 | 0 | 0 | 0.0002994 | 1.778e-05 |
| amt_round_1000 | 0.001906 | 0.002022 | 0 | 0 | 0 | 0 | 0.0001165 | 6.925e-06 |
| new_payee | 0.2312 | 0.2302 | 0 | 0 | 1 | 1 | 0.0009998 | 5.632e-06 |
| amt_is_integer | 0.2689 | 0.2684 | 0 | 0 | 1 | 1 | 0.0005557 | 1.572e-06 |
| home_device | 0.9185 | 0.9188 | 1 | 1 | 1 | 1 | 0.0002393 | 7.664e-07 |
| pin_retries | 0.1206 | 0.1205 | 0 | 0 | 1 | 1 | 0.0001648 | 3.004e-07 |
| is_night | 0.03344 | 0.0335 | 0 | 0 | 0 | 0 | 5.469e-05 | 9.247e-08 |
| away_from_home | 0.06877 | 0.06875 | 0 | 0 | 0 | 0 | 2.697e-05 | 1.136e-08 |

## Legit-only September vs all of October
| feature | legitSep_mean | oct_mean | PSI_legitSep_vs_oct |
|---|---|---|---|
| ip_risk_score | 0.2005 | 0.2204 | 0.2721 |
| mer_age_days | 999.2 | 1028 | 0.09947 |
| amount | 1069 | 1434 | 0.017 |
| rcv_acct_age_days | 923.5 | 953.3 | 0.01676 |
| amt_to_balance | 0.137 | 0.1599 | 0.01565 |
| sender_account_age_days | 920.7 | 950.7 | 0.01547 |
| device_age_days | 300.3 | 307.4 | 0.002319 |
| amt_round_500 | 0.00322 | 0.005219 | 0.0009697 |
| amt_round_1000 | 0.000979 | 0.002022 | 0.0007581 |
| amt_just_under_1000s | 0.02239 | 0.02608 | 0.0005764 |
| balance_before | 1.114e+04 | 1.146e+04 | 0.0004704 |
| amt_is_integer | 0.2606 | 0.2684 | 0.0003109 |
| amt_round_100 | 0.02453 | 0.02695 | 0.0002338 |
| qr_vpa_mismatch | 0.003373 | 0.004053 | 0.0001252 |
| is_night | 0.03178 | 0.0335 | 9.298e-05 |
| new_payee | 0.2269 | 0.2302 | 6.345e-05 |
| home_device | 0.9209 | 0.9188 | 6.01e-05 |
| sim_age_hours | 8972 | 8917 | 5.657e-05 |
| login_to_txn_sec | 67.05 | 66.9 | 4.674e-05 |
| amt_tiny_le10 | 0.001598 | 0.001865 | 4.144e-05 |
| away_from_home | 0.06813 | 0.06875 | 5.862e-06 |
| pin_retries | 0.1198 | 0.1205 | 3.082e-06 |

## txn_type: Sep vs Oct share (top |diff|)
| txn_type | sep_% | oct_% | diff_pp |
|---|---|---|---|
| PAY | 57.44 | 57.08 | -0.36 |
| QR | 35.38 | 35.65 | 0.268 |
| AUTOPAY | 3.158 | 3.229 | 0.071 |
| COLLECT | 3.301 | 3.333 | 0.032 |
| REFUND | 0.72 | 0.709 | -0.011 |

## receiver_type: Sep vs Oct share (top |diff|)
| receiver_type | sep_% | oct_% | diff_pp |
|---|---|---|---|
| MERCHANT | 63.49 | 63.53 | 0.039 |
| P2P | 36.51 | 36.47 | -0.039 |

## merchant_category: Sep vs Oct share (top |diff|)
| merchant_category | sep_% | oct_% | diff_pp |
|---|---|---|---|
| gaming | 2.298 | 2.191 | -0.107 |
| fuel | 6.892 | 6.984 | 0.092 |
| utilities | 5.211 | 5.289 | 0.078 |
| electronics | 2.209 | 2.151 | -0.058 |
| pharmacy | 3.976 | 4.023 | 0.047 |
| lending | 0.872 | 0.832 | -0.04 |
| grocery | 12.55 | 12.51 | -0.039 |
| education | 1.336 | 1.367 | 0.032 |
| travel | 3.971 | 3.943 | -0.028 |
| NA | 36.38 | 36.35 | -0.028 |
| ecommerce | 14.18 | 14.21 | 0.026 |
| other | 2.238 | 2.262 | 0.024 |

## app: Sep vs Oct share (top |diff|)
| app | sep_% | oct_% | diff_pp |
|---|---|---|---|
| APP_B | 29.96 | 30.08 | 0.119 |
| APP_A | 33.77 | 33.87 | 0.1 |
| APP_D | 10.06 | 9.976 | -0.084 |
| APP_E | 5.96 | 5.884 | -0.076 |
| APP_C | 20.25 | 20.19 | -0.059 |

## device_city: Sep vs Oct share (top |diff|)
| device_city | sep_% | oct_% | diff_pp |
|---|---|---|---|
| Agra | 10.98 | 11.12 | 0.14 |
| Bhopal | 5.584 | 5.696 | 0.112 |
| Nagpur | 3.79 | 3.691 | -0.1 |
| Bengaluru | 7.417 | 7.499 | 0.082 |
| Jodhpur | 2.594 | 2.518 | -0.076 |
| Giridih | 3.32 | 3.395 | 0.076 |
| Bharatpur | 2.888 | 2.836 | -0.052 |
| Ahmedabad | 1.052 | 1.101 | 0.049 |
| Visakhapatnam | 0.715 | 0.756 | 0.041 |
| Patna | 1.879 | 1.839 | -0.041 |
| Vadodara | 0.983 | 1.02 | 0.038 |
| Indore | 1.057 | 1.02 | -0.037 |

## Median amount by merchant category, Sep vs Oct
| merchant_category | sep_median | oct_median | ratio |
|---|---|---|---|
| P2P | 380 | 440 | 1.16 |
| ecommerce | 654.1 | 777.5 | 1.19 |
| education | 1334 | 1541 | 1.16 |
| electronics | 1837 | 2090 | 1.14 |
| food | 244.9 | 288.5 | 1.18 |
| fuel | 601.9 | 693.2 | 1.15 |
| gaming | 202.4 | 231.5 | 1.14 |
| grocery | 366.6 | 436.6 | 1.19 |
| lending | 1051 | 1158 | 1.1 |
| other | 444.8 | 536.2 | 1.21 |
| pharmacy | 410 | 472.1 | 1.15 |
| travel | 1472 | 1738 | 1.18 |
| utilities | 735.1 | 851.2 | 1.16 |

## Univariate AUC fraud-vs-legit, per month (0.5 = no signal; <0.5 = inverse)
| feature | 2026-07 | 2026-08 | 2026-09 | max_abs_dev |
|---|---|---|---|---|
| amt_is_integer | 0.6205 | 0.6136 | 0.6139 | 0.1205 |
| ip_risk_score | 0.6009 | 0.6003 | 0.606 | 0.106 |
| amt_to_balance | 0.5872 | 0.5908 | 0.5855 | 0.0908 |
| app_major_is3 | 0.5875 | 0.5853 | 0.5865 | 0.0875 |
| amount | 0.5776 | 0.5865 | 0.58 | 0.0865 |
| new_payee | 0.5665 | 0.5609 | 0.5594 | 0.0665 |
| rcv_acct_age_days | 0.5662 | 0.5628 | 0.5609 | 0.0662 |
| mer_age_days | 0.4352 | 0.4348 | 0.4457 | 0.0652 |
| home_device | 0.4627 | 0.4706 | 0.468 | 0.0373 |
| device_age_days | 0.4638 | 0.468 | 0.4694 | 0.0362 |
| login_to_txn_sec | 0.4676 | 0.4723 | 0.472 | 0.0324 |
| amt_round_100 | 0.5219 | 0.5253 | 0.5226 | 0.0253 |
| amt_round_500 | 0.5218 | 0.525 | 0.5232 | 0.025 |
| is_night | 0.5244 | 0.5228 | 0.5227 | 0.0244 |
| balance_before | 0.5129 | 0.5208 | 0.5189 | 0.0208 |
| amt_just_under_1000s | 0.5171 | 0.5165 | 0.5163 | 0.0171 |
| qr_vpa_mismatch | 0.5164 | 0.5165 | 0.5169 | 0.0169 |
| sim_age_hours | 0.4921 | 0.4859 | 0.4886 | 0.0141 |
| amt_round_1000 | 0.5101 | 0.5135 | 0.5127 | 0.0135 |
| away_from_home | 0.5133 | 0.51 | 0.5087 | 0.0133 |
| amt_tiny_le10 | 0.51 | 0.5088 | 0.5092 | 0.01 |
| pin_retries | 0.5065 | 0.5066 | 0.5077 | 0.0077 |
| sender_account_age_days | 0.4986 | 0.4987 | 0.5032 | 0.0032 |

## ip_risk_score by month and fraud/legit (train)
|  | mean | 50% | 75% | max |
|---|---|---|---|---|
| ('2026-07', 'fraud') | 0.2795 | 0.226 | 0.389 | 0.97 |
| ('2026-07', 'legit') | 0.2005 | 0.18 | 0.273 | 0.946 |
| ('2026-08', 'fraud') | 0.2785 | 0.225 | 0.386 | 0.946 |
| ('2026-08', 'legit') | 0.2002 | 0.18 | 0.273 | 0.903 |
| ('2026-09', 'fraud') | 0.2795 | 0.229 | 0.385 | 0.955 |
| ('2026-09', 'legit') | 0.2005 | 0.18 | 0.273 | 0.923 |
| ('2026-10', 'fraud') | 0.404 | 0.393 | 0.594 | 0.895 |
| ('2026-10', 'legit') | 0.4392 | 0.503 | 0.6028 | 0.703 |

ip_risk_score test: mean 0.2204, median 0.2110, p75 0.2790, p99 0.4710

## ip_risk_score by label (train)
| label | mean | median | pct_gt_0.5 |
|---|---|---|---|
| legit | 0.2004 | 0.18 | 2.01 |
| mule_account | 0.2651 | 0.215 | 13.16 |
| velocity_bot | 0.2873 | 0.233 | 16.95 |
| layering_chain | 0.2925 | 0.238 | 17.93 |
| sim_swap | 0.2768 | 0.227 | 15.45 |
| account_takeover | 0.2846 | 0.232 | 16.79 |
| phishing_collect | 0.2827 | 0.232 | 16.22 |
| qr_swap | 0.283 | 0.2295 | 16.49 |
| refund_abuse | 0.2867 | 0.23 | 17.03 |
| fake_cashback | 0.2721 | 0.2275 | 13.59 |
| loan_app_extortion | 0.2642 | 0.216 | 13.16 |
| merchant_collusion | 0.2643 | 0.214 | 13.5 |

## ip_risk_score AUC (label vs legit) per month
|  | 2026-07 | 2026-08 | 2026-09 |
|---|---|---|---|
| mule_account | 0.583 | 0.581 | 0.588 |
| velocity_bot | 0.612 | 0.609 | 0.62 |
| layering_chain | 0.622 | 0.617 | 0.618 |
| sim_swap | 0.597 | 0.597 | 0.604 |
| account_takeover | 0.615 | 0.605 | 0.609 |
| phishing_collect | 0.609 | 0.609 | 0.607 |
| qr_swap | 0.601 | 0.607 | 0.609 |
| refund_abuse | 0.606 | 0.606 | 0.615 |
| fake_cashback | 0.58 | 0.593 | 0.616 |
| loan_app_extortion | 0.58 | 0.582 | 0.584 |
| merchant_collusion | 0.567 | 0.58 | 0.588 |

## Adversarial validation (Sep vs Oct, row features excl. app_version & date)
AUC = 0.6685
|  | gain_% |
|---|---|
| ip_risk_score | 48.06 |
| sender_account_age_days | 22.76 |
| weekday | 6.57 |
| device_city | 4.7 |
| amt_to_balance | 3.5 |
| amount | 3.37 |
| sim_age_hours | 2.07 |
| login_to_txn_sec | 1.96 |
| device_age_days | 1.93 |
| balance_before | 1.57 |
| merchant_category | 1.18 |
| hour | 1.11 |
| qr_vpa_mismatch | 0.25 |
| app | 0.23 |
| new_payee | 0.11 |
- without ['ip_risk_score']: AUC = 0.5917
- without ['ip_risk_score', 'sender_account_age_days', 'weekday']: AUC = 0.5352