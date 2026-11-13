## Density: September (train) vs October (test)
| set | rows | uniq_senders | uniq_receivers | uniq_devices | rows_per_sender | rows_per_receiver |
|---|---|---|---|---|---|---|
| train Sep | 5.1e+05 | 2.153e+05 | 1.668e+05 | 2.223e+05 | 2.369 | 3.057 |
| test Oct | 2.605e+05 | 1.482e+05 | 1.004e+05 | 1.507e+05 | 1.758 | 2.594 |

## Relational features: mean by label (train)
|  | legit | mule_account | velocity_bot | layering_chain | sim_swap | account_takeover | phishing_collect | qr_swap | refund_abuse | fake_cashback | loan_app_extortion | merchant_collusion |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rcv_cnt_1h | 17.5 | 13.1 | 19.6 | 11 | 11.3 | 10.7 | 11.9 | 11.2 | 10.7 | 12.4 | 11.5 | 13.3 |
| rcv_cnt_24h | 287 | 205 | 184 | 169 | 183 | 169 | 187 | 175 | 164 | 197 | 183 | 211 |
| rcv_cnt_total | 1.54e+04 | 1.08e+04 | 9.38e+03 | 9.01e+03 | 9.8e+03 | 9.09e+03 | 1e+04 | 9.39e+03 | 8.74e+03 | 1.05e+04 | 9.78e+03 | 1.11e+04 |
| rcv_uniq_snd_day | 144 | 104 | 94.2 | 85 | 92 | 85 | 93.9 | 88.3 | 82.4 | 98.9 | 91.8 | 104 |
| rcv_uniq_snd_total | 1.18e+04 | 8.26e+03 | 7.16e+03 | 6.89e+03 | 7.41e+03 | 6.95e+03 | 7.64e+03 | 7.14e+03 | 6.72e+03 | 8.03e+03 | 7.46e+03 | 8.47e+03 |
| rcv_newpayee_share_day | 0.227 | 0.389 | 0.254 | 0.331 | 0.489 | 0.51 | 0.453 | 0.228 | 0.227 | 0.344 | 0.446 | 0.242 |
| rcv_out_30m | 0.000862 | 0.0359 | 0.00284 | 0.162 | 0.0027 | 0.000741 | 0.00373 | 0.00138 | 0.00182 | 0.0122 | 0.001 | 0.00314 |
| rcv_out_2h | 0.00297 | 0.1 | 0.0111 | 0.182 | 0.00467 | 0.0037 | 0.0203 | 0.00509 | 0.00424 | 0.0825 | 0.00201 | 0.00588 |
| fwd_amt_ratio_30m | 37.9 | 2.02 | 92.6 | 0.986 | 1.03 | 0.575 | 2.15e+03 | 955 | 1.56e+03 | 3.92e+03 | 0.913 | 4.87 |
| snd_in_before_30m | 0.000861 | 0.0363 | 0.00284 | 0.163 | 0.00196 | 0.000988 | 0.00299 | 0.000964 | 0.000908 | 0.0134 | 0.000502 | 0.00314 |
| prev_in_amt_ratio_30m | 43.2 | 1.91 | 4.95 | 1.01 | 1.23 | 0.79 | 2.69e+03 | 2.7 | 0.973 | 3.13e+03 | 0.988 | 0.87 |
| snd_out_cnt_10m | 1 | 1.01 | 2.68 | 1.01 | 1.32 | 1.24 | 1.04 | 1.01 | 1.01 | 1.13 | 1.07 | 1.07 |
| snd_out_cnt_24h | 1.28 | 1.31 | 2.9 | 1.24 | 2.08 | 2.05 | 1.31 | 1.31 | 1.67 | 1.42 | 1.59 | 2.39 |
| snd_uniq_rcv_day | 1.13 | 1.17 | 1.24 | 1.12 | 1.93 | 1.89 | 1.13 | 1.12 | 1.16 | 1.15 | 1.28 | 1.13 |
| snd_in_cnt_24h | 0.0493 | 0.489 | 0.115 | 0.272 | 0.0567 | 0.0571 | 0.0936 | 0.0513 | 0.0484 | 0.346 | 0.0713 | 0.0596 |
| dev_cnt_5m | 1.01 | 1.02 | 13.3 | 1.05 | 1.23 | 1.19 | 1.05 | 1.05 | 1.07 | 1.14 | 1.3 | 1.23 |
| dev_cnt_1h | 1.03 | 1.06 | 15.8 | 1.08 | 1.83 | 1.8 | 1.08 | 1.09 | 1.14 | 1.21 | 1.42 | 1.34 |
| dev_uniq_acct_total | 1.08 | 1.09 | 11.9 | 1.12 | 1.13 | 1.1 | 1.1 | 1.12 | 1.15 | 1.13 | 1.23 | 1.2 |
| dev_uniq_acct_day | 1.01 | 1.02 | 11 | 1.04 | 1.07 | 1.04 | 1.03 | 1.04 | 1.07 | 1.04 | 1.15 | 1.11 |
| acct_uniq_dev_total | 1.48 | 1.47 | 1.81 | 1.46 | 1.59 | 1.72 | 1.5 | 1.51 | 1.6 | 1.55 | 1.5 | 1.56 |
| pair_cnt_total | 1.21 | 1.15 | 2.03 | 1.12 | 1.15 | 1.15 | 1.17 | 1.18 | 2.37 | 1.28 | 1.19 | 2.56 |
| pair_cnt_7d | 1.03 | 1.02 | 1.92 | 1.02 | 1.03 | 1.03 | 1.06 | 1.07 | 2.16 | 1.14 | 1.07 | 2.42 |
| pair_cnt_24h | 1.01 | 1.01 | 1.9 | 1.01 | 1.01 | 1.02 | 1.05 | 1.06 | 1.36 | 1.13 | 1.04 | 2.11 |
| rev_pair_cnt_total | 0.000119 | 0.000292 | 0.000617 | 0.000293 | 0.000491 | 0.000247 | 0.0403 | 0.000688 | 0.000605 | 0.328 | 0.001 | 0.00157 |
| rev_small_before_24h | 5.38e-05 | 9.75e-05 | 0.000493 | 0.000147 | 0.000246 | 0 | 0.0197 | 0.000138 | 0 | 0.142 | 0.000502 | 0.000392 |
| rev_big_after_24h | 4.45e-05 | 0.000195 | 0.000123 | 0.000147 | 0.000246 | 0.000247 | 0.0205 | 0.000551 | 0.000605 | 0.143 | 0.000502 | 0.00118 |
| snd_refund_cnt_30d | 0.032 | 0.0305 | 0.0584 | 0.04 | 0.041 | 0.0393 | 0.0329 | 0.0365 | 1.79 | 0.0288 | 0.0311 | 0.0314 |
| snd_refund_cnt_total | 0.102 | 0.0984 | 0.173 | 0.116 | 0.112 | 0.105 | 0.125 | 0.13 | 1.91 | 0.107 | 0.0914 | 0.138 |
| snd_collect_cnt_7d | 0.0936 | 0.0813 | 0.063 | 0.0734 | 0.0879 | 0.0887 | 0.356 | 0.0833 | 0.0672 | 0.197 | 0.118 | 0.0836 |
| rcv_round500_cnt_7d | 0.0644 | 0.149 | 0.069 | 0.055 | 0.181 | 0.176 | 0.0496 | 0.386 | 0.65 | 0.0553 | 0.0789 | 4.16 |
| pair_round500_cnt_total | 0.00425 | 0.0111 | 0.00839 | 0.00469 | 0.122 | 0.116 | 0.00627 | 0.165 | 0.256 | 0.00711 | 0.0131 | 1.38 |
| pair_share_of_rcv | 0.208 | 0.177 | 0.167 | 0.238 | 0.275 | 0.264 | 0.266 | 0.236 | 0.265 | 0.292 | 0.201 | 0.21 |
| evening_pair_cnt_7d | 0.205 | 0.213 | 0.328 | 0.219 | 0.171 | 0.209 | 0.216 | 0.219 | 0.379 | 0.244 | 0.418 | 0.428 |

## Relational features: median by label (train)
|  | legit | mule_account | velocity_bot | layering_chain | sim_swap | account_takeover | phishing_collect | qr_swap | refund_abuse | fake_cashback | loan_app_extortion | merchant_collusion |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rcv_cnt_1h | 1 | 1 | 6 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| rcv_cnt_24h | 2 | 9 | 17 | 1 | 1 | 1 | 1 | 1 | 2 | 1 | 1 | 8 |
| rcv_cnt_total | 59 | 22 | 31 | 6 | 5 | 5 | 5 | 7 | 14 | 5 | 12 | 31 |
| rcv_uniq_snd_day | 2 | 4 | 6 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 2 |
| rcv_uniq_snd_total | 59 | 22 | 26 | 6 | 5 | 5 | 5 | 7 | 10 | 5 | 11 | 14 |
| rcv_newpayee_share_day | 0.193 | 0.247 | 0.214 | 0.176 | 0.276 | 0.291 | 0.262 | 0 | 0 | 0.232 | 0.261 | 0.2 |
| rcv_out_30m | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| rcv_out_2h | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| fwd_amt_ratio_30m | 0.993 | 0.984 | 2.27 | 0.98 | 0.972 | 0.514 | 6.21 | 1.6 | 2.64 | 2.51e+03 | 0.913 | 0.982 |
| snd_in_before_30m | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| prev_in_amt_ratio_30m | 0.995 | 0.983 | 2.78 | 0.98 | 0.994 | 0.971 | 249 | 0.984 | 0.975 | 1.62e+03 | 0.988 | 0.971 |
| snd_out_cnt_10m | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| snd_out_cnt_24h | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| snd_uniq_rcv_day | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| snd_in_cnt_24h | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| dev_cnt_5m | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| dev_cnt_1h | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| dev_uniq_acct_total | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| dev_uniq_acct_day | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| acct_uniq_dev_total | 1 | 1 | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| pair_cnt_total | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| pair_cnt_7d | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| pair_cnt_24h | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| rev_pair_cnt_total | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| rev_small_before_24h | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| rev_big_after_24h | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| snd_refund_cnt_30d | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| snd_refund_cnt_total | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| snd_collect_cnt_7d | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| rcv_round500_cnt_7d | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pair_round500_cnt_total | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pair_share_of_rcv | 0.0169 | 0.0455 | 0.0417 | 0.167 | 0.2 | 0.2 | 0.2 | 0.15 | 0.176 | 0.2 | 0.0909 | 0.125 |
| evening_pair_cnt_7d | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## % of rows where a relational rule fires, by label (train)
|  | legit | mule_account | velocity_bot | layering_chain | sim_swap | account_takeover | phishing_collect | qr_swap | refund_abuse | fake_cashback | loan_app_extortion | merchant_collusion |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rcv_cnt_1h>=5 | 27.8 | 24.2 | 51.9 | 17.1 | 16.4 | 16.6 | 17.8 | 16.9 | 16.6 | 18.7 | 18.1 | 20.8 |
| rcv_uniq_snd_day>=5 | 40.4 | 49.7 | 51.3 | 28.8 | 24.4 | 24.4 | 25.9 | 24.7 | 25.8 | 26.8 | 28 | 31.3 |
| rcv_out_30m>0 | 0.09 | 3.44 | 0.26 | 16.1 | 0.27 | 0.07 | 0.37 | 0.14 | 0.18 | 1.22 | 0.1 | 0.31 |
| fwd_ratio_0.8-1.0 | 0.02 | 2.66 | 0.02 | 16.1 | 0.15 | 0.02 | 0.04 | 0.04 | 0.09 | 0.16 | 0 | 0.24 |
| snd_in_before_30m>0 | 0.09 | 3.43 | 0.26 | 16.3 | 0.2 | 0.1 | 0.3 | 0.1 | 0.09 | 0.99 | 0.05 | 0.31 |
| prev_in_ratio_0.8-1.0 | 0.01 | 2.73 | 0.02 | 16.2 | 0.1 | 0.07 | 0.04 | 0.04 | 0.09 | 0.16 | 0.05 | 0.24 |
| dev_cnt_5m>=5 | 0.03 | 0.05 | 34.9 | 0.15 | 0.17 | 0.22 | 0.06 | 0.15 | 0.18 | 0.2 | 0.75 | 0.59 |
| dev_uniq_acct_total>=3 | 0.24 | 0.25 | 35.1 | 0.41 | 0.39 | 0.44 | 0.22 | 0.37 | 0.51 | 0.47 | 1.1 | 0.67 |
| pair_cnt_7d>=3 | 0.24 | 0.2 | 11.7 | 0.28 | 0.34 | 0.42 | 0.3 | 0.95 | 28.6 | 0.2 | 0.9 | 23.8 |
| rev_pair>0 | 0.01 | 0.03 | 0.05 | 0.03 | 0.05 | 0.02 | 4.03 | 0.07 | 0.06 | 27.1 | 0.1 | 0.16 |
| rev_small_before_24h>0 | 0 | 0.01 | 0.04 | 0.01 | 0.02 | 0 | 1.97 | 0.01 | 0 | 9.28 | 0.05 | 0.04 |
| rev_big_after_24h>0 | 0 | 0.02 | 0.01 | 0.01 | 0.02 | 0.02 | 2.05 | 0.06 | 0.06 | 14.3 | 0.05 | 0.12 |
| snd_refund_cnt_30d>=3 | 0.03 | 0.07 | 0.81 | 0.22 | 0.12 | 0.1 | 0.1 | 0.18 | 28.4 | 0.08 | 0.1 | 0.08 |
| pair_round500>=2 | 0.01 | 0.05 | 0.07 | 0.01 | 0.05 | 0.07 | 0.06 | 3.92 | 7.81 | 0.04 | 0.15 | 20.8 |
| evening_pair_cnt_7d>=3 | 0.01 | 0.01 | 1.6 | 0.04 | 0.02 | 0.02 | 0.01 | 0.08 | 1.54 | 0 | 0.05 | 1.96 |

## Rule firing rate on all rows: Sep vs Oct (%)
|  | Sep legit | Sep all | Oct all (test) |
|---|---|---|---|
| rcv_cnt_1h>=5 | 27.7 | 27.55 | 23.42 |
| rcv_uniq_snd_day>=5 | 40.3 | 40.09 | 36.56 |
| rcv_out_30m>0 | 0.086 | 0.182 | 0.137 |
| fwd_ratio_0.8-1.0 | 0.013 | 0.102 | 0.092 |
| snd_in_before_30m>0 | 0.087 | 0.181 | 0.138 |
| prev_in_ratio_0.8-1.0 | 0.012 | 0.102 | 0.092 |
| dev_cnt_5m>=5 | 0.026 | 0.209 | 0.103 |
| dev_uniq_acct_total>=3 | 0.243 | 0.426 | 0.307 |
| pair_cnt_7d>=3 | 0.245 | 0.414 | 0.21 |
| rev_pair>0 | 0.012 | 0.079 | 0.077 |
| rev_small_before_24h>0 | 0.004 | 0.03 | 0.028 |
| rev_big_after_24h>0 | 0.005 | 0.039 | 0.038 |
| snd_refund_cnt_30d>=3 | 0.032 | 0.106 | 0.082 |
| pair_round500>=2 | 0.012 | 0.085 | 0.078 |
| evening_pair_cnt_7d>=3 | 0.008 | 0.035 | 0.008 |

## One-vs-legit AUC of each relational feature (train). 0.5 = useless, far from 0.5 = strong
|  | mule_account | velocity_bot | layering_chain | sim_swap | account_takeover | phishing_collect | qr_swap | refund_abuse | fake_cashback | loan_app_extortion | merchant_collusion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rcv_cnt_1h | 0.511 | 0.619 | 0.453 | 0.424 | 0.425 | 0.445 | 0.438 | 0.447 | 0.475 | 0.441 | 0.502 |
| rcv_cnt_24h | 0.519 | 0.56 | 0.437 | 0.4 | 0.399 | 0.416 | 0.427 | 0.475 | 0.443 | 0.429 | 0.52 |
| rcv_cnt_total | 0.472 | 0.488 | 0.412 | 0.396 | 0.399 | 0.405 | 0.421 | 0.465 | 0.413 | 0.453 | 0.482 |
| rcv_uniq_snd_day | 0.522 | 0.557 | 0.443 | 0.405 | 0.404 | 0.412 | 0.424 | 0.438 | 0.421 | 0.429 | 0.503 |
| rcv_uniq_snd_total | 0.473 | 0.48 | 0.417 | 0.402 | 0.405 | 0.409 | 0.424 | 0.449 | 0.41 | 0.455 | 0.471 |
| rcv_newpayee_share_day | 0.592 | 0.551 | 0.532 | 0.647 | 0.661 | 0.629 | 0.47 | 0.487 | 0.56 | 0.629 | 0.525 |
| rcv_out_30m | 0.517 | 0.501 | 0.58 | 0.501 | 0.5 | 0.501 | 0.5 | 0.5 | 0.506 | 0.5 | 0.501 |
| rcv_out_2h | 0.534 | 0.502 | 0.589 | 0.501 | 0.5 | 0.508 | 0.501 | 0.5 | 0.539 | 0.5 | 0.501 |
| fwd_amt_ratio_30m | 0.517 | 0.501 | 0.58 | 0.501 | 0.5 | 0.501 | 0.5 | 0.5 | 0.506 | 0.5 | 0.501 |
| snd_in_before_30m | 0.517 | 0.501 | 0.581 | 0.501 | 0.5 | 0.501 | 0.5 | 0.5 | 0.505 | 0.5 | 0.501 |
| prev_in_amt_ratio_30m | 0.517 | 0.501 | 0.581 | 0.501 | 0.5 | 0.501 | 0.5 | 0.5 | 0.505 | 0.5 | 0.501 |
| snd_out_cnt_10m | 0.504 | 0.561 | 0.501 | 0.613 | 0.59 | 0.52 | 0.501 | 0.504 | 0.558 | 0.506 | 0.508 |
| snd_out_cnt_24h | 0.509 | 0.547 | 0.485 | 0.671 | 0.667 | 0.515 | 0.513 | 0.607 | 0.549 | 0.572 | 0.617 |
| snd_uniq_rcv_day | 0.509 | 0.532 | 0.493 | 0.679 | 0.675 | 0.5 | 0.496 | 0.511 | 0.503 | 0.55 | 0.498 |
| snd_in_cnt_24h | 0.53 | 0.504 | 0.605 | 0.502 | 0.501 | 0.52 | 0.501 | 0.5 | 0.616 | 0.505 | 0.505 |
| dev_cnt_5m | 0.502 | 0.675 | 0.501 | 0.569 | 0.551 | 0.514 | 0.501 | 0.502 | 0.542 | 0.505 | 0.507 |
| dev_cnt_1h | 0.51 | 0.676 | 0.5 | 0.679 | 0.678 | 0.518 | 0.506 | 0.52 | 0.556 | 0.53 | 0.533 |
| dev_uniq_acct_total | 0.498 | 0.675 | 0.501 | 0.493 | 0.496 | 0.498 | 0.502 | 0.5 | 0.499 | 0.507 | 0.496 |
| dev_uniq_acct_day | 0.5 | 0.674 | 0.501 | 0.501 | 0.5 | 0.501 | 0.501 | 0.501 | 0.501 | 0.503 | 0.504 |
| acct_uniq_dev_total | 0.498 | 0.614 | 0.497 | 0.537 | 0.578 | 0.508 | 0.509 | 0.545 | 0.514 | 0.507 | 0.525 |
| pair_cnt_total | 0.48 | 0.543 | 0.474 | 0.479 | 0.479 | 0.495 | 0.496 | 0.669 | 0.538 | 0.48 | 0.615 |
| pair_cnt_7d | 0.497 | 0.558 | 0.495 | 0.496 | 0.497 | 0.515 | 0.517 | 0.676 | 0.555 | 0.501 | 0.622 |
| pair_cnt_24h | 0.5 | 0.559 | 0.5 | 0.5 | 0.5 | 0.519 | 0.52 | 0.613 | 0.557 | 0.502 | 0.617 |
| rev_pair_cnt_total | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.52 | 0.5 | 0.5 | 0.635 | 0.5 | 0.501 |
| rev_small_before_24h | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.51 | 0.5 | 0.5 | 0.546 | 0.5 | 0.5 |
| rev_big_after_24h | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.51 | 0.5 | 0.5 | 0.572 | 0.5 | 0.501 |
| snd_refund_cnt_30d | 0.498 | 0.499 | 0.499 | 0.501 | 0.502 | 0.498 | 0.499 | 0.64 | 0.498 | 0.498 | 0.498 |
| snd_refund_cnt_total | 0.498 | 0.498 | 0.497 | 0.5 | 0.5 | 0.499 | 0.501 | 0.643 | 0.498 | 0.494 | 0.496 |
| snd_collect_cnt_7d | 0.494 | 0.486 | 0.49 | 0.497 | 0.497 | 0.613 | 0.496 | 0.489 | 0.545 | 0.509 | 0.495 |
| rcv_round500_cnt_7d | 0.53 | 0.5 | 0.498 | 0.553 | 0.55 | 0.494 | 0.551 | 0.546 | 0.496 | 0.494 | 0.606 |
| pair_round500_cnt_total | 0.503 | 0.5 | 0.5 | 0.558 | 0.555 | 0.501 | 0.557 | 0.552 | 0.501 | 0.501 | 0.607 |
| pair_share_of_rcv | 0.527 | 0.532 | 0.581 | 0.596 | 0.593 | 0.59 | 0.576 | 0.594 | 0.595 | 0.546 | 0.548 |
| evening_pair_cnt_7d | 0.504 | 0.494 | 0.506 | 0.481 | 0.5 | 0.502 | 0.505 | 0.555 | 0.507 | 0.603 | 0.561 |

## Label consistency / noise probes (train)
- P2P rows paid to a receiver within ±1h of a mule_account inbound row: n=6,048 -> {'mule_account': 94.99, 'legit': 3.92, 'layering_chain': 0.25, 'fake_cashback': 0.15, 'velocity_bot': 0.12}
- rows from the same device within ±5 min of a velocity_bot row: n=8,627 -> {'velocity_bot': 93.96, 'legit': 4.98, 'merchant_collusion': 0.17, 'loan_app_extortion': 0.17, 'layering_chain': 0.13}
- REFUND rows by senders who have any refund_abuse row: n=1,303 -> {'refund_abuse': 72.91, 'legit': 24.64, 'velocity_bot': 0.46, 'layering_chain': 0.46, 'mule_account': 0.31}
- ALL rows by senders who have any refund_abuse row: n=30,806 -> {'legit': 84.99, 'refund_abuse': 10.73, 'velocity_bot': 1.09, 'phishing_collect': 0.59, 'mule_account': 0.58}
- rows with qr_vpa_mismatch == 1: n=3,042 -> {'legit': 57.76, 'qr_swap': 38.13, 'merchant_collusion': 0.95, 'phishing_collect': 0.56, 'layering_chain': 0.49}
- receivers with >=2 fraud inbound rows having >1 distinct fraud label: n=5,973 -> {'share_%': np.float64(75.41)}
- senders with >=2 fraud rows having >1 distinct fraud label: n=8,245 -> {'share_%': np.float64(71.76)}

## Most common multi-label combinations on one sender (fraud rows only)
- ('layering_chain', 'mule_account'): 270
- ('qr_swap', 'velocity_bot'): 268
- ('mule_account', 'velocity_bot'): 256
- ('mule_account', 'qr_swap'): 252
- ('layering_chain', 'velocity_bot'): 251
- ('mule_account', 'phishing_collect'): 234
- ('phishing_collect', 'velocity_bot'): 210
- ('layering_chain', 'qr_swap'): 181
- ('phishing_collect', 'qr_swap'): 178
- ('layering_chain', 'phishing_collect'): 172
- ('refund_abuse', 'velocity_bot'): 127
- ('account_takeover', 'mule_account'): 121
- ('mule_account', 'sim_swap'): 113
- ('qr_swap', 'sim_swap'): 105
- ('fake_cashback', 'velocity_bot'): 99

## Most common multi-label combinations on one receiver (fraud rows only)
- ('phishing_collect', 'qr_swap'): 364
- ('layering_chain', 'phishing_collect'): 218
- ('account_takeover', 'sim_swap'): 202
- ('layering_chain', 'qr_swap'): 188
- ('account_takeover', 'layering_chain'): 186
- ('layering_chain', 'sim_swap'): 182
- ('mule_account', 'sim_swap'): 128
- ('layering_chain', 'mule_account'): 119
- ('account_takeover', 'mule_account'): 108
- ('mule_account', 'qr_swap'): 106
- ('mule_account', 'phishing_collect'): 88
- ('fake_cashback', 'phishing_collect'): 85
- ('qr_swap', 'velocity_bot'): 75
- ('qr_swap', 'refund_abuse'): 68
- ('mule_account', 'velocity_bot'): 68

## Do train fraud entities reappear in October?
- test rows whose receiver had a fraud inbound row in train: 55.91% (P2P only: 7.37%)
- test rows whose sender had a fraud row in train: 25.66%
- test rows whose device had a fraud row in train: 24.60%
- for reference, Sep rows whose receiver had a fraud inbound row in Jul-Aug: 53.06%
- Sep: label distribution of rows whose receiver_id had fraud in Jul-Aug (n=270,620): {'legit': 96.6, 'mule_account': 0.6, 'qr_swap': 0.5, 'layering_chain': 0.4}
- Sep: label distribution of rows whose sender_id had fraud in Jul-Aug (n=98,526): {'legit': 95.3, 'velocity_bot': 0.8, 'mule_account': 0.7, 'phishing_collect': 0.6}
- Sep: label distribution of rows whose device_id had fraud in Jul-Aug (n=93,475): {'legit': 96.1, 'mule_account': 0.7, 'phishing_collect': 0.6, 'qr_swap': 0.5}