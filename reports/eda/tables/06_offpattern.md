## Off-pattern fraud rows: linked to an on-pattern row of the same label? (train)
on = rows matching that type's signature rule, off = the rest.
| label | n_on | n_off | off linked via sender ±24h % | off linked via receiver ±24h % | off linked via device ±1h % | off linked any % | recoverable ceiling % (on + linked off) |
|---|---|---|---|---|---|---|---|
| mule_account | 2.46e+03 | 7.8e+03 | 0 | 0.9 | 0 | 0.9 | 24.7 |
| velocity_bot | 2.88e+03 | 5.23e+03 | 0 | 1.3 | 0 | 1.3 | 36.3 |
| layering_chain | 1.43e+03 | 5.39e+03 | 0 | 0 | 0 | 0.1 | 21 |
| sim_swap | 1.05e+03 | 3.02e+03 | 0.6 | 0.9 | 0.5 | 1.5 | 26.8 |
| account_takeover | 989 | 3.06e+03 | 0.9 | 0.7 | 0.8 | 1.6 | 25.7 |
| phishing_collect | 1.48e+03 | 5.22e+03 | 0 | 0 | 0 | 0 | 22.1 |
| qr_swap | 1.84e+03 | 5.43e+03 | 0 | 0 | 0 | 0 | 25.3 |
| refund_abuse | 935 | 2.37e+03 | 0 | 0 | 0 | 0 | 28.3 |
| fake_cashback | 598 | 1.93e+03 | 0 | 0 | 0 | 0 | 23.6 |
| loan_app_extortion | 480 | 1.51e+03 | 0 | 0.1 | 0 | 0.1 | 24.2 |
| merchant_collusion | 530 | 2.02e+03 | 0 | 0.4 | 0 | 0.4 | 21.1 |

## Profile of on- vs off-pattern rows (train)
|  | n | app3.x_% | ip>0.5_% | ip_mean | amt_integer_% | new_payee_% | P2P_% | amount_median |
|---|---|---|---|---|---|---|---|---|
| ('mule_account', 'on') | 2457 | 60.85 | 37.85 | 0.41 | 100 | 89.54 | 100 | 2600 |
| ('mule_account', 'off') | 7804 | 20.16 | 5.38 | 0.22 | 33.32 | 23.74 | 42.13 | 559.3 |
| ('velocity_bot', 'on') | 2876 | 61.61 | 39.46 | 0.42 | 99.06 | 21.91 | 0.87 | 252 |
| ('velocity_bot', 'off') | 5230 | 18.55 | 4.57 | 0.22 | 30.65 | 27.25 | 40.13 | 530 |
| ('layering_chain', 'on') | 1430 | 63.22 | 41.19 | 0.43 | 100 | 21.89 | 99.93 | 1.053e+05 |
| ('layering_chain', 'off') | 5392 | 26.89 | 11.76 | 0.26 | 45.73 | 35.4 | 52.63 | 760 |
| ('sim_swap', 'on') | 1047 | 59.69 | 35.82 | 0.39 | 96.37 | 92.07 | 96.56 | 1108 |
| ('sim_swap', 'off') | 3025 | 24.66 | 8.4 | 0.24 | 38.91 | 35.17 | 47.8 | 540 |
| ('account_takeover', 'on') | 989 | 57.03 | 38.52 | 0.41 | 95.85 | 96.26 | 96.56 | 1134 |
| ('account_takeover', 'off') | 3060 | 23.4 | 9.77 | 0.24 | 41.24 | 37.58 | 48.56 | 540 |
| ('phishing_collect', 'on') | 1478 | 61.43 | 37.69 | 0.41 | 99.86 | 99.86 | 100 | 4999 |
| ('phishing_collect', 'off') | 5219 | 26.21 | 10.14 | 0.25 | 42.58 | 29.51 | 49.2 | 419.2 |
| ('qr_swap', 'on') | 1836 | 58.88 | 38.78 | 0.41 | 25.82 | 23.2 | 95.86 | 452.8 |
| ('qr_swap', 'off') | 5428 | 24.12 | 8.95 | 0.24 | 37.8 | 22.42 | 30.2 | 600 |
| ('refund_abuse', 'on') | 935 | 63.74 | 40.43 | 0.42 | 1.39 | 22.99 | 0 | 2302 |
| ('refund_abuse', 'off') | 2370 | 21.14 | 7.81 | 0.24 | 37.85 | 22.87 | 32.62 | 590.1 |
| ('fake_cashback', 'on') | 598 | 59.2 | 36.79 | 0.41 | 100 | 48.49 | 100 | 10 |
| ('fake_cashback', 'off') | 1934 | 21.51 | 6.41 | 0.23 | 36.56 | 29.73 | 44.98 | 523.4 |
| ('loan_app_extortion', 'on') | 480 | 61.04 | 35.62 | 0.39 | 97.5 | 97.71 | 1.04 | 7653 |
| ('loan_app_extortion', 'off') | 1511 | 20.25 | 6.02 | 0.22 | 33.29 | 28.86 | 39.51 | 523.3 |
| ('merchant_collusion', 'on') | 530 | 64.34 | 41.13 | 0.42 | 100 | 21.13 | 0 | 2500 |
| ('merchant_collusion', 'off') | 2019 | 19.86 | 6.24 | 0.22 | 31.6 | 25.31 | 38.09 | 460 |
| ('legit', '-') | 1.506e+06 | 15.06 | 2.01 | 0.2 | 26.02 | 22.69 | 35.98 | 460 |

## Univariate AUC: off-pattern fraud (all types pooled) vs legit
|  | AUC |
|---|---|
| amt_is_integer | 0.557 |
| amt_to_balance | 0.546 |
| ip_risk_score | 0.545 |
| amount | 0.544 |
| app_major_is3 | 0.538 |
| rcv_acct_age_days | 0.534 |
| mer_age_days | 0.468 |
| pair_share_of_rcv | 0.53 |
| rcv_cnt_total | 0.47 |
| rcv_uniq_snd_total | 0.471 |
| new_payee | 0.529 |
| snd_out_cnt_24h | 0.523 |
| rcv_cnt_24h | 0.481 |
| device_age_days | 0.494 |
| home_device | 0.496 |
| sim_age_hours | 0.496 |
| login_to_txn_sec | 0.497 |