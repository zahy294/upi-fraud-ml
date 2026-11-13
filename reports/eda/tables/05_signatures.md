## Signature rules evaluated on September (train); firing rate compared with October (test)
coverage = % of that label's Sep rows the rule catches; precision = % of fired Sep rows that carry the label.

| label | rule | coverage_% | precision_% | any_fraud_precision_% | fires_Sep_% | fires_Oct_% | top_labels_when_fired |
|---|---|---|---|---|---|---|---|
| mule_account | P2P & rcv_uniq_snd_day>=3 | 23.9 | 59.1 | 85.3 | 0.267 | 0.285 | {'mule_account': 59.1, 'layering_chain': 16.5, 'legit': 14.7} |
| layering_chain | forwarded 80-100% within 30m (in or out) | 21.7 | 69.7 | 86.5 | 0.133 | 0.123 | {'layering_chain': 69.7, 'mule_account': 14.4, 'legit': 13.5} |
| velocity_bot | dev_cnt_5m>=3 | login_to_txn_sec<=5 | 34.7 | 14.7 | 19.1 | 1.23 | 1.11 | {'legit': 80.9, 'velocity_bot': 14.7, 'sim_swap': 1.4} |
| sim_swap | sim_age_hours<168 | 26.2 | 2.81 | 7.06 | 2.32 | 2.42 | {'legit': 92.9, 'sim_swap': 2.8, 'account_takeover': 1.2} |
| account_takeover | device_age_days<7 | 25.2 | 2.6 | 7.34 | 2.44 | 2.37 | {'legit': 92.7, 'account_takeover': 2.6, 'velocity_bot': 1.1} |
| phishing_collect | COLLECT & amount just under round (x950-x999 or ends 99) | 20.8 | 47 | 56.9 | 0.188 | 0.202 | {'phishing_collect': 47.0, 'legit': 43.1, 'fake_cashback': 5.1} |
| qr_swap | qr_vpa_mismatch==1 | (QR & P2P) | 24.8 | 45.6 | 50.7 | 0.251 | 0.225 | {'legit': 49.3, 'qr_swap': 45.6, 'phishing_collect': 1.6} |
| refund_abuse | REFUND & sender refunds in +-15d >= 3 | 30.3 | 81.5 | 85 | 0.0784 | 0.0729 | {'refund_abuse': 81.5, 'legit': 15.0, 'velocity_bot': 1.0} |
| fake_cashback | reverse <=10 before | (<=10 & reverse >=100 after) | 23.6 | 55.8 | 86.6 | 0.0688 | 0.066 | {'fake_cashback': 55.8, 'phishing_collect': 28.2, 'legit': 13.4} |
| loan_app_extortion | lending merchant | (late evening & pair_cnt_7d>=2) | 24 | 2.21 | 8.12 | 1.37 | 1.13 | {'legit': 91.9, 'loan_app_extortion': 2.2, 'velocity_bot': 1.4} |
| merchant_collusion | MERCHANT & round-500 amount & pair has >=2 round-500 txns | 23.8 | 46.6 | 86.9 | 0.0841 | 0.0756 | {'merchant_collusion': 46.6, 'qr_swap': 21.7, 'refund_abuse': 15.6} |
| (any) known fraud receiver | P2P receiver had fraud inbound in an earlier month |  |  | 23.2 | 1.9 | 2.69 | {'legit': 76.8, 'mule_account': 5.2, 'layering_chain': 4.6} |

## How much of each label is 'on-pattern'? (train, all months)
|  | own_signature_% | any_signature_% |
|---|---|---|
| legit |  | 7 |
| mule_account | 23.9 | 33 |
| velocity_bot | 35.5 | 43.4 |
| layering_chain | 21 | 37.7 |
| sim_swap | 25.7 | 38.5 |
| account_takeover | 24.4 | 37.5 |
| phishing_collect | 22.1 | 33.6 |
| qr_swap | 25.3 | 34.5 |
| refund_abuse | 28.3 | 41.2 |
| fake_cashback | 23.6 | 36.7 |
| loan_app_extortion | 24.1 | 33.9 |
| merchant_collusion | 20.8 | 29.9 |

## Connectedness: % of a label's rows linked to ANOTHER row of the same label (train)
|  | sender ±24h | receiver ±24h | device ±1h | money-flow ±2h | any link |
|---|---|---|---|---|---|
| mule_account | 2.5 | 41.6 | 1.9 | 9.9 | 47.8 |
| velocity_bot | 12.5 | 51.3 | 35.1 | 0.8 | 51.7 |
| layering_chain | 0.2 | 24.8 | 0 | 22 | 46.7 |
| sim_swap | 35.5 | 10.6 | 34.1 | 0 | 45.8 |
| account_takeover | 34.5 | 9.8 | 33.4 | 0 | 44.1 |
| phishing_collect | 3.5 | 16.5 | 2.9 | 2.3 | 19.2 |
| qr_swap | 3.5 | 21 | 1 | 0.1 | 21.1 |
| refund_abuse | 23.4 | 32.9 | 3.4 | 0 | 36.4 |
| fake_cashback | 10.1 | 20.1 | 9.6 | 11.1 | 27.1 |
| loan_app_extortion | 14.4 | 8.6 | 4.6 | 0 | 23 |
| merchant_collusion | 22.5 | 35 | 5 | 0 | 35.1 |

## Single-signal prevalence: Sep legit vs Sep all vs Oct (%)
If 'Oct all' sits far below 'Sep all' (toward 'Sep legit'), that fraud pattern got rarer or changed shape in October.
|  | Sep legit | Sep all | Oct all |
|---|---|---|---|
| login<=5s_% | 1.024 | 1.162 | 1.086 |
| non-home device_% | 7.911 | 8.146 | 8.122 |
| dev_cnt_5m>=3_% | 0.027 | 0.219 | 0.11 |
| dev_cnt_1h>=3_% | 0.08 | 0.399 | 0.265 |
| dev_uniq_acct_total>=3_% | 0.243 | 0.426 | 0.307 |
| sim_age<168h_% | 2.238 | 2.319 | 2.42 |
| device_age<7d_% | 2.347 | 2.441 | 2.371 |
| lending_% | 1.32 | 1.37 | 1.307 |
| lending & 20-23h_% | 0.183 | 0.214 | 0.195 |
| late evening & pair_cnt_7d>=2_% | 0.466 | 0.503 | 0.295 |
| COLLECT & just-under_% | 0.073 | 0.177 | 0.195 |
| QR & P2P_% | 0.018 | 0.135 | 0.122 |
| qr_vpa_mismatch==1_% | 0.12 | 0.196 | 0.146 |

velocity_bot (Sep) gap between consecutive payments on the same device: median 8s, p75 18s, p90 36s; rows per device: median 1
Oct, non-home-device rows: gap between consecutive payments on the same device: median 362511s, share of gaps <=120s 4.5%  (Sep non-home: 7.0%)