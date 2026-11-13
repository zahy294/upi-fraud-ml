# Conditional signal probe

38 fixed masks: initial 21 row/episode masks, 2 follow-up burst-overlap checks, 10 prior-month label-history masks, and 5 follow-up refund/phishing/QR overlap or P2P-stratum checks. History masks use strictly earlier months' labels, with July unavailable for history-based evaluation. No model fits, threshold search, test-row inspection, or test labels. All scored months are exploratory diagnostics once inspected; no untouched evaluation is claimed.

Training data SHA256: `da48ff9b2823b8460511d8f5bdc660b4a92b74cc2a64395bddb32b89710f61c5`.

No additional signal here has yet demonstrated an improvement in macro-F1. Precision below refers to each mask as a one-vs-rest prediction, without resolving overlaps.

## amount_above_balance

- 2026-07: 640 flagged. mule_account: 178 positives, precision 27.8%, recall 5.2%; sim_swap: 134 positives, precision 20.9%, recall 9.5%; account_takeover: 119 positives, precision 18.6%, recall 8.3%.
- 2026-08: 610 flagged. mule_account: 157 positives, precision 25.7%, recall 4.6%; sim_swap: 133 positives, precision 21.8%, recall 9.5%; account_takeover: 90 positives, precision 14.8%, recall 6.8%.
- 2026-09: 639 flagged. mule_account: 177 positives, precision 27.7%, recall 5.3%; sim_swap: 121 positives, precision 18.9%, recall 9.6%; account_takeover: 97 positives, precision 15.2%, recall 7.6%.

## above_balance_integer_new_p2p

- 2026-07: 552 flagged. mule_account: 162 positives, precision 29.3%, recall 4.7%; sim_swap: 124 positives, precision 22.5%, recall 8.8%; account_takeover: 118 positives, precision 21.4%, recall 8.2%.
- 2026-08: 500 flagged. mule_account: 138 positives, precision 27.6%, recall 4.0%; sim_swap: 119 positives, precision 23.8%, recall 8.5%; account_takeover: 90 positives, precision 18.0%, recall 6.8%.
- 2026-09: 501 flagged. mule_account: 154 positives, precision 30.7%, recall 4.6%; sim_swap: 106 positives, precision 21.2%, recall 8.4%; account_takeover: 97 positives, precision 19.4%, recall 7.6%.

## drain_integer_new_p2p

- 2026-07: 2582 flagged. phishing_collect: 431 positives, precision 16.7%, recall 18.9%; mule_account: 366 positives, precision 14.2%, recall 10.7%; sim_swap: 349 positives, precision 13.5%, recall 24.7%.
- 2026-08: 2389 flagged. phishing_collect: 416 positives, precision 17.4%, recall 18.5%; sim_swap: 318 positives, precision 13.3%, recall 22.8%; mule_account: 314 positives, precision 13.1%, recall 9.1%.
- 2026-09: 2353 flagged. phishing_collect: 375 positives, precision 15.9%, recall 17.3%; mule_account: 339 positives, precision 14.4%, recall 10.1%; sim_swap: 283 positives, precision 12.0%, recall 22.3%.

## drain_integer_new_p2p_fresh_sim

- 2026-07: 435 flagged. sim_swap: 230 positives, precision 52.9%, recall 16.3%; account_takeover: 104 positives, precision 23.9%, recall 7.2%; mule_account: 15 positives, precision 3.4%, recall 0.4%.
- 2026-08: 367 flagged. sim_swap: 199 positives, precision 54.2%, recall 14.3%; account_takeover: 96 positives, precision 26.2%, recall 7.2%; phishing_collect: 10 positives, precision 2.7%, recall 0.4%.
- 2026-09: 360 flagged. sim_swap: 203 positives, precision 56.4%, recall 16.0%; account_takeover: 86 positives, precision 23.9%, recall 6.7%; mule_account: 13 positives, precision 3.6%, recall 0.4%.

## drain_integer_new_p2p_new_device

- 2026-07: 367 flagged. account_takeover: 186 positives, precision 50.7%, recall 13.0%; sim_swap: 94 positives, precision 25.6%, recall 6.7%; phishing_collect: 11 positives, precision 3.0%, recall 0.5%.
- 2026-08: 343 flagged. account_takeover: 167 positives, precision 48.7%, recall 12.5%; sim_swap: 96 positives, precision 28.0%, recall 6.9%; phishing_collect: 9 positives, precision 2.6%, recall 0.4%.
- 2026-09: 329 flagged. account_takeover: 167 positives, precision 50.8%, recall 13.0%; sim_swap: 65 positives, precision 19.8%, recall 5.1%; phishing_collect: 13 positives, precision 4.0%, recall 0.6%.

## 500_then_drain_episode_1h

- 2026-07: 2 flagged. mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%; layering_chain: 0 positives, precision 0.0%, recall 0.0%.
- 2026-08: 0 flagged. mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%; layering_chain: 0 positives, precision 0.0%, recall 0.0%.
- 2026-09: 3 flagged. account_takeover: 1 positives, precision 33.3%, recall 0.1%; mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%.

## 500_then_drain_episode_1h_fresh_sim

- 2026-07: 0 flagged. mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%; layering_chain: 0 positives, precision 0.0%, recall 0.0%.
- 2026-08: 0 flagged. mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%; layering_chain: 0 positives, precision 0.0%, recall 0.0%.
- 2026-09: 0 flagged. mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%; layering_chain: 0 positives, precision 0.0%, recall 0.0%.

## 500_then_drain_episode_1h_new_device

- 2026-07: 0 flagged. mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%; layering_chain: 0 positives, precision 0.0%, recall 0.0%.
- 2026-08: 0 flagged. mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%; layering_chain: 0 positives, precision 0.0%, recall 0.0%.
- 2026-09: 2 flagged. account_takeover: 1 positives, precision 50.0%, recall 0.1%; mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%.

## 500_then_drain_episode_1h_integer_new_p2p

- 2026-07: 0 flagged. mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%; layering_chain: 0 positives, precision 0.0%, recall 0.0%.
- 2026-08: 0 flagged. mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%; layering_chain: 0 positives, precision 0.0%, recall 0.0%.
- 2026-09: 2 flagged. account_takeover: 1 positives, precision 50.0%, recall 0.1%; mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%.

## 500_then_drain_episode_1h_outside_sim_row_rule

- 2026-07: 2 flagged. mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%; layering_chain: 0 positives, precision 0.0%, recall 0.0%.
- 2026-08: 0 flagged. mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%; layering_chain: 0 positives, precision 0.0%, recall 0.0%.
- 2026-09: 3 flagged. account_takeover: 1 positives, precision 33.3%, recall 0.1%; mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%.

## 500_then_drain_episode_1h_outside_ato_row_rule

- 2026-07: 2 flagged. mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%; layering_chain: 0 positives, precision 0.0%, recall 0.0%.
- 2026-08: 0 flagged. mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%; layering_chain: 0 positives, precision 0.0%, recall 0.0%.
- 2026-09: 1 flagged. mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%; layering_chain: 0 positives, precision 0.0%, recall 0.0%.

## night_drain_new_p2p_sim_7_to_30d

- 2026-07: 21 flagged. mule_account: 11 positives, precision 52.4%, recall 0.3%; layering_chain: 4 positives, precision 19.0%, recall 0.2%; phishing_collect: 3 positives, precision 14.3%, recall 0.1%.
- 2026-08: 6 flagged. phishing_collect: 2 positives, precision 33.3%, recall 0.1%; mule_account: 1 positives, precision 16.7%, recall 0.0%; account_takeover: 1 positives, precision 16.7%, recall 0.1%.
- 2026-09: 13 flagged. mule_account: 4 positives, precision 30.8%, recall 0.1%; layering_chain: 3 positives, precision 23.1%, recall 0.1%; sim_swap: 2 positives, precision 15.4%, recall 0.2%.

## drain_new_p2p_device_7_to_30d

- 2026-07: 152 flagged. phishing_collect: 33 positives, precision 21.7%, recall 1.4%; mule_account: 28 positives, precision 18.4%, recall 0.8%; layering_chain: 19 positives, precision 12.5%, recall 0.8%.
- 2026-08: 159 flagged. phishing_collect: 28 positives, precision 17.6%, recall 1.2%; mule_account: 25 positives, precision 15.7%, recall 0.7%; sim_swap: 24 positives, precision 15.1%, recall 1.7%.
- 2026-09: 187 flagged. phishing_collect: 32 positives, precision 17.1%, recall 1.5%; mule_account: 28 positives, precision 15.0%, recall 0.8%; layering_chain: 25 positives, precision 13.4%, recall 1.1%.

## integer_new_p2p_fast_login

- 2026-07: 313 flagged. sim_swap: 45 positives, precision 14.4%, recall 3.2%; account_takeover: 14 positives, precision 4.5%, recall 1.0%; mule_account: 9 positives, precision 2.9%, recall 0.3%.
- 2026-08: 322 flagged. sim_swap: 39 positives, precision 12.1%, recall 2.8%; account_takeover: 19 positives, precision 5.9%, recall 1.4%; mule_account: 9 positives, precision 2.8%, recall 0.3%.
- 2026-09: 322 flagged. sim_swap: 43 positives, precision 13.4%, recall 3.4%; account_takeover: 15 positives, precision 4.7%, recall 1.2%; layering_chain: 8 positives, precision 2.5%, recall 0.4%.

## integer_nonhome_merchant_slow_login

- 2026-07: 207 flagged. velocity_bot: 125 positives, precision 60.4%, recall 4.4%; qr_swap: 7 positives, precision 3.4%, recall 0.3%; loan_app_extortion: 3 positives, precision 1.4%, recall 0.4%.
- 2026-08: 156 flagged. velocity_bot: 81 positives, precision 51.9%, recall 3.1%; qr_swap: 6 positives, precision 3.8%, recall 0.2%; merchant_collusion: 4 positives, precision 2.6%, recall 0.4%.
- 2026-09: 185 flagged. velocity_bot: 106 positives, precision 57.3%, recall 4.0%; qr_swap: 6 positives, precision 3.2%, recall 0.3%; loan_app_extortion: 4 positives, precision 2.2%, recall 0.6%.

## integer_nonhome_merchant_slow_login_outside_bot_rule

- 2026-07: 57 flagged. qr_swap: 7 positives, precision 12.3%, recall 0.3%; refund_abuse: 2 positives, precision 3.5%, recall 0.2%; loan_app_extortion: 2 positives, precision 3.5%, recall 0.3%.
- 2026-08: 51 flagged. qr_swap: 5 positives, precision 9.8%, recall 0.2%; merchant_collusion: 4 positives, precision 7.8%, recall 0.4%; refund_abuse: 2 positives, precision 3.9%, recall 0.2%.
- 2026-09: 57 flagged. qr_swap: 6 positives, precision 10.5%, recall 0.3%; loan_app_extortion: 3 positives, precision 5.3%, recall 0.5%; merchant_collusion: 1 positives, precision 1.8%, recall 0.1%.

## device_hour_burst_outside_bot_rule

- 2026-07: 0 flagged. mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%; layering_chain: 0 positives, precision 0.0%, recall 0.0%.
- 2026-08: 2 flagged. mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%; layering_chain: 0 positives, precision 0.0%, recall 0.0%.
- 2026-09: 0 flagged. mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%; layering_chain: 0 positives, precision 0.0%, recall 0.0%.

## fractional_refund

- 2026-07: 3555 flagged. refund_abuse: 244 positives, precision 6.9%, recall 23.4%; mule_account: 19 positives, precision 0.5%, recall 0.6%; layering_chain: 12 positives, precision 0.3%, recall 0.5%.
- 2026-08: 3670 flagged. refund_abuse: 305 positives, precision 8.3%, recall 27.4%; mule_account: 11 positives, precision 0.3%, recall 0.3%; velocity_bot: 11 positives, precision 0.3%, recall 0.4%.
- 2026-09: 3633 flagged. refund_abuse: 326 positives, precision 9.0%, recall 30.3%; mule_account: 15 positives, precision 0.4%, recall 0.4%; velocity_bot: 11 positives, precision 0.3%, recall 0.4%.

## integer_refund

- 2026-07: 37 flagged. sim_swap: 1 positives, precision 2.7%, recall 0.1%; phishing_collect: 1 positives, precision 2.7%, recall 0.0%; refund_abuse: 1 positives, precision 2.7%, recall 0.1%.
- 2026-08: 38 flagged. refund_abuse: 7 positives, precision 18.4%, recall 0.6%; phishing_collect: 1 positives, precision 2.6%, recall 0.0%; mule_account: 0 positives, precision 0.0%, recall 0.0%.
- 2026-09: 39 flagged. refund_abuse: 4 positives, precision 10.3%, recall 0.4%; phishing_collect: 2 positives, precision 5.1%, recall 0.1%; mule_account: 1 positives, precision 2.6%, recall 0.0%.

## round500_refund

- 2026-07: 0 flagged. mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%; layering_chain: 0 positives, precision 0.0%, recall 0.0%.
- 2026-08: 0 flagged. mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%; layering_chain: 0 positives, precision 0.0%, recall 0.0%.
- 2026-09: 0 flagged. mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%; layering_chain: 0 positives, precision 0.0%, recall 0.0%.

## round500_merchant_pay

- 2026-07: 287 flagged. qr_swap: 125 positives, precision 43.6%, recall 5.2%; refund_abuse: 66 positives, precision 23.0%, recall 6.3%; merchant_collusion: 59 positives, precision 20.6%, recall 7.6%.
- 2026-08: 378 flagged. qr_swap: 140 positives, precision 37.0%, recall 5.6%; merchant_collusion: 108 positives, precision 28.6%, recall 11.8%; refund_abuse: 60 positives, precision 15.9%, recall 5.4%.
- 2026-09: 342 flagged. qr_swap: 135 positives, precision 39.5%, recall 5.7%; merchant_collusion: 100 positives, precision 29.2%, recall 11.9%; refund_abuse: 53 positives, precision 15.5%, recall 4.9%.

## round500_sim_new_p2p

- 2026-07: 193 flagged. sim_swap: 119 positives, precision 61.7%, recall 8.4%; account_takeover: 40 positives, precision 20.7%, recall 2.8%; phishing_collect: 2 positives, precision 1.0%, recall 0.1%.
- 2026-08: 173 flagged. sim_swap: 104 positives, precision 60.1%, recall 7.5%; account_takeover: 44 positives, precision 25.4%, recall 3.3%; loan_app_extortion: 3 positives, precision 1.7%, recall 0.5%.
- 2026-09: 154 flagged. sim_swap: 97 positives, precision 63.0%, recall 7.7%; account_takeover: 31 positives, precision 20.1%, recall 2.4%; merchant_collusion: 2 positives, precision 1.3%, recall 0.2%.

## round500_device_new_p2p

- 2026-07: 171 flagged. account_takeover: 115 positives, precision 67.3%, recall 8.0%; sim_swap: 33 positives, precision 19.3%, recall 2.3%; phishing_collect: 2 positives, precision 1.2%, recall 0.1%.
- 2026-08: 138 flagged. account_takeover: 85 positives, precision 61.6%, recall 6.4%; sim_swap: 38 positives, precision 27.5%, recall 2.7%; refund_abuse: 1 positives, precision 0.7%, recall 0.1%.
- 2026-09: 140 flagged. account_takeover: 95 positives, precision 67.9%, recall 7.4%; sim_swap: 19 positives, precision 13.6%, recall 1.5%; phishing_collect: 2 positives, precision 1.4%, recall 0.1%.

## receiver_prior_qr_current_qr

- 2026-07: 0 flagged. mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%; layering_chain: 0 positives, precision 0.0%, recall 0.0%.
- 2026-08: 105703 flagged. mule_account: 478 positives, precision 0.5%, recall 13.9%; qr_swap: 403 positives, precision 0.4%, recall 16.2%; velocity_bot: 351 positives, precision 0.3%, recall 13.4%.
- 2026-09: 114019 flagged. mule_account: 562 positives, precision 0.5%, recall 16.7%; qr_swap: 516 positives, precision 0.5%, recall 21.9%; velocity_bot: 377 positives, precision 0.3%, recall 14.2%.

## receiver_prior_qr_current_qr_outside_qr_rule

- 2026-07: 0 flagged. mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%; layering_chain: 0 positives, precision 0.0%, recall 0.0%.
- 2026-08: 105338 flagged. mule_account: 475 positives, precision 0.5%, recall 13.8%; velocity_bot: 349 positives, precision 0.3%, recall 13.4%; qr_swap: 347 positives, precision 0.3%, recall 13.9%.
- 2026-09: 113552 flagged. mule_account: 559 positives, precision 0.5%, recall 16.6%; qr_swap: 413 positives, precision 0.4%, recall 17.6%; velocity_bot: 376 positives, precision 0.3%, recall 14.1%.

## receiver_prior_qr_current_p2p

- 2026-07: 0 flagged. mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%; layering_chain: 0 positives, precision 0.0%, recall 0.0%.
- 2026-08: 719 flagged. phishing_collect: 64 positives, precision 8.9%, recall 2.8%; qr_swap: 55 positives, precision 7.6%, recall 2.2%; layering_chain: 47 positives, precision 6.5%, recall 2.0%.
- 2026-09: 1364 flagged. phishing_collect: 124 positives, precision 9.1%, recall 5.7%; qr_swap: 107 positives, precision 7.8%, recall 4.5%; layering_chain: 55 positives, precision 4.0%, recall 2.5%.

## receiver_prior_phishing_current_collect

- 2026-07: 0 flagged. mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%; layering_chain: 0 positives, precision 0.0%, recall 0.0%.
- 2026-08: 137 flagged. phishing_collect: 54 positives, precision 39.4%, recall 2.4%; fake_cashback: 4 positives, precision 2.9%, recall 0.5%; velocity_bot: 2 positives, precision 1.5%, recall 0.1%.
- 2026-09: 283 flagged. phishing_collect: 117 positives, precision 41.3%, recall 5.4%; fake_cashback: 27 positives, precision 9.5%, recall 3.2%; loan_app_extortion: 3 positives, precision 1.1%, recall 0.5%.

## sender_prior_refund_current_refund

- 2026-07: 0 flagged. mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%; layering_chain: 0 positives, precision 0.0%, recall 0.0%.
- 2026-08: 48 flagged. refund_abuse: 21 positives, precision 43.8%, recall 1.9%; account_takeover: 1 positives, precision 2.1%, recall 0.1%; mule_account: 0 positives, precision 0.0%, recall 0.0%.
- 2026-09: 97 flagged. refund_abuse: 40 positives, precision 41.2%, recall 3.7%; sim_swap: 1 positives, precision 1.0%, recall 0.1%; qr_swap: 1 positives, precision 1.0%, recall 0.0%.

## sender_prior_2refund_current_refund

- 2026-07: 0 flagged. mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%; layering_chain: 0 positives, precision 0.0%, recall 0.0%.
- 2026-08: 6 flagged. refund_abuse: 5 positives, precision 83.3%, recall 0.4%; mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%.
- 2026-09: 31 flagged. refund_abuse: 23 positives, precision 74.2%, recall 2.1%; sim_swap: 1 positives, precision 3.2%, recall 0.1%; qr_swap: 1 positives, precision 3.2%, recall 0.0%.

## device_prior_bot_current_merchant

- 2026-07: 0 flagged. mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%; layering_chain: 0 positives, precision 0.0%, recall 0.0%.
- 2026-08: 4265 flagged. velocity_bot: 86 positives, precision 2.0%, recall 3.3%; mule_account: 19 positives, precision 0.4%, recall 0.6%; phishing_collect: 13 positives, precision 0.3%, recall 0.6%.
- 2026-09: 8153 flagged. mule_account: 37 positives, precision 0.5%, recall 1.1%; qr_swap: 37 positives, precision 0.5%, recall 1.6%; velocity_bot: 30 positives, precision 0.4%, recall 1.1%.

## device_prior_bot_current_merchant_outside_bot_rule

- 2026-07: 0 flagged. mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%; layering_chain: 0 positives, precision 0.0%, recall 0.0%.
- 2026-08: 4182 flagged. mule_account: 19 positives, precision 0.5%, recall 0.6%; phishing_collect: 13 positives, precision 0.3%, recall 0.6%; qr_swap: 13 positives, precision 0.3%, recall 0.5%.
- 2026-09: 8153 flagged. mule_account: 37 positives, precision 0.5%, recall 1.1%; qr_swap: 37 positives, precision 0.5%, recall 1.6%; velocity_bot: 30 positives, precision 0.4%, recall 1.1%.

## receiver_prior_anyfraud_current_p2p

- 2026-07: 0 flagged. mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%; layering_chain: 0 positives, precision 0.0%, recall 0.0%.
- 2026-08: 5435 flagged. mule_account: 306 positives, precision 5.6%, recall 8.9%; layering_chain: 267 positives, precision 4.9%, recall 11.6%; phishing_collect: 211 positives, precision 3.9%, recall 9.4%.
- 2026-09: 9712 flagged. mule_account: 501 positives, precision 5.2%, recall 14.9%; layering_chain: 449 positives, precision 4.6%, recall 20.6%; phishing_collect: 332 positives, precision 3.4%, recall 15.4%.

## receiver_prior_2fraud_current_p2p

- 2026-07: 0 flagged. mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%; layering_chain: 0 positives, precision 0.0%, recall 0.0%.
- 2026-08: 653 flagged. mule_account: 89 positives, precision 13.6%, recall 2.6%; layering_chain: 44 positives, precision 6.7%, recall 1.9%; sim_swap: 31 positives, precision 4.7%, recall 2.2%.
- 2026-09: 1690 flagged. mule_account: 183 positives, precision 10.8%, recall 5.4%; layering_chain: 129 positives, precision 7.6%, recall 5.9%; phishing_collect: 91 positives, precision 5.4%, recall 4.2%.

## receiver_prior_phishing_current_collect_outside_phish_rule

- 2026-07: 0 flagged. mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%; layering_chain: 0 positives, precision 0.0%, recall 0.0%.
- 2026-08: 82 flagged. phishing_collect: 11 positives, precision 13.4%, recall 0.5%; velocity_bot: 2 positives, precision 2.4%, recall 0.1%; mule_account: 1 positives, precision 1.2%, recall 0.0%.
- 2026-09: 143 flagged. phishing_collect: 18 positives, precision 12.6%, recall 0.8%; fake_cashback: 12 positives, precision 8.4%, recall 1.4%; mule_account: 2 positives, precision 1.4%, recall 0.1%.

## receiver_prior_phishing_current_collect_p2p_outside_phish_rule

- 2026-07: 0 flagged. mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%; layering_chain: 0 positives, precision 0.0%, recall 0.0%.
- 2026-08: 82 flagged. phishing_collect: 11 positives, precision 13.4%, recall 0.5%; velocity_bot: 2 positives, precision 2.4%, recall 0.1%; mule_account: 1 positives, precision 1.2%, recall 0.0%.
- 2026-09: 143 flagged. phishing_collect: 18 positives, precision 12.6%, recall 0.8%; fake_cashback: 12 positives, precision 8.4%, recall 1.4%; mule_account: 2 positives, precision 1.4%, recall 0.1%.

## receiver_prior_qr_current_p2p_outside_qr_rule

- 2026-07: 0 flagged. mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%; layering_chain: 0 positives, precision 0.0%, recall 0.0%.
- 2026-08: 649 flagged. phishing_collect: 62 positives, precision 9.6%, recall 2.8%; layering_chain: 47 positives, precision 7.2%, recall 2.0%; mule_account: 10 positives, precision 1.5%, recall 0.3%.
- 2026-09: 1239 flagged. phishing_collect: 120 positives, precision 9.7%, recall 5.6%; layering_chain: 55 positives, precision 4.4%, recall 2.5%; fake_cashback: 32 positives, precision 2.6%, recall 3.9%.

## sender_prior_refund_current_refund_outside_refund_rule

- 2026-07: 0 flagged. mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%; layering_chain: 0 positives, precision 0.0%, recall 0.0%.
- 2026-08: 27 flagged. refund_abuse: 2 positives, precision 7.4%, recall 0.2%; mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%.
- 2026-09: 49 flagged. refund_abuse: 3 positives, precision 6.1%, recall 0.3%; mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%.

## sender_prior_2refund_current_refund_outside_refund_rule

- 2026-07: 0 flagged. mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%; layering_chain: 0 positives, precision 0.0%, recall 0.0%.
- 2026-08: 1 flagged. mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%; layering_chain: 0 positives, precision 0.0%, recall 0.0%.
- 2026-09: 2 flagged. mule_account: 0 positives, precision 0.0%, recall 0.0%; velocity_bot: 0 positives, precision 0.0%, recall 0.0%; layering_chain: 0 positives, precision 0.0%, recall 0.0%.

