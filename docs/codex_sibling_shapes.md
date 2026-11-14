# Exact tells for sibling-shaped fraud events

14 November 2026. **No new qualifying tell found for any of the six pairs.** The searches found shared recipes, weak distribution differences, and coincidences that failed another month. None supplied an exact label-free reassignment covering at least 30 reporting events with support in July, August and September. This is a negative result for the checks performed, not proof that no undiscovered tell exists.

No model or weights were fitted, and no predictions were changed. Every Python run used `C:/Users/ASUS/AppData/Local/Programs/Python/Python311/python.exe -I`. Numerical libraries were limited to four threads before import; both loaded OpenBLAS pools reported four. Nothing was written under `data/`, `models/` or `submissions/`, and no pre-existing files were edited.

## Evidence and runnable files

Complete side-by-side event prints were generated **before** the parameter/rule scans. Each comparison has two examples per month from each side, including the two separate phishing comparisons. Every in-scope row in a selected unit is printed, without a row cap or label-based trimming.

- [Reporting-component prints](codex_sibling_events.txt): the supplied detected-row grouping; all band-zero detected members of each selected component.
- [Reconstructed-sequence prints](codex_sibling_physical_events.txt): finer label-free units, including eligible companions predicted legit. These expose complete forwarding sequences that the reporting graph splits into separate components.
- [Parameter scan](codex_sibling_parameter_scan.txt): 151 observable parameters per reconstructed unit, July-ranked candidates and outcomes in all three months.
- [Targeted checks](codex_sibling_targeted_checks.txt): recurrence, transaction order, device position, payout accounting and repeated-sender checks; monthly cleaned/raw counts and purity diagnostics.
- [Runnable analysis](../scripts/explore/18_codex_sibling_shapes.py): loading, grouping, exact parameter definitions, scans and evaluation. This is the only new Python file for this task. The four text files above and this Markdown file are the other files created.

Run from the repository root. These commands print to the terminal:

```powershell
& C:/Users/ASUS/AppData/Local/Programs/Python/Python311/python.exe -I scripts/explore/18_codex_sibling_shapes.py --mode inspect
& C:/Users/ASUS/AppData/Local/Programs/Python/Python311/python.exe -I scripts/explore/18_codex_sibling_shapes.py --mode physical
& C:/Users/ASUS/AppData/Local/Programs/Python/Python311/python.exe -I scripts/explore/18_codex_sibling_shapes.py --mode scan
& C:/Users/ASUS/AppData/Local/Programs/Python/Python311/python.exe -I scripts/explore/18_codex_sibling_shapes.py --mode targeted
```

Append, for example, `--output docs/my_new_sibling_scan.txt` to retain a new log. Output must be a **new** file under `docs/`; existing files are refused. The code does not call a model-fitting API. Its parameter extractor rejects frames containing `label`, `raw`, `clean`, `app_version`, `app_major` or `ip_risk_score`; that rejection was checked directly. App-major share is computed only after a candidate selects its rows.

## Population, grouping and evaluation

The 1,564,060 train rows retain base-file order. The OOF array and cleaned-label vector have that many rows, and all five feature packs have matching transaction IDs. `pred = argmax(oof * weights_s5_c3)` reproduces **every supplied row and event count**. There are 18,872 detected band-zero training rows overall. No band override is applied.

Reporting components follow the supplied clarification: connect consecutive detected rows sharing sender within seven days, eligible receiver within one day, device within ten minutes, or unordered P2P pair within seven days; then take undirected connected components. Receiver links apply to P2P or `S_flag == 1`; merchant and account namespaces remain separate. As specified, band rows can connect this reporting graph, but **no band-positive row supplies numerical/sequence attributes to a candidate rule**.

Parameter units are built without labels/traps from all band-zero training rows, including nondetected companions:

| Family | Exact reconstruction |
|---|---|
| Drain | Whole-rupee P2P PAY, new payee; same sender, consecutive eligible gap at most 7,200 seconds. |
| Bait/payback | Same unordered P2P pair, consecutive gap at most 604,800 seconds; retain direction, amounts and transaction types. |
| Forwarding chain | Whole-rupee P2P PAY; find the most recent eligible receipt into the sender, strictly earlier by at most 7,200 seconds. Link if `0.90 <= child_amount / receipt_amount <= 1`; take connected components of those links. |
| Fan-in | Whole-rupee P2P PAY, new payee, amount divisible by ten; same receiver, consecutive gap at most 86,400 seconds. Attach P2P PAY outflows by that receiver strictly after the last receipt, within 7,200 seconds. A payout anchor may recover the preceding receiver session from receipts in the preceding three days if the last receipt is within two hours. |
| Merchant session | `S_flag == 1`: merchant PAY/QR with amount in `{1000,2000,2500,5000,10000}`; same receiver, consecutive gap at most 86,400 seconds. Other QR anchors are singletons. |

An anchor outside a recognized pool gets a singleton. These are reproducible observational units, not guaranteed generator IDs: band exclusion, missed links and unrelated coincident transactions can fragment or contaminate them. Sequence ties use `(t, txn_id)`. Gap cutoffs are inclusive except the explicitly strict-before/after receipt/payout relations.

| Prediction / cleaned type | Rows | Reporting events | Reconstructed units containing those rows |
|---|---:|---:|---:|
| sim_swap / sim_swap | 1,229 | 431 | 515 |
| sim_swap / account_takeover | 402 | 160 | 160 |
| fake_cashback / fake_cashback | 728 | 308 | 318 |
| fake_cashback / phishing_collect | 276 | 156 | 160 |
| phishing_collect / phishing_collect | 2,385 | 2,150 | 2,258 |
| phishing_collect / fake_cashback | 177 | 168 | 176 |
| phishing_collect / loan_app_extortion | 73 | 69 | 73 |
| layering_chain / layering_chain | 3,018 | 2,214 | 1,848 |
| layering_chain / mule_account | 319 | 319 | 106 |
| mule_account / mule_account | 3,188 | 364 | 523 |
| mule_account / velocity_bot | 332 | 36 | 19 |
| qr_swap / qr_swap | 2,780 | 2,346 | 2,386 |
| qr_swap / refund_abuse | 49 | part of combined 71 | 22 |
| qr_swap / merchant_collusion | 49 | part of combined 71 | 49 |

Different labels can share a unit, so counts are not necessarily additive. The 147 October-dated training tails are assigned to September, matching the modelling convention. A reporting component spanning months can appear in both monthly counts; `ALL` deduplicates globally.

Every candidate is gated by the stated original prediction and `band == 0`, then its observable predicate. **Other cleaned labels and legit rows stay in the denominator.** Right means that the proposed class equals the row's cleaned/raw label; wrong means it does not. An event is right only when **all moved rows in that component/month** are right. Otherwise it is wrong; mixed-event counts are retained in the logs. This conservative convention avoids inventing event labels from noisy raw labels.

For an *exact* qualifying tell I required zero cleaned-label counterexamples, nonempty support in each month, and at least 30 supplied reporting components overall. Raw disagreement is reported separately. The first scan ranks candidates on July only, then checks August/September. Targeted follow-ups are descriptive checks after inspection, not independent validation. No rule passed.

## 1. Predicted sim_swap: sim_swap versus account_takeover

**No tell found.** Both types have cent-rounded quarter-balance recurrences, two-to-four-payment support, whole-rupee amounts and persistent device use. Burst length remains a statistical difference.

Two complete two-payment sequences illustrate the shared shape. Each has one sender and one device throughout; every row is P2P PAY/new-payee:

| Clean type | Transaction | Gap seconds | Amount | Balance before | Home device |
|---|---|---:|---:|---:|---:|
| sim_swap | TX80202d80 | 0 | 2,988 | 4,057.98 | 0 |
| sim_swap | TX0a7810b5 | 456 | 776 | 1,014.49 | 0 |
| account_takeover | TX40f7fff1 | 0 | 896 | 946.93 | 1 |
| account_takeover | TXd81cb897 | 30 | 500 | 236.73 | 1 |

I tested whether one latent unrounded starting balance generates **all** balances by quartering and rounding to cents:

```python
scale = 0.25 ** np.arange(len(B))
latent_quarter = len(B) >= 2 and (
    np.max((B - 0.00500001) / scale)
    <= np.min((B + 0.00500001) / scale)
)
```

The extra `1e-8` rupee allows representation error. Other checks: rounding each previous observed balance; floor/ceil in cents; residual signs; the `.35` alternative; first-balance cents modulo 2/4/5/10/25/100; amount rounding against eleven constants; full gap tuples, gcd/moduli; and non-home positions.

| Type | July quarter-compatible units | August | September |
|---|---:|---:|---:|
| sim_swap | 142 | 119 | 115 |
| account_takeover | 59 | 53 | 47 |

Both groups have median maximum adjacent-quarter residual **0.0025 rupee**, and 75th percentile **0.005 rupee**. Rounded-from-previous balances hold in 284 SIM units and 128 ATO units. Non-home-only-first/last selects **zero** quarter-compatible multirow units: non-home device use generally persists through the burst.

Rejected: quarter-compatible two-row unit, reassign sim_swap to ATO.

| Month | Rows | Events | Clean rows right/wrong | Clean events right/wrong | Raw rows right/wrong | Raw events right/wrong |
|---|---:|---:|---:|---:|---:|---:|
| July | 172 | 85 | 63 / 109 | 30 / 55 | 55 / 117 | 22 / 63 |
| August | 143 | 72 | 45 / 98 | 23 / 49 | 41 / 102 | 19 / 53 |
| September | 137 | 69 | 47 / 90 | 23 / 46 | 44 / 93 | 20 / 49 |

Quarter recurrence plus all gaps divisible by 30 seconds is 6/0 right/wrong rows in July, but **2/10** in August and **4/2** in September; only 11 reporting events overall. This is a clock coincidence, not a tell.

## 2. Predicted fake_cashback: fake_cashback versus phishing_collect

**No tell found.** One-bait overlap survives exact bait amount/order, reverse direction, gap seconds, payback type, balance-floor residuals, cents residues and device placement. Both sides use baits from `{1,2,5,10}`, and both have PAY and COLLECT paybacks.

Complete one-bait/COLLECT examples:

| Clean type | Bait ID / amount / balance | Payback ID / amount / balance | Gap seconds |
|---|---|---|---:|
| fake_cashback | TXe1042ad9 / 10 / 3,747.08 | TX40cd36aa / 8,876 / 9,763.60 | 2,388 |
| phishing_collect | TXd688748e / 2 / 4,865.28 | TXfb59956d / 12,217 / 13,438.70 | 5,545 |

Both payback balances equal `1.1 * amount` at currency precision. First-row account attributes and device ownership give no repeatable split.

Rejected: exactly one reverse PAY bait of at most ten rupees, last-bait-to-payback gap at most 10,800 seconds, and last large payback is COLLECT; reassign qualifying predicted-cashback members to phishing. This has statistical enrichment but is neither exact nor a new feature family.

| Month | Rows | Events | Clean rows right/wrong | Clean events right/wrong | Raw rows right/wrong | Raw events right/wrong |
|---|---:|---:|---:|---:|---:|---:|
| July | 73 | 36 | 41 / 32 | 19 / 17 | 35 / 38 | 14 / 22 |
| August | 85 | 45 | 49 / 36 | 26 / 19 | 42 / 43 | 19 / 26 |
| September | 82 | 43 | 57 / 25 | 29 / 14 | 53 / 29 | 26 / 17 |

Requiring instead that payback balance exceed the `1.1` floor by more than 0.011 rupee gives 22/29, 26/30 and 35/38 right/wrong rows.

The two-bait positive control is already consumed by the model. Among predicted-cashback paybacks it selects 152 rows in 147 reporting events: 149 match the supplied cleaned cashback label, 135 match raw. The three cleaned counterexamples mean “always” is not literal under this label file; I did not relabel them. Among **predicted-phishing paybacks, the same predicate selects zero rows in every month**. It offers no residual moves.

## 3. Predicted phishing_collect: fake_cashback and loan_app_extortion siblings

**No tell found for either sibling.** Under label-free pair reconstruction, 155 of 176 cashback-containing units are single COLLECTs, as are 71 of 73 loan-containing units. No gap/order parameter exists inside those singleton pair units.

Complete singleton examples are `TXaa48cc12` (clean phishing, COLLECT 9,999, balance 10,998.90), `TX06932723` (clean cashback, COLLECT 4,999, balance 19,013.25), and `TXb3eb729f` (clean loan, COLLECT 4,999, balance 5,498.90). The loan example shares the native phishing `1.1` floor. Other examples/months are printed side by side in the logs.

COLLECT amount palettes overlap: amounts near 2,000/5,000/10,000/25,000, ending in 50, 90 or 99. Exact amount 4,999 occurs in **326 phishing, 39 cashback and 18 loan** COLLECT rows in this prediction cohort. Arbitrary whole amounts also occur. First-row floor/rounding, account age, type/category, device ownership and amount sets yielded no qualifying split.

I expanded to same-sender P2P COLLECT histories, split at consecutive gaps greater than seven days. Of the 71 loan-labelled COLLECT rows, 59 are in size-one histories and 12 in size-two histories; none is in a history of three or more.

Rejected cashback candidate: one reverse bait and payback delay greater than 3,600 seconds.

| Month | Rows | Events | Clean rows right/wrong | Clean events right/wrong | Raw rows right/wrong | Raw events right/wrong |
|---|---:|---:|---:|---:|---:|---:|
| July | 8 | 8 | 6 / 2 | 6 / 2 | 5 / 3 | 5 / 3 |
| August | 4 | 3 | 0 / 4 | 0 / 3 | 0 / 4 | 0 / 3 |
| September | 14 | 13 | 6 / 8 | 5 / 8 | 6 / 8 | 5 / 8 |

Only 24 reporting events overall. Two reverse baits select no residual paybacks.

Rejected loan candidate: same-sender COLLECT history has at least two rows and all receivers differ.

| Month | Rows | Events | Clean rows right/wrong | Clean events right/wrong | Raw rows right/wrong | Raw events right/wrong |
|---|---:|---:|---:|---:|---:|---:|
| July | 119 | 78 | 2 / 117 | 2 / 76 | 2 / 117 | 2 / 76 |
| August | 123 | 87 | 7 / 116 | 4 / 83 | 6 / 117 | 3 / 84 |
| September | 70 | 53 | 3 / 67 | 2 / 51 | 2 / 68 | 1 / 52 |

Three COLLECTs gives **zero** correct loan rows in every month; so does requiring repeated exact amounts. These repeated-sender checks do not isolate cashback either.

## 4. Predicted layering_chain: layering_chain versus mule_account

**No tell found.** The 319 mule-labelled rows span 319 reporting components but only 106 reconstructed chain units. The supplied reporting graph does not directly link an incoming payment to a subsequent outgoing payment by its receiver.

Per-hop amount ratios overlap to more decimals across multirow units:

| Quantity | Layering-labelled units | Mule-labelled units |
|---|---:|---:|
| Smallest observed chain minimum ratio | 0.965084 | 0.965036 |
| Median chain minimum ratio | 0.972015 | 0.972414 |
| Median chain maximum ratio | 0.988107 | 0.989327 |
| Largest observed chain maximum ratio | 0.994992 | 0.994914 |

These are observed extrema/medians, not fitted constants. I tested complete ratio tuples to ten decimals, gap tuples, gcd/divisibility, equality of successive ratios, first/last amount/balance relations, chronological amount monotonicity, transaction/new-payee order, and non-home-device position/owner. Multirow chains are PAY sequences on both sides.

A literal July gap tuple `(274,496,1120,944,626)` selects six correct reporting components, but they are **one physical chain**, and the tuple never recurs in August or September. Memorizing it is not a tell.

Rejected: at least four chain rows, every linked child/parent amount ratio between 0.965 and 0.995 inclusive; reassign predicted layering rows to mule.

| Month | Rows | Events | Clean rows right/wrong | Clean events right/wrong | Raw rows right/wrong | Raw events right/wrong |
|---|---:|---:|---:|---:|---:|---:|
| July | 540 | 523 | 83 / 457 | 82 / 441 | 91 / 449 | 88 / 435 |
| August | 539 | 529 | 102 / 437 | 102 / 427 | 110 / 429 | 108 / 421 |
| September | 488 | 478 | 74 / 414 | 70 / 408 | 75 / 413 | 70 / 408 |

“Non-home only first” in multirow chains gives 6/32, 3/18 and 4/30 right/wrong rows; “non-home only last” gives 6/26, 17/39 and 5/20. Both positions occur in both types. A longer chain remains an enrichment, not an exact generator identifier.

## 5. Predicted mule_account: mule_account versus velocity_bot

**No tell found; independent event support is particularly small.** The bot side comprises only **3 / 10 / 6 reconstructed fan-in units** in July/August/September, versus 6 / 18 / 12 reporting components. A perfect rule on all 19 observed units would still not constitute 30 independent physical examples, even though it could exceed 30 under the supplied reporting IDs.

The fuller sequences share a broad recipe: many whole multiples of ten into one account, followed by one to three payouts. Among units with at least three incoming rows:

| Parameter | Mule-labelled units | Bot-labelled units |
|---|---:|---:|
| Units | 179 | 18 |
| One / two / three attached payouts | 63 / 51 / 54 | 5 / 6 / 6 |
| No attached payout | 11 | 1 |
| Every attached payout balance within 0.011 of `1.02 * amount` | 94 | 8 |
| All incoming amounts distinct | 139 | 9 |
| Sum of payouts equals first payout's balance | 0 | 0 |

I tested total outgoing versus total incoming amounts, individual payouts, decreasing payout amounts/balances, gaps before and between payouts, repeated incoming amounts, complete incoming amount sets and non-home-device positions. For every coefficient in `{.5,.6,.65,.7,.75,.8,.85,.9,.95,.98,1}`, neither group's units satisfy `sum(payout_amounts) == round(coefficient * sum(incoming_amounts))`. This concerns the reconstructed in-scope units, not a universal conservation law; exclusion and timing can omit part of a real flow.

Rejected: at least three incoming rows, at least one outgoing row, and **every** outgoing balance within 0.011 rupee of `1.02 * amount`; reassign predicted mule rows to bot.

| Month | Rows | Events | Clean rows right/wrong | Clean events right/wrong | Raw rows right/wrong | Raw events right/wrong |
|---|---:|---:|---:|---:|---:|---:|
| July | 649 | 77 | 23 / 626 | 2 / 75 | 21 / 628 | 1 / 76 |
| August | 659 | 73 | 93 / 566 | 10 / 63 | 82 / 577 | 6 / 67 |
| September | 489 | 54 | 45 / 444 | 4 / 50 | 41 / 448 | 2 / 52 |

The July-ranked “first sender's bank B20” coincidence is 23/13 right/wrong rows in July, but **0/19 and 0/27** in August/September. That and cents-residue coincidences are discarded. They provide no evidence for a bank or rounding rule.

## 6. Predicted qr_swap: qr_swap versus merchant_collusion/refund_abuse

**No tell found for either target.** Refund and collusion were tested separately: “either sibling” is not a usable 12-class output.

Of 49 collusion-containing units, 46 are singleton QR units, two have QR/QR order and one has PAY/QR/PAY order. Refund-containing units have one to three rows and share the fixed-amount merchant palette with small QR-shaped sessions. Type order, amount-set membership, balances at the `1.1` floor, repeated senders, non-home position/owner and first-row merchant/account relations overlap.

The printed singleton examples `TX61a9db12` (clean QR: grocery-category QR to a person, amount 610, balance 15,719.44) and `TXa02bffe0` (clean collusion: the same transaction/category/receiver-type shape, amount 669.54, balance 15,774.47) illustrate the native-QR overlap. Full merchant sessions, including mixed PAY/QR sequences, are in the accompanying prints.

Rejected refund candidate: exactly three rows in a fixed-amount merchant session, chronological types `QR,PAY,PAY`.

| Month | Rows | Events | Clean rows right/wrong | Clean events right/wrong | Raw rows right/wrong | Raw events right/wrong |
|---|---:|---:|---:|---:|---:|---:|
| July | 18 | 6 | 6 / 12 | 2 / 4 | 6 / 12 | 2 / 4 |
| August | 24 | 8 | 0 / 24 | 0 / 8 | 1 / 23 | 0 / 8 |
| September | 15 | 5 | 0 / 15 | 0 / 5 | 0 / 15 | 0 / 5 |

Only 19 reporting components overall. For collusion, a QR singleton whose non-home device belongs to a transaction participant selects **zero** rows in every month. The highest-July-supported cents coincidence, first amount's cents modulo ten equal to four, yields just 3/49, 0/56 and 1/46 right/wrong rows. No account/merchant relation improved this into an exact rule.

## Reference tables, purity and uncertainty

For the 9,045 distinct first rows used by these units, a fresh join to `accounts.csv` reproduced sender age-from-opened with **zero mismatches**. The 540 merchant first rows had **zero merchant age mismatches** against `merchants.csv`. Only one first-row amount equalled its sender's age in days; none equalled merchant age. First-row bank/OS/KYC, sender-versus-receiver KYC/city, merchant category/city, and whether a non-home device belonged to another participant were checked as event attributes, not label lookups. No relation reached the support and cross-month requirement.

App-major-3 is only a post-selection purity check. The rejected two-row quarter-drain selector has shares **0.640 / 0.587 / 0.642**; the rejected long-chain selector **0.631 / 0.627 / 0.639**; the rejected all-1.02-payout selector **0.561 / 0.631 / 0.632**. They usually select patterned activity while getting its **type** wrong. Purity does not resolve sibling typing. No rule uses app version/major or IP risk, directly or through an aggregate.

The claim that cleaned labels are constant within every event needs qualification. Among scoped predicted rows within reconstructed units, cleaned labels are mixed in **23/711 drain**, **32/479 predicted-cashback pair**, **14/2,790 predicted-phishing pair**, **189/1,944 chain**, **11/569 fan-in**, and **23/2,588 QR/shop** units. This can reflect grouping imperfections and label cleaning. A concrete two-row quarter-drain has `TX64b64b31` raw ATO/clean QR and `TX8932bf3d` raw QR/clean ATO: cleaned labels are not constant even in that tight sequence. Some complete chains likewise carry multiple cleaned classes. I did not use majority labels to hide these contradictions or alter `y_clean.npy`.

The scan covered fixed recipe-scale inequalities and exact-value predicates over 151 parameters (1,077–3,135 predicate instances per prediction family), plus targeted conjunctions and sender histories. Exact whole amount/gap/ratio tuples were allowed to expose repetition; one-event memorization and absent-month matches were rejected. Predicates are correlated, and no significance claim is attached to this search. The evidence supports **no actionable exact tell found**, not a new ceiling estimate or measured F1 improvement.

Input SHA-256 fingerprints:

```text
base_v0.parquet       e88ee1fa76cc49613049c136534b6c7b16189c1859500caf07261f0ea2800cd9
y_clean.npy          f26d496b3802f260df34e80a5c5fa8b4a071d2731a305f6c4241de0766a427ca
oof_c3/oof.npy       3003dc59114f257e02089a3b94ee0a6a9279a21847a691b179f983bdc5373deb
weights_s5_c3.npy    94c53b602c087799cef814f6e2e11b566b2d6cd633f5b24c959f03f935331f39
```
