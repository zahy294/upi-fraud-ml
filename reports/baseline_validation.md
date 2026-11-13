# Verified local baseline

Run completed 13 November 2026 on the full organizer dataset using `configs/baseline_lgb.yaml`.

- Fit: 1,053,917 July-August rows.
- Calibration: 390,780 rows from 1-23 September; macro-F1 0.4652.
- Frozen assessment: 119,216 rows from 24-30 September; macro-F1 0.4655.
- Full September: macro-F1 0.4657.
- Best LightGBM iteration: 73.

Assessment F1 by official class: legit 0.9871; mule_account 0.3762; velocity_bot 0.5478; layering_chain 0.4759; sim_swap 0.3650; account_takeover 0.3764; phishing_collect 0.4400; qr_swap 0.4756; refund_abuse 0.4171; fake_cashback 0.3704; loan_app_extortion 0.4358; merchant_collusion 0.3187.

The final-fit path also completed, producing 260,582 predictions in sample order, all 12 classes represented, a 4.72 MB CSV, and SHA-256 `7e14af84541cff7a6a6ab2852dfcddb52c6bdb9751e49a285af6f915cdfff657`. The CSV and fitted artifacts are intentionally Git-ignored. These noisy-label local metrics are engineering verification, not an expected public score.
