LABELS = (
    "legit", "mule_account", "velocity_bot", "layering_chain", "sim_swap",
    "account_takeover", "phishing_collect", "qr_swap", "refund_abuse",
    "fake_cashback", "loan_app_extortion", "merchant_collusion",
)
ROW = (
    "hour", "weekday", "amount", "sim_age_hours", "device_age_days",
    "sender_account_age_days", "new_payee", "pin_retries", "login_to_txn_sec",
    "balance_before", "qr_vpa_mismatch", "amt_to_balance", "away_from_home",
    "home_device", "amt_is_integer", "amt_round_100", "amt_round_500",
    "amt_round_1000", "amt_just_under_1000s", "amt_ends_99", "amt_tiny_le10",
    "rcv_acct_age_days", "mer_age_days",
)
CATEGORICAL = ("receiver_type", "txn_type", "merchant_category", "app", "device_city")
RELATIONAL = (
    "rcv_cnt_1h", "rcv_cnt_24h", "rcv_cnt_total", "rcv_uniq_snd_day",
    "rcv_uniq_snd_total", "rcv_newpayee_share_day", "rcv_out_30m", "rcv_out_2h",
    "fwd_amt_ratio_30m", "snd_in_before_30m", "prev_in_amt_ratio_30m",
    "snd_out_cnt_10m", "snd_out_cnt_24h", "snd_uniq_rcv_day", "snd_in_cnt_24h",
    "dev_cnt_5m", "dev_cnt_1h", "dev_uniq_acct_total", "dev_uniq_acct_day",
    "acct_uniq_dev_total", "pair_cnt_total", "pair_cnt_7d", "pair_cnt_24h",
    "rev_pair_cnt_total", "rev_small_before_24h", "rev_big_after_24h",
    "snd_refund_cnt_30d", "snd_refund_cnt_total", "snd_collect_cnt_7d",
    "rcv_round500_cnt_7d", "pair_round500_cnt_total", "pair_share_of_rcv",
    "evening_pair_cnt_7d",
)
FEATURES = ROW + RELATIONAL + CATEGORICAL
META = ("txn_id", "timestamp", "split", "_source_order", "sender_id", "receiver_id", "device_id")
T0 = "2026-07-01"
