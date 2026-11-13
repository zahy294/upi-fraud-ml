from __future__ import annotations

LABELS = [
    "legit",
    "mule_account",
    "velocity_bot",
    "layering_chain",
    "sim_swap",
    "account_takeover",
    "phishing_collect",
    "qr_swap",
    "refund_abuse",
    "fake_cashback",
    "loan_app_extortion",
    "merchant_collusion",
]
FRAUD_LABELS = LABELS[1:]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = dict(enumerate(LABELS))

RAW_TRANSACTION_COLUMNS = [
    "txn_id", "timestamp", "sender_id", "receiver_id", "receiver_type",
    "device_id", "amount", "txn_type", "merchant_category", "device_city",
    "sim_age_hours", "device_age_days", "sender_account_age_days", "new_payee",
    "pin_retries", "login_to_txn_sec", "balance_before", "app", "app_version",
    "ip_risk_score", "qr_vpa_mismatch",
]

NON_FEATURE_COLUMNS = {
    "txn_id", "timestamp", "sender_id", "receiver_id", "device_id", "label",
    "split", "month", "is_tail", "context_source",
}

