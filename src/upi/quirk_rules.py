"""Rows the legit generator cannot have produced.

In this data `qr_vpa_mismatch` is filled exactly when a merchant is paid by QR. That holds for every row on an
ordinary new device (the fully occupied legit block: 45,177 train rows, 7,462 test rows) and for every legit-looking
row in training. Fraud generators break it in two ways, and the models already catch more than 99% of those rows
in training. A few October rows slip through; this script labels them:

  Q1  merchant paid by QR, field blank, predicted legit        -> qr_swap
      (the statement: "blank when not checked"; qr_swap is "a fake QR code at a shop". In October these rows have
      ordinary amounts and sit only in grocery, food, pharmacy and fuel; one is on a device in the qr_swap band.)
  Q2  field filled on a row that is not merchant-QR, predicted legit   -> phishing_collect
      (in training, 21 of the 24 fraud-labelled rows of this kind that the model missed are phishing_collect)

Usage: python -I src/upi/quirk_rules.py <project_root> <in.csv> <out.csv>
"""
import sys
from pathlib import Path

import pandas as pd

ROOT, SRC, DST = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
sub = pd.read_csv(SRC)
idc, labc = sub.columns[0], sub.columns[1]
te = pd.read_csv(ROOT / "data/raw/test.csv").set_index("txn_id").reindex(sub[idc])
merchant_qr = ((te.receiver_type == "MERCHANT") & (te.txn_type == "QR")).values
blank = te.qr_vpa_mismatch.isna().values
legit = (sub[labc] == "legit").values
q1 = merchant_qr & blank
q2 = ~merchant_qr & ~blank
print(f"Q1 merchant-QR rows with a blank field: {int(q1.sum())}, of which predicted legit: {int((q1 & legit).sum())} -> qr_swap")
print(f"Q2 other rows with the field filled: {int(q2.sum())}, of which predicted legit: {int((q2 & legit).sum())} -> phishing_collect")
sub.loc[q1 & legit, labc] = "qr_swap"
sub.loc[q2 & legit, labc] = "phishing_collect"
sub.to_csv(DST, index=False)
print(f"wrote {DST}: {int(((q1 | q2) & legit).sum())} rows changed")
