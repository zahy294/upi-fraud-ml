import unittest
import numpy as np
import pandas as pd
from upi.features import add_prior_label_history, add_row_features

def transactions():
    rows=[]
    for i,(ts,label,kind) in enumerate([
        ("2026-07-10 10:00:00","phishing_collect","COLLECT"),
        ("2026-08-10 10:00:00","legit","COLLECT"),
        ("2026-09-10 10:00:00","legit","PAY")]):
        rows.append(dict(txn_id=f"T{i}",timestamp=pd.Timestamp(ts),sender_id="AC1",receiver_id="AC2",
          receiver_type="P2P",device_id="DV01",amount=999.,txn_type=kind,merchant_category=np.nan,
          device_city="Mumbai",sim_age_hours=1000,device_age_days=100,sender_account_age_days=1000,
          new_payee=1,pin_retries=0,login_to_txn_sec=10,balance_before=5000,app="app_a",
          app_version="4.0",ip_risk_score=.1,qr_vpa_mismatch=np.nan,label=label,split="train"))
    return pd.DataFrame(rows)

class FeaturesTest(unittest.TestCase):
    def setUp(self):
        self.accounts=pd.DataFrame([
          {"account_id":"AC1","home_city":"Mumbai","bank_code":"B1","kyc_level":"full","opened_on":"2020-01-01","os":"Android"},
          {"account_id":"AC2","home_city":"Delhi","bank_code":"B2","kyc_level":"min","opened_on":"2021-01-01","os":"iOS"}])
        self.accounts["opened_on"]=pd.to_datetime(self.accounts.opened_on)
        self.merchants=pd.DataFrame(columns=["merchant_id","category","city","onboarded_on"])
        self.merchants["onboarded_on"]=pd.to_datetime(self.merchants.onboarded_on)
    def test_row_features(self):
        result=add_row_features(transactions(),self.accounts,self.merchants)
        self.assertNotIn("app_version",result); self.assertNotIn("ip_risk_score",result)
        self.assertEqual(result.loc[0,"amt_just_under_1000"],1)
    def test_history_is_strictly_earlier_month(self):
        base=add_row_features(transactions(),self.accounts,self.merchants)
        result=add_prior_label_history(base)
        self.assertEqual(result.loc[0,"hist_receiver_phishing"],0)
        self.assertEqual(result.loc[1,"hist_receiver_phishing"],1)
        changed=base.copy(); changed.loc[1,"label"]="phishing_collect"
        result2=add_prior_label_history(changed)
        self.assertEqual(result2.loc[1,"hist_receiver_phishing"],1)
        self.assertEqual(result2.loc[2,"hist_receiver_phishing"],2)
