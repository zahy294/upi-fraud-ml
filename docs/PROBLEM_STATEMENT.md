# UPI Fraud Rings: catch them before the money moves 🕵️‍♀️💸

**ML Gradient Wars 2026 · 48-hour live round**
**13 Nov 2026, 10:00 AM IST → 15 Nov 2026, 10:00 AM IST · Results 15 Nov, 5:00 PM IST**

---

## 1. The story

India makes more than 15 billion UPI payments a month. Almost all of them are ordinary: chai, rent, groceries, a friend paying back their share of dinner.

A tiny fraction are not. Behind them are organised **fraud rings**: people who rent "mule" bank accounts, swap SIM cards, send fake "collect" requests, paste their own QR codes over a shopkeeper's, and pass stolen money through chains of accounts within minutes so it can't be traced.

A single fraudulent payment usually **looks normal on its own**. A ₹2,000 transfer to a new contact is something millions of honest people do every day. Fraud shows itself only in **patterns**: who else paid that account in the last hour, how many people used that phone, whether the money left again ten minutes later, whether the same customer has asked for six refunds this week.

You are the risk team of a payments company. You get three months of history and one month you have never seen. **Label every transaction in that month.**

> All data is synthetic. No real person, account, bank, app or merchant is in it.

---

## 2. The task

Predict one of **12 labels** for every transaction in `test.csv`:

| Label | What it is |
|---|---|
| `legit` | An ordinary payment. About 96% of all transactions |
| `mule_account` | Victims are tricked into paying a "mule" account (rented or bought by a fraud ring), which quickly passes the money on. Every payment into the mule in that burst, and the mule's own onward transfers, carry this label |
| `velocity_bot` | Scripted payments: a lot of small payments from one phone in a few minutes, usually to gaming or e-commerce merchants |
| `layering_chain` | Stolen money hopping through a chain of accounts (A → B → C → …), a few minutes apart, a little less each hop |
| `sim_swap` | The fraudster gets a new SIM for the victim's number, then drains the account, often at night |
| `account_takeover` | The fraudster logs in to the victim's account from a new phone, often far from where the victim lives, and drains it |
| `phishing_collect` | The victim approves a fake payment request ("collect"), usually for an amount just under a round limit |
| `qr_swap` | A fake QR code at a shop: the customer thinks they're paying the shop, but the money goes to someone else |
| `refund_abuse` | One customer claims refunds from online merchants again and again |
| `fake_cashback` | "You've won cashback!": the fraudster sends ₹1–₹10 to look genuine, then asks for a much bigger payment back |
| `loan_app_extortion` | Victims of illegal loan apps pay again and again, late in the evening, under threats |
| `merchant_collusion` | A merchant and a few accounts make fake round-number sales to each other |

You don't have to read these descriptions as rules. **Some fraud looks completely ordinary, and some fraud types look like each other.** That is what makes this hard, and it's how real fraud works.

---

## 3. The data

| File | Rows | What's inside |
|---|---|---|
| `train.csv` | ~1.6 million | Jul–Sep 2026 transactions, with `label` |
| `test.csv` | ~0.26 million | Oct 2026 transactions, no label. Predict these |
| `accounts.csv` | 420,000 | One row per account: home city, bank, KYC level, the date it was opened, phone OS |
| `merchants.csv` | 30,000 | One row per merchant: category, city, onboarding date |
| `sample_submission.csv` | same as test | The upload format |

### Columns in `train.csv` / `test.csv`

| Column | Meaning |
|---|---|
| `txn_id` | Transaction ID. Your submission's `id` |
| `timestamp` | When it happened, IST (`YYYY-MM-DD HH:MM:SS`) |
| `sender_id` | Account that paid (`AC…`). Join with `accounts.csv` |
| `receiver_id` | Who got paid: an account (`AC…`) or a merchant (`MR…`) |
| `receiver_type` | `P2P` (a person's account) or `MERCHANT` |
| `device_id` | The phone the payment came from |
| `amount` | ₹ |
| `txn_type` | `PAY`, `QR` (scanned a QR code), `COLLECT` (approved someone's payment request), `AUTOPAY`, `REFUND` (refund claim) |
| `merchant_category` | For merchant payments |
| `device_city` | Where the phone was at the time |
| `sim_age_hours` | Hours since the SIM in that phone last changed |
| `device_age_days` | Days since this account was first used on this phone |
| `sender_account_age_days` | Age of the sender's account on that day |
| `new_payee` | 1 if the sender had never paid this receiver before |
| `pin_retries` | Wrong UPI PIN entries before this payment |
| `login_to_txn_sec` | Seconds between opening the app and paying |
| `balance_before` | Sender's balance before the payment, ₹ |
| `app`, `app_version` | Which payments app and version |
| `ip_risk_score` | A score from a third-party IP-reputation vendor (0 = safe, 1 = risky) |
| `qr_vpa_mismatch` | For QR payments: 1 if the QR's payee didn't match the shop's registered ID. Blank when not checked |
| `label` | Train only |

### Things you should know

1. **Test is the next month.** October is festival season: people spend more, and the payments app pushed a new version to everyone. Fraud rings adapt too. A model that only memorises September will slip.
2. **Training labels are not perfect.** Like real fraud data, some fraud was never reported (it's labelled `legit`) and some was filed under the wrong type. The test labels are clean.
3. **Not every column is as useful as it looks in training.** Check that a feature still behaves the same way in October before you trust it.
4. **The rows are shuffled.** The patterns are in how rows connect (same account, same phone, same receiver, money moving on minutes later), not in the row order. You may use `test.csv` itself to build those connections. That's allowed and expected.

---

## 4. Scoring

- **Macro-F1** over all 12 labels. Every label counts equally, so `legit` is worth exactly as much as `merchant_collusion`. Predicting `legit` for everything scores about **0.08**.
- The leaderboard during the round uses **30%** of the test set (**Public**). The final ranking scores each team's best public upload on the other **70%** (**Private**), revealed after the round. Don't tune to the public leaderboard.
- **10 uploads per team per day.** The leaderboard refreshes every 15 minutes.
- This is a hard problem on purpose. **A score of 0.40 is excellent.** We don't expect anyone to reach 0.50.

### Submission format

A CSV with a header and exactly one row per `txn_id` in `test.csv`:

```
id,label
TX3fa9c01b,legit
TX0b77e2d4,mule_account
...
```

Labels must be spelled exactly as in the table above. Uploads up to 15 MB.

---

## 5. Rules

- Teams of 1 to 4. One account per person, one team per person.
- Train only on the dataset provided. Any language, open-source library or publicly available pretrained weights are fine.
- **No hand-labelling the test set.** Rules and features you write in code are fine; looking at test rows and typing labels is not.
- The final ranking uses each team's **best public upload**, scored on the private part.
- **Submit your code** (a GitHub repository or notebook link) from the **Submission** tab before the round closes on 15 Nov, 10:00 AM IST. Prize-winning teams must be able to reproduce their best upload from that code.
- Sharing code, predictions or labels between teams is not allowed and leads to disqualification.
- Certificates for every team that uploads a scored submission.

## 6. Prizes

₹2,00,000 for 1st · ₹1,50,000 for 2nd · ₹1,00,000 for 3rd · ₹30,000 for 4th · ₹20,000 for 5th.
Every member of the top 3 teams is referred to The Data Foundation for a 3-month internship. The final offer is the company's decision.

---

## 7. Where to start

Open `starter_notebook.ipynb`. It loads everything, trains a gradient-boosting model on single-row features and writes a valid submission. It scores low on purpose. Then think like a fraud investigator:

- How many **different people** paid this receiver in the last few hours?
- How many **different accounts** used this phone?
- Did the receiver **send the money on** soon after?
- Has this pair of accounts ever paid each other **the other way**?
- Is this amount **just under** a round number?
- Is this phone **new**, in a **different city** from the sender's home, with a **fresh SIM**?

Good luck. Catch them. 🕵️
