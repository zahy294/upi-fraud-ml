# ML Gradient Wars 2026: in-depth analysis

Researched 2026-11-13 (round day 1). Builds on [research.md](research.md). Sources: the official event page, the problem statement in [../PROBLEM_STATEMENT.md](../PROBLEM_STATEMENT.md), Hack4Hack's own pages, and web search. Statements about the organiser that I could not verify are marked as such.

## 1. The facts that matter

| Item | Detail |
|---|---|
| Organiser | Hack4Hack (hack4hack.com), a hackathon-hosting startup. It is also the host of this event, so there is no separate sponsor. |
| Problem | "UPI Fraud Rings": classify every October 2026 UPI transaction into 1 of 12 labels (legit + 11 fraud types). The data is synthetic. |
| Metric | **Macro-F1 over all 12 labels.** `legit` counts exactly as much as each fraud type. All-legit scores about 0.08. The organisers say "0.40 is excellent, nobody reaches 0.50". |
| Round | 48 h live: **13 Nov 10:00 IST → 15 Nov 10:00 IST**. Results 15 Nov, 17:00 IST. |
| Uploads | 10 per team per day, resetting at midnight IST. That is **3 upload days**: 13 Nov (10:00–24:00), 14 Nov, and 15 Nov (00:00–10:00). Up to 30 in total. Files that fail validation don't count. Max 15 MB. |
| Leaderboard | Refreshes every 15 min. Public = 30% of test, private = 70%. |
| Final ranking | **Your *best public* upload is scored on the private split.** You cannot choose your final submission. |
| Code | GitHub repo or notebook link from the Submission tab **before 15 Nov 10:00 IST**. Prize winners must reproduce their best upload. |
| Fee / registration | ₹1,999 per team. Registration was extended from 10 Nov to 13 Nov 10:00 IST. The page showed 284 → 305 teams registered on 12 Nov. |
| Prizes | ₹2L / ₹1.5L / ₹1L / ₹30k / ₹20k. The top 3 teams get an internship *referral* to "The Data Foundation", with the offer at that company's discretion after an interview. |
| Hardware | The organisers say no GPU is needed. The data is 1.8 M rows and about 300 MB, so a 16 GB laptop is enough (see the EDA report). |

## 2. Rules that change how you should play

1. **The best public upload becomes your final.** Every upload is a candidate for the final ranking. An experiment that tops the public board by luck *becomes* your private entry. So:
   - Only upload files you would be happy to be judged on.
   - Public is about 78k rows. Fraud is about 3.7%, which leaves roughly 250 public positives per fraud class. A class F1 swings by a few hundredths from sampling alone, so the macro moves by roughly ±0.01. **Treat public gains under ~0.01 as noise.**
   - Rely on the time-based validation (train Jul–Aug → validate Sep, then refit) to choose between candidates, not on the public board.
2. **Macro-F1 punishes empty classes.** A class you never predict scores 0 and drags the mean down by 1/12 ≈ 0.083. Make sure every class gets *some* predictions, but only where precision is reasonable. The best threshold for each class differs a lot.
3. **Test-time transduction is explicitly allowed.** You may build features from the test rows themselves (bursts, chains, shared devices). Hand-labelling test rows is forbidden. Rules written in code are fine.
4. **The organisers warn about three traps:** time drift (Oct = festival season plus a new app version), noisy training labels, and "columns that look useful in training". The EDA confirms all three and names the trap columns (`app_version`, `ip_risk_score`).
5. **Reproducibility is a prize condition.** Use fixed seeds and one script that goes from raw CSVs to `submission.csv`, committed before the deadline. The `.gitignore` in this repo already keeps `kaggle.json` and the data out of git.

## 3. Upload plan for the 48 hours

| When (IST) | Goal |
|---|---|
| Day 1 (13 Nov), first hours | Upload a safe baseline early, to confirm the format and get on the board. `submissions/rules_v0.csv` (rules only) is ready. Then a LightGBM on row + relational features with traps removed. |
| Day 1 evening → Day 2 | Feature work driven by the EDA: per-type signature features, event/burst propagation, and per-class threshold tuning on September. 3–5 careful uploads per day, keeping the rest in reserve. |
| Day 2 night | Freeze the pipeline. Refit on all of Jul–Sep. Push code to GitHub (private repo, or public only if the rules require it, plus the link). |
| Day 3 (15 Nov) 00:00–10:00 | Fresh quota of 10. Use it only for small, validated changes. **Submit the code link well before 10:00.** |

Lesson carried over from the Amazon ML Challenge post-mortem ([../past_runs_amazon_ml_postmortem.md](../past_runs_amazon_ml_postmortem.md)): that run had a strong architecture but no submission, because infrastructure and over-engineering ate the clock. Here the data fits on the laptop, so **get a valid upload in the first hours and improve from there.**

## 4. Is the event legitimate?

There are no signs it is a scam. Treat it as a young startup's paid contest, not an established competition.

**Points in favour**
- Hack4Hack is a real, active platform with many listed events: AgentX Hackathon 2026, a Crew Hiring Challenge, "ML Challenge", and third-party events such as NovaStark Hacks. It also has a LinkedIn page and steady Instagram marketing.
- It announced a **US$150k seed round from (unnamed) angel investors**. That claim comes from its own news page and was picked up by a funding-news aggregator. I could not verify it independently.
- The dataset and problem statement are high quality: carefully designed drift traps, a clean submission format and sensible rules. That takes real effort.

**Points to be careful about**
- **Entry fees roughly cover the prize pool.** 305 teams × ₹1,999 ≈ ₹6.1L against ₹5L in prizes. Prizes don't depend on a sponsor, but they are funded by entrants.
- **"The Data Foundation" could not be identified.** Searches return only unrelated organisations with that name: a US data-policy nonprofit (datafoundation.org) and a crypto project that rebranded as the "DATA Foundation". The internship is only a *referral*, with the offer at the company's discretion. Don't count on it.
- **No public record of past prize payouts** by Hack4Hack turned up in searches (no winners posts, no participant reviews). That isn't evidence of a problem, just of a short track record.
- Some third-party Instagram reels that promote it use exaggerated hooks ("₹10 Lakhs prize", "GOVERNMENT OPPORTUNITY"). Trust the official page, not the reels.

**Practical advice (the fee is already paid):** keep the payment receipt, registration email, and screenshots of the rules page and your final leaderboard position. Submit the code link before the deadline, because prize eligibility depends on it.

## 5. Similar competitions and what transfers

Lower priority, as requested. These are short pointers, not a literature review.

| Competition / dataset | Similarity | Lesson that transfers here |
|---|---|---|
| **Hack4Hack "ML Challenge"** (same platform, 3–9 Dec 2026, free Round 1, ₹299 Round 2, ₹1L prizes) | Same platform mechanics: macro-F1, 15-min leaderboard, best-public-upload rule, code verification | Not a past event: it's the platform's *next* ML contest. Whatever you build here (pipeline, validation habits) carries straight over. Registration closes 2 Dec. |
| **Your Amazon ML Challenge 2026 run** (entity resolution, 26 M rows) | Same pressure: tight live window, laptop + cloud | Get a baseline first and smoke-test on a sample. Don't migrate infrastructure mid-round. The post-mortem's checklist applies. |
| **Kaggle IEEE-CIS Fraud Detection (2019)** | Tabular payments fraud with time-split train/test and drifting columns | The winning idea was to **reconstruct the entity** (card/user ID) and aggregate features per entity. They also used **adversarial validation** to drop features that drift over time and **time-based validation**. All three apply directly: here the entities are sender, receiver, device and pair. |
| **IBM AMLworld synthetic AML data (NeurIPS 2023)** | Synthetic laundering patterns: fan-in, fan-out, chains/"stack", cycles, scatter-gather | Fan-in ≈ `mule_account`, chain/stack ≈ `layering_chain`. Graph-aware features (or GNNs) beat single-row models, and minority-class F1 is the metric there too. |
| **Elliptic Bitcoin dataset (2019)** | Temporal transaction graph with illicit-vs-licit labels | Its well-known failure mode is that models collapse after a regime change (a dark-market shutdown late in the timeline). That mirrors the "October is different" warning here. |
| **PaySim (Kaggle)** | Synthetic mobile-money fraud (transfer → cash-out chains) | Balance and amount-ratio features plus "money in, then out" patterns. |
| **RBI MuleHunter.AI** (real-world context, India) | Mule-account detection on UPI-like rails | The domain framing for `mule_account`: many unrelated senders paying one account, with fast onward transfers. |

Facts about the external competitions above come from memory and general knowledge. Check the specifics before citing them anywhere formal.
