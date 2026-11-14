# Kaggle follow-up, 14 November 2026

## Primary sources and transfer limits

Chris Deotte's coauthored first-place IEEE-CIS account emphasizes grouping
transactions by inferred client, engineered group means/variation, and
time-aware validation. Client aggregates helped its models; the raw synthetic
client identifier was excluded to limit memorization. This supports testing
account/payee context, not importing its numerical score into our task.
[Winner's NVIDIA article](https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution/)

The original Kaggle group-aggregation discussion reports that its inferred
clients had homogeneous binary fraud labels. That assumption does **not** hold
here: the local EDA found legitimate payments and multiple fraud types sharing
entities. Consequently, whole-account averaging or propagating one fraud label
through a connected component is inappropriate. We restrict the probe to short
role-specific events and leave the current row out of its supporting evidence.
[Author's Kaggle discussion](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111510)

CPMP's second-place write-up describes independent model/feature development
for diversity and checking raw and frequency-encoded train/test distributions.
Our inference is to compare a distinct detection/type factorization and bounded
entity context, rather than another seed of the same multiclass setup.
[CPMP's Kaggle write-up](https://www.kaggle.com/competitions/ieee-fraud-detection/writeups/2-uncles-and-3-puppies-2nd-solution-cpmp-view)

Kaggle's JavaScript pages did not expose full readable bodies to direct page
opening. The two Kaggle claims above are limited to their indexed passages;
the coauthored NVIDIA article was readable in full. No external training data,
other teams' UPI predictions or test labels were downloaded.

LightGBM4.6 supports binary objectives and `average_precision` evaluation.
The new detector uses that metric for early stopping; final decisions are
still selected using twelve-class macro-F1. Conditional type classification
uses the existing custom balanced log loss.
[Pinned LightGBM documentation](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html)

## What the local experiments actually found

The initial event-propagation probe changed only two selection-window fraud
decisions at the tested stronger settings. The fixed skipped-drain, regular
device timing and repeated-amount probes found little additional support in
July/August. Neither is evidence for a large improvement. The feature candidates
are retained as optional ablations, not claimed discoveries.

The implemented next experiment is described in [HIERARCHY_V9.md](HIERARCHY_V9.md).
It requires the user's local model fitting. Binary fraud detection followed by
conditional type prediction is our dataset-specific hypothesis; the Kaggle
write-ups do not establish that architecture as a winning UPI solution.
Target0.66 remains unverified. Preserve chronological selection and the daily
submission allowance.
