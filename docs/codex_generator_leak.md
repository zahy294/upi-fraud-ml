# Generator-order leak audit — 14 November 2026

New files only: `scripts/explore/41_codex_generator_leak.py` and this report. No fitting or existing-file changes. Numerical threads limited to four.

**Result: no second usable generation-order leak found in the tested families. Zero additional train or test rows have a recovered generation position or phase. There is therefore no new position-based label-purity result to report; counting accidental ID matches as recovered rows would be misleading.**

- Shuffle: 10,021 seeds, three independent shuffle algorithms, three row counts and both directions; 180,378 scores. Largest absolute tail-anchor rank correlation is 0.40714, versus the pre-set 0.85 screening threshold. The exact sparse check uses 4,096 final permutation positions and was verified against actual APIs. It is not a reconstruction of each entire permutation.
- ID streams: 32,421 seeded 4,096-word prefixes; largest overlap is nine IDs, versus 1.74 expected for one arbitrary prefix. After selecting the strongest prefix seed in each broad family, full-length follow-up gives 768/772/769 matches versus 775.17 expected. Only 21/22/22 matched creation-device anchors exist, with correlations −0.309/0.011/0.107: no recovered order.
- MT19937: no exact recurrence matches in file/device order; the unordered necessary-relation test gives 1,685 hits versus 1,698.97 expected, with 1,738 in a matched-size random control. This cannot exclude an MT stream with shuffled, omitted or interleaved outputs. It is not successful state reconstruction.
- Hashes: zero corresponding-row matches in the tested content recipes. Full-domain counter hashes, extra salts/layouts, seeded Faker provider recipes, packed-counter CRC32 inverses and Murmur3 inverses produce no coherent order or excess stream coverage.
- Timestamp, decimal and other numeric structure shows recipe/phase differences, not an ordering key. The exact numeric-RNG probes are deliberately limited to simple scaled/rounded uniforms; arbitrary distributions, affine offsets and skipped RNG calls are not excluded.

All 1,824,642 IDs are unique across both files, unlike unconstrained independent 32-bit draws. This indicates uniqueness enforcement or an injective construction, but orders **zero** rows; it is not a fraud feature. A leaderboard jump alone does not establish that a second leak exists.

**Positive controls passed:** synthetic shuffled monotone device IDs recover rho=1 in both directions for legacy NumPy, Generator and Python shuffles; untempering a known MT19937 stream satisfies all 9,376 tested recurrences. Environment: Python 3.11.0, NumPy 2.4.6, pandas 2.3.3, SciPy 1.16.3, scikit-learn 1.7.2. No models were fitted. The report lists each test family, candidate range, result and important exclusions; it does not claim exhaustive coverage of arbitrary generators.

Created files: this report and [the runnable audit](../scripts/explore/41_codex_generator_leak.py). The code refuses to overwrite a pre-existing report; replay in a fresh output state. Main-sweep timing below excludes the additional provider, inverse-transform and partial-order follow-ups.

## Scope and inputs

Train 1,564,060; test 260,582; unique combined 32-bit IDs 1,824,642. Shuffles use 10021 seeds: 0–10000 inclusive plus [12345, 31337, 54321, 65535, 65536, 123456, 271828, 314159, 8675309, 13112026, 14112026, 15112026, 20260701, 20260801, 20260901, 20261001, 20261113, 20261114, 20261115, 3735928559]. Labels never select shuffle/stream candidates. DV420001–423000 is a reused bot pool, not a creation counter, so monotonicity anchors are DV>423000. Both shuffle directions are checked. Known-device ordering is a sparse constraint, not the original unshuffled row index. Repeated-device rows and interleaved RNG calls weaken it. No row-level stealth classifier is fitted.

## Sparse shuffle validation

| n | seed | api | exact |
| --- | --- | --- | --- |
| 1564060 | 0 | RandomState | True |
| 1564060 | 0 | Generator | True |
| 1564060 | 0 | Python | True |
| 1564060 | 42 | RandomState | True |
| 1564060 | 42 | Generator | True |
| 1564060 | 42 | Python | True |
| 1564060 | 10000 | RandomState | True |
| 1564060 | 10000 | Generator | True |
| 1564060 | 10000 | Python | True |
| 260582 | 0 | RandomState | True |
| 260582 | 0 | Generator | True |
| 260582 | 0 | Python | True |
| 260582 | 42 | RandomState | True |
| 260582 | 42 | Generator | True |
| 260582 | 42 | Python | True |
| 260582 | 10000 | RandomState | True |
| 260582 | 10000 | Generator | True |
| 260582 | 10000 | Python | True |
| 1824642 | 0 | RandomState | True |
| 1824642 | 0 | Generator | True |
| 1824642 | 0 | Python | True |
| 1824642 | 42 | RandomState | True |
| 1824642 | 42 | Generator | True |
| 1824642 | 42 | Python | True |
| 1824642 | 10000 | RandomState | True |
| 1824642 | 10000 | Generator | True |
| 1824642 | 10000 | Python | True |

On these versions pandas.sample(integer random_state) and sklearn.utils.shuffle equal RandomState.permutation; Generator permutation/shuffle/permuted on a 1-D row index are identical. Exact prefix validation covers all three lengths and three seeds. This tests fresh seeded RNGs, not arbitrary prior RNG consumption. The prefix returns the last 4096 final permutation positions exactly, using rejection-sampled Fisher–Yates swaps. No sampling approximation is used for those positions.

## Shuffle sweep results

90,189 seed/API/length candidates; both directions = 180,378 scores. Positive screening criterion fixed before search: |Spearman rho|>0.85 with >=20 creation-device anchors. Strong candidates: 0. This rules out the tested exact fresh-state shuffles when their tail contains the creation-order signal; it does not rule out RNG burn-in, multi-stage shuffling, prior RNG use, or a different concatenation order. Combined-file test assumes raw train followed by raw test.

| file | api | seed | direction | anchors | rho | abs_rho |
| --- | --- | --- | --- | --- | --- | --- |
| train | RandomState | 9468 | file=original[perm] | 127 | 0.364777 | 0.364777 |
| train | RandomState | 1605 | file=original[perm] | 127 | -0.345584 | 0.345584 |
| train | RandomState | 2309 | file=original[perm] | 127 | 0.339368 | 0.339368 |
| train | Generator | 7104 | file=original[perm] | 127 | 0.359363 | 0.359363 |
| train | Generator | 6824 | file[perm]=original | 129 | -0.342805 | 0.342805 |
| train | Generator | 300 | file=original[perm] | 127 | -0.331576 | 0.331576 |
| train | Python | 9833 | file=original[perm] | 127 | -0.397796 | 0.397796 |
| train | Python | 7374 | file=original[perm] | 127 | 0.370981 | 0.370981 |
| train | Python | 954 | file[perm]=original | 139 | -0.36793 | 0.36793 |
| test | RandomState | 1701 | file[perm]=original | 123 | 0.385813 | 0.385813 |
| test | RandomState | 435 | file[perm]=original | 128 | -0.363046 | 0.363046 |
| test | RandomState | 547 | file=original[perm] | 125 | -0.361296 | 0.361296 |
| test | Generator | 2323 | file=original[perm] | 125 | -0.399189 | 0.399189 |
| test | Generator | 5145 | file[perm]=original | 127 | 0.364988 | 0.364988 |
| test | Generator | 1307 | file[perm]=original | 140 | 0.328469 | 0.328469 |
| test | Python | 2419 | file[perm]=original | 121 | 0.40714 | 0.40714 |
| test | Python | 4145 | file[perm]=original | 122 | 0.372661 | 0.372661 |
| test | Python | 4748 | file[perm]=original | 106 | 0.341955 | 0.341955 |
| train+test | RandomState | 3359 | file[perm]=original | 126 | 0.354535 | 0.354535 |
| train+test | RandomState | 6895 | file=original[perm] | 125 | 0.354052 | 0.354052 |
| train+test | RandomState | 2293 | file=original[perm] | 125 | -0.326734 | 0.326734 |
| train+test | Generator | 7263 | file[perm]=original | 122 | -0.335724 | 0.335724 |
| train+test | Generator | 597 | file[perm]=original | 104 | 0.331281 | 0.331281 |
| train+test | Generator | 3121 | file=original[perm] | 125 | 0.320197 | 0.320197 |
| train+test | Python | 9345 | file[perm]=original | 100 | 0.33583 | 0.33583 |
| train+test | Python | 1853 | file[perm]=original | 116 | 0.330696 | 0.330696 |
| train+test | Python | 41 | file[perm]=original | 123 | 0.32861 | 0.32861 |

## Partial and block shuffle

| file | width | offset | blocks | adjacent_pairs | increasing_fraction | fully_monotone_blocks |
| --- | --- | --- | --- | --- | --- | --- |
| train | 32 | 0 | 864 | 2793 | 0.50555 | 59 |
| train | 32 | 16 | 877 | 2807 | 0.497684 | 66 |
| train | 32 | 31 | 857 | 2769 | 0.511376 | 59 |
| train | 64 | 0 | 3323 | 11705 | 0.50551 | 192 |
| train | 64 | 32 | 3366 | 11844 | 0.504222 | 186 |
| train | 64 | 63 | 3318 | 11670 | 0.505656 | 191 |
| train | 128 | 0 | 6833 | 29667 | 0.504163 | 235 |
| train | 128 | 64 | 6856 | 29749 | 0.50358 | 230 |
| train | 128 | 127 | 6848 | 29721 | 0.503886 | 234 |
| train | 256 | 0 | 5859 | 41891 | 0.502638 | 27 |
| train | 256 | 128 | 5858 | 41885 | 0.500418 | 41 |
| train | 256 | 255 | 5855 | 41887 | 0.503307 | 26 |
| train | 512 | 0 | 3054 | 45351 | 0.502018 | 0 |
| train | 512 | 256 | 3054 | 45350 | 0.501808 | 0 |
| train | 512 | 511 | 3054 | 45351 | 0.502172 | 0 |
| train | 1024 | 0 | 1528 | 46880 | 0.501536 | 0 |
| train | 1024 | 512 | 1528 | 46880 | 0.501728 | 0 |
| train | 1024 | 1023 | 1528 | 46880 | 0.501557 | 0 |
| train | 4096 | 0 | 382 | 48026 | 0.501478 | 0 |
| train | 4096 | 2048 | 383 | 48025 | 0.501156 | 0 |
| train | 4096 | 4095 | 382 | 48026 | 0.501478 | 0 |
| train | 16384 | 0 | 96 | 48312 | 0.501408 | 0 |
| train | 16384 | 8192 | 96 | 48312 | 0.501387 | 0 |
| train | 16384 | 16383 | 96 | 48312 | 0.501408 | 0 |
| train | 65536 | 0 | 24 | 48384 | 0.501343 | 0 |
| train | 65536 | 32768 | 25 | 48383 | 0.501333 | 0 |
| train | 65536 | 65535 | 24 | 48384 | 0.501343 | 0 |
| test | 32 | 0 | 131 | 414 | 0.555556 | 9 |
| test | 32 | 16 | 148 | 472 | 0.533898 | 14 |
| test | 32 | 31 | 128 | 405 | 0.550617 | 9 |
| test | 64 | 0 | 550 | 1907 | 0.511274 | 41 |
| test | 64 | 32 | 528 | 1853 | 0.500809 | 32 |
| test | 64 | 63 | 543 | 1883 | 0.51248 | 41 |
| test | 128 | 0 | 1141 | 4932 | 0.503852 | 38 |
| test | 128 | 64 | 1142 | 4894 | 0.502656 | 35 |
| test | 128 | 127 | 1133 | 4906 | 0.505911 | 39 |
| test | 256 | 0 | 976 | 6938 | 0.503747 | 12 |
| test | 256 | 128 | 988 | 6955 | 0.501222 | 5 |
| test | 256 | 255 | 975 | 6936 | 0.505046 | 12 |
| test | 512 | 0 | 509 | 7517 | 0.501929 | 0 |
| test | 512 | 256 | 510 | 7516 | 0.503991 | 0 |
| test | 512 | 511 | 509 | 7517 | 0.502328 | 0 |
| test | 1024 | 0 | 255 | 7771 | 0.501222 | 0 |
| test | 1024 | 512 | 255 | 7771 | 0.503153 | 0 |
| test | 1024 | 1023 | 255 | 7771 | 0.501222 | 0 |
| test | 4096 | 0 | 64 | 7962 | 0.502638 | 0 |
| test | 4096 | 2048 | 65 | 7961 | 0.502324 | 0 |
| test | 4096 | 4095 | 64 | 7962 | 0.502638 | 0 |
| test | 16384 | 0 | 16 | 8010 | 0.502372 | 0 |
| test | 16384 | 8192 | 17 | 8009 | 0.502684 | 0 |
| test | 16384 | 16383 | 16 | 8010 | 0.502372 | 0 |
| test | 65536 | 0 | 4 | 8022 | 0.502493 | 0 |
| test | 65536 | 32768 | 5 | 8021 | 0.502556 | 0 |
| test | 65536 | 65535 | 4 | 8022 | 0.502493 | 0 |

Offsets 0, half-block and last position are tested. Fully monotone tiny blocks can occur by chance (four anchors: probability 2/4!); counts are descriptive, not automatically positives. No labels select widths.

## Seeded transaction-ID stream tests

Full results summarized per API below: 32,421 candidate streams, each first 4096 values. Most hits at any seed are shown; expected chance uses the observed ID-set size. Threshold >=max(30,8×chance). Candidates: []. Faker is not installed: its general providers were NOT run; seeded stdlib UUID4 first/last-eight recipes were tested as explicit proxies. No package was installed. Prefix negatives do not reject large burn-in, salted encoding, another Faker provider/version, or an RNG mixed with other field draws.

| api | seed | draws | hits | chance | scope |
| --- | --- | --- | --- | --- | --- |
| Python getrandbits32 | 2945 | 4096 | 9 | 1.74011 | all seeds |
| Generator uint32 [0,2^32) | 3822 | 4096 | 9 | 1.74011 | all seeds |
| RandomState uint32 [0,2^32) | 4266 | 4096 | 8 | 1.74011 | all seeds |
| Generator int64 upper=4294967296 | 123 | 4096 | 8 | 1.74011 | expanded constants/0-100 |
| Generator uint64 upper=4294967296 | 123 | 4096 | 8 | 1.74011 | expanded constants/0-100 |
| Generator uint64 upper=2147483648 | 15112026 | 4096 | 7 | 1.74011 | expanded constants/0-100 |
| Generator int64 upper=2147483648 | 15112026 | 4096 | 7 | 1.74011 | expanded constants/0-100 |
| Generator int64 upper=4294967295 | 20261113 | 4096 | 7 | 1.74011 | expanded constants/0-100 |
| Generator uint64 upper=4294967295 | 20261113 | 4096 | 7 | 1.74011 | expanded constants/0-100 |
| RandomState int64 upper=4294967296 | 82 | 4096 | 6 | 1.74011 | expanded constants/0-100 |
| RandomState int64 upper=4294967295 | 82 | 4096 | 6 | 1.74011 | expanded constants/0-100 |
| RandomState uint64 upper=4294967296 | 20260701 | 4096 | 6 | 1.74011 | expanded constants/0-100 |
| RandomState uint64 upper=4294967295 | 20260701 | 4096 | 6 | 1.74011 | expanded constants/0-100 |
| Python choices hex8 | 54 | 4096 | 6 | 1.74011 | expanded constants/0-100 |
| Generator int32 upper=2147483647 | 15112026 | 4096 | 6 | 1.74011 | expanded constants/0-100 |
| seeded UUID4 last8 | 999 | 4096 | 6 | 1.74011 | Faker uuid4 recipe proxy; package unavailable |
| RandomState int64 upper=2147483648 | 20261113 | 4096 | 5 | 1.74011 | expanded constants/0-100 |
| RandomState uint64 upper=2147483648 | 9 | 4096 | 5 | 1.74011 | expanded constants/0-100 |
| RandomState int32 upper=2147483647 | 9 | 4096 | 5 | 1.74011 | expanded constants/0-100 |
| Python randrange32 | 12 | 4096 | 5 | 1.74011 | expanded constants/0-100 |
| seeded UUID4 first8 | 33 | 4096 | 5 | 1.74011 | Faker uuid4 recipe proxy; package unavailable |

## Seed-free MT19937

| file | order | recurrences | matches | chance |
| --- | --- | --- | --- | --- |
| train | file | 1563436 | 0 | 0.000364016 |
| train | device | 47784 | 0 | 1.11256e-05 |
| test | file | 259958 | 0 | 6.05262e-05 |
| test | device | 7402 | 0 | 1.72341e-06 |

| kind | sampled_pairs | necessary_relation_hits | independence_expectation |
| --- | --- | --- | --- |
| untempered observed ID set | 2000000 | 1685 | 1698.97 |
| matched-size uniform null | 2000000 | 1738 | 1698.63 |

Inverting the twist of a sampled pair XOR tests whether its required next-state low 31 bits occurs in the ID set. This is only a necessary unordered-set relation; it omits unknown lag positions and the predecessor high bit. With millions of words the set is dense enough for many accidental completions. A negative/near-null result does NOT exclude MT19937 under shuffling and missing/interleaved draws. Device-order recurrence assumes consecutive outputs per observed device, which is stronger than known creation order. File-order recurrence assumes no shuffle. Neither assumption is established. No state recovery claim is made.

## Hashes of row content

2016 fixed recipes tested on 1024 evenly spaced TRAIN rows, with exact per-row ID equality; no labels used. All 24 orders of sender/receiver/timestamp/amount, six separators, raw or two-decimal amount, md5/sha1/sha256 first/last eight and crc32. Maximum matches: 0. This does not cover arbitrary salts, JSON serialization or other field subsets.

| fields | separator | amount | hash | slice | rows | matches |
| --- | --- | --- | --- | --- | --- | --- |
| sender_id+receiver_id+timestamp+amount | '' | raw | md5 | first | 1024 | 0 |
| sender_id+receiver_id+timestamp+amount | '' | raw | md5 | last | 1024 | 0 |
| sender_id+receiver_id+timestamp+amount | '' | raw | sha1 | first | 1024 | 0 |
| sender_id+receiver_id+timestamp+amount | '' | raw | sha1 | last | 1024 | 0 |
| sender_id+receiver_id+timestamp+amount | '' | raw | sha256 | first | 1024 | 0 |
| sender_id+receiver_id+timestamp+amount | '' | raw | sha256 | last | 1024 | 0 |
| sender_id+receiver_id+timestamp+amount | '' | raw | crc32 | first | 1024 | 0 |
| sender_id+receiver_id+timestamp+amount | '' | fixed2 | md5 | first | 1024 | 0 |

## Counter hashes with prefixes/salts

| salt | hash | counters | hits | chance | first_hit_counters |
| --- | --- | --- | --- | --- | --- |
| '' | md5 | 1824643 | 742 | 775.168 | [2634, 4058, 4127, 4437, 5772] |
| '' | sha1 | 1824643 | 763 | 775.168 | [2215, 2919, 9942, 14048, 18814] |
| '' | sha256 | 1824643 | 818 | 775.168 | [2814, 9449, 11696, 12649, 20798] |
| '' | crc32 | 1824643 | 802 | 775.168 | [5235, 5557, 6722, 8354, 20386] |
| 'TX' | md5 | 1824643 | 809 | 775.168 | [2288, 3217, 4424, 6718, 7990] |
| 'TX' | sha1 | 1824643 | 805 | 775.168 | [1812, 2052, 4088, 7243, 20314] |
| 'TX' | sha256 | 1824643 | 809 | 775.168 | [45, 451, 1225, 1387, 6620] |
| 'TX' | crc32 | 1824643 | 782 | 775.168 | [2672, 4772, 5743, 6199, 8591] |
| 'txn' | md5 | 1824643 | 767 | 775.168 | [2511, 4041, 11034, 14678, 17642] |
| 'txn' | sha1 | 1824643 | 794 | 775.168 | [5749, 7329, 9088, 10330, 12730] |
| 'txn' | sha256 | 1824643 | 765 | 775.168 | [516, 852, 3231, 3905, 19820] |
| 'txn' | crc32 | 1824643 | 796 | 775.168 | [12, 405, 7674, 9582, 10079] |
| 'transaction' | md5 | 1824643 | 804 | 775.168 | [4603, 6505, 6747, 8283, 8976] |
| 'transaction' | sha1 | 1824643 | 796 | 775.168 | [2487, 3988, 6006, 9969, 13462] |
| 'transaction' | sha256 | 1824643 | 765 | 775.168 | [418, 597, 1963, 7343, 12471] |
| 'transaction' | crc32 | 1824643 | 791 | 775.168 | [6895, 7101, 13841, 15307, 18542] |
| 'train' | md5 | 1824643 | 757 | 775.168 | [471, 1764, 6431, 9087, 10828] |
| 'train' | sha1 | 1824643 | 759 | 775.168 | [150, 3003, 3615, 4035, 5071] |
| 'train' | sha256 | 1824643 | 753 | 775.168 | [87, 143, 674, 1412, 1462] |
| 'train' | crc32 | 1824643 | 778 | 775.168 | [3523, 4864, 7826, 8520, 12736] |
| 'test' | md5 | 1824643 | 781 | 775.168 | [813, 1158, 6706, 8853, 9746] |
| 'test' | sha1 | 1824643 | 802 | 775.168 | [4211, 11691, 13972, 14134, 16853] |
| 'test' | sha256 | 1824643 | 725 | 775.168 | [4435, 4747, 7055, 7485, 12587] |
| 'test' | crc32 | 1824643 | 797 | 775.168 | [3503, 4774, 6218, 9589, 11302] |
| '42' | md5 | 1824643 | 751 | 775.168 | [546, 2576, 23047, 31347, 31694] |
| '42' | sha1 | 1824643 | 798 | 775.168 | [724, 4462, 4576, 4791, 5190] |
| '42' | sha256 | 1824643 | 754 | 775.168 | [909, 2735, 3572, 5220, 14802] |
| '42' | crc32 | 1824643 | 724 | 775.168 | [1505, 7164, 9011, 12307, 20016] |
| '2026' | md5 | 1824643 | 771 | 775.168 | [1204, 2934, 7602, 11636, 11936] |
| '2026' | sha1 | 1824643 | 763 | 775.168 | [2054, 5199, 9278, 15538, 16431] |
| '2026' | sha256 | 1824643 | 789 | 775.168 | [1466, 4916, 6934, 9442, 10339] |
| '2026' | crc32 | 1824643 | 779 | 775.168 | [458, 657, 6511, 18951, 20428] |
| 'upi' | md5 | 1824643 | 824 | 775.168 | [902, 3606, 10523, 10701, 13283] |
| 'upi' | sha1 | 1824643 | 749 | 775.168 | [308, 1936, 2346, 2521, 5522] |
| 'upi' | sha256 | 1824643 | 759 | 775.168 | [6113, 6699, 8030, 8608, 9977] |
| 'upi' | crc32 | 1824643 | 808 | 775.168 | [103, 4637, 5187, 8472, 9068] |

Each recipe scans counters 0 through train+test length inclusive, covering both per-file lengths and the combined length. Prefix is literal, decimal counter unpadded, digest first eight. Chance collisions are not evidence of recovered rows. Suffix salts, nondecimal encodings, padding and arbitrary offsets are outside this bounded family.

## Sequential numeric and timestamp structure

| file | column | transform | anchors | rho_device | lag1 | unique |
| --- | --- | --- | --- | --- | --- | --- |
| train | amount | value | 48408 | -0.0675134 | 0.129958 | 32675 |
| train | amount | fraction | 48408 | 0.0843014 | 0.0419377 | 100 |
| train | balance_before | value | 48408 | 0.0378483 | 0.0114964 | 47690 |
| train | balance_before | fraction | 48408 | 0.00422119 | -0.002653 | 100 |
| train | sim_age_hours | value | 48408 | -0.00273665 | -0.00181117 | 42568 |
| train | sim_age_hours | fraction | 48408 | 0.00409991 | -0.0064901 | 10 |
| train | device_age_days | value | 48408 | 0.083524 | 0.0694646 | 301 |
| train | device_age_days | fraction | 48408 | 0.0115652 | 0.00470311 | 10 |
| train | second | value | 48408 | -0.00101148 | -0.00217808 | 60 |
| train | minute | value | 48408 | 0.0038059 | 0.000717252 | 60 |
| train | seconds_of_day | value | 48408 | -0.00346052 | 0.0253979 | 33863 |
| test | amount | value | 8026 | -0.0769809 | 0.137626 | 6192 |
| test | amount | fraction | 8026 | 0.0900954 | 0.0428553 | 100 |
| test | balance_before | value | 8026 | 0.0100193 | 0.0104905 | 8009 |
| test | balance_before | fraction | 8026 | -0.00697495 | -0.00426072 | 100 |
| test | sim_age_hours | value | 8026 | 0.0116368 | 0.00859045 | 7845 |
| test | sim_age_hours | fraction | 8026 | 0.00687816 | -0.0158783 | 10 |
| test | device_age_days | value | 8026 | 0.089908 | 0.0614082 | 301 |
| test | device_age_days | fraction | 8026 | 0.00923572 | 0.00365112 | 10 |
| test | second | value | 8026 | -0.017897 | 0.0178504 | 60 |
| test | minute | value | 8026 | 0.00246745 | 0.00260658 | 60 |
| test | seconds_of_day | value | 8026 | 0.0038778 | 0.0197838 | 7530 |

| file | phase | n | mean_second | mean_minute | sim_fraction_mean | device_fraction_mean | balance_fraction_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | 45177 | 29.4536 | 29.6441 | 0.449771 | 0.45065 | 0.495319 |
| train | 1 | 349 | 28.2149 | 30.1318 | 0.429513 | 0.432665 | 0.512779 |
| train | 2 | 175 | 29.2743 | 27.5429 | 0.421714 | 0.434286 | 0.510171 |
| train | 3 | 213 | 28.8685 | 30.4695 | 0.411737 | 0.442254 | 0.515023 |
| train | 4 | 570 | 29.314 | 29.9175 | 0.432105 | 0.408596 | 0.493158 |
| train | 5 | 1165 | 30.4824 | 28.9923 | 0.442918 | 0.383348 | 0.491099 |
| train | 6 | 225 | 29.7467 | 29.1022 | 0.42 | 0.433778 | 0.489511 |
| train | 7 | 230 | 30.5261 | 29.487 | 0.451739 | 0.428261 | 0.520304 |
| train | 8 | 108 | 32.0926 | 23.7593 | 0.480556 | 0.425926 | 0.511481 |
| train | 9 | 75 | 27.6933 | 31.8267 | 0.434667 | 0.404 | 0.519733 |
| train | 10 | 51 | 31.9608 | 26.6275 | 0.462745 | 0.407843 | 0.495294 |
| train | 11 | 70 | 32.0429 | 30.5429 | 0.481429 | 0.428571 | 0.478 |
| test | 0 | 7462 | 29.2124 | 29.5568 | 0.451715 | 0.456861 | 0.497692 |
| test | 1 | 80 | 29.1875 | 28.9 | 0.44625 | 0.365 | 0.499 |
| test | 2 | 22 | 33.4545 | 34.0455 | 0.436364 | 0.368182 | 0.575 |
| test | 3 | 46 | 31.6522 | 31.6304 | 0.417391 | 0.384783 | 0.546739 |
| test | 4 | 61 | 33.1311 | 28.1148 | 0.532787 | 0.381967 | 0.50459 |
| test | 5 | 218 | 29.633 | 29.2752 | 0.421101 | 0.387156 | 0.485 |
| test | 6 | 43 | 27.5581 | 31.6047 | 0.437209 | 0.469767 | 0.456512 |
| test | 7 | 40 | 29.675 | 24.625 | 0.465 | 0.52 | 0.47075 |
| test | 8 | 23 | 29.7826 | 32.3043 | 0.395652 | 0.430435 | 0.382174 |
| test | 9 | 10 | 27.2 | 27 | 0.46 | 0.45 | 0.663 |
| test | 10 | 8 | 25.875 | 29.625 | 0.5375 | 0.3625 | 0.43625 |
| test | 11 | 13 | 28.3077 | 39.9231 | 0.215385 | 0.592308 | 0.586923 |

Phase 0 here means ordinary newly created devices; phases 1–11 use existing band edges. Shape-related differences in ages, timestamps, whole amounts or balances are not a second generation-order leak. These statistics do not recover home-device order. Fractions and age ranges are often rounded or recipe-specific, so matching coarse decimals alone is uninformative.

## Seeded numeric-sequence probes

| file | column | scale | seeds | apis | anchors | max_equal | best_seed | best_api |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | sim_age_hours | 1 | 131 | 2 | 256 | 0 | 0 | RandomState |
| train | sim_age_hours | 10 | 131 | 2 | 256 | 0 | 0 | RandomState |
| train | sim_age_hours | 100 | 131 | 2 | 256 | 0 | 0 | RandomState |
| train | sim_age_hours | 1000 | 131 | 2 | 256 | 0 | 0 | RandomState |
| train | sim_age_hours | 10000 | 131 | 2 | 256 | 0 | 0 | RandomState |
| train | sim_age_hours | 100000 | 131 | 2 | 256 | 0 | 0 | RandomState |
| train | device_age_days | 1 | 131 | 2 | 256 | 4 | 92 | Generator |
| train | device_age_days | 10 | 131 | 2 | 256 | 5 | 4 | Generator |
| train | device_age_days | 100 | 131 | 2 | 256 | 2 | 12 | Generator |
| train | device_age_days | 1000 | 131 | 2 | 256 | 1 | 5 | RandomState |
| train | device_age_days | 10000 | 131 | 2 | 256 | 0 | 0 | RandomState |
| train | device_age_days | 100000 | 131 | 2 | 256 | 0 | 0 | RandomState |
| train | balance_before | 1 | 131 | 2 | 256 | 0 | 0 | RandomState |
| train | balance_before | 10 | 131 | 2 | 256 | 0 | 0 | RandomState |
| train | balance_before | 100 | 131 | 2 | 256 | 0 | 0 | RandomState |
| train | balance_before | 1000 | 131 | 2 | 256 | 0 | 0 | RandomState |
| train | balance_before | 10000 | 131 | 2 | 256 | 0 | 0 | RandomState |
| train | balance_before | 100000 | 131 | 2 | 256 | 0 | 0 | RandomState |
| test | sim_age_hours | 1 | 131 | 2 | 256 | 0 | 0 | RandomState |
| test | sim_age_hours | 10 | 131 | 2 | 256 | 1 | 16 | RandomState |
| test | sim_age_hours | 100 | 131 | 2 | 256 | 1 | 14 | Generator |
| test | sim_age_hours | 1000 | 131 | 2 | 256 | 1 | 29 | RandomState |
| test | sim_age_hours | 10000 | 131 | 2 | 256 | 0 | 0 | RandomState |
| test | sim_age_hours | 100000 | 131 | 2 | 256 | 0 | 0 | RandomState |
| test | device_age_days | 1 | 131 | 2 | 256 | 15 | 88 | Generator |
| test | device_age_days | 10 | 131 | 2 | 256 | 6 | 53 | Generator |
| test | device_age_days | 100 | 131 | 2 | 256 | 2 | 6 | RandomState |
| test | device_age_days | 1000 | 131 | 2 | 256 | 1 | 30 | RandomState |
| test | device_age_days | 10000 | 131 | 2 | 256 | 1 | 314159 | Generator |
| test | device_age_days | 100000 | 131 | 2 | 256 | 0 | 0 | RandomState |
| test | balance_before | 1 | 131 | 2 | 256 | 0 | 0 | RandomState |
| test | balance_before | 10 | 131 | 2 | 256 | 0 | 0 | RandomState |
| test | balance_before | 100 | 131 | 2 | 256 | 0 | 0 | RandomState |
| test | balance_before | 1000 | 131 | 2 | 256 | 0 | 0 | RandomState |
| test | balance_before | 10000 | 131 | 2 | 256 | 0 | 0 | RandomState |
| test | balance_before | 100000 | 131 | 2 | 256 | 0 | 0 | RandomState |

Fresh-state RandomState/default_rng uniform draws scaled by listed constants and rounded to the CSV precision, matched in order to the first 256 creation-device rows. Tests exclude arbitrary affine offsets, nonuniform distributions and unknown skipped draws. Device creation does not imply consecutive calls to any one field RNG. A negative here rejects only this aligned simple-uniform construction, not seeded generation in general.

## Positives and limitations

No candidates.

A leaderboard jump is not evidence identifying a generator leak. This audit tests explicit finite families. No test labels are read. There is no defensible label-purity-by-recovered-position claim unless a candidate actually recovers order beyond known bands. The pre-existing DV leak is used only as the anchor. Full results are summarized per family with exact seed ranges, lengths, thresholds and maximums; candidate ties at zero do not imply a preferred seed. Faker providers, unknown RNG burn-in/interleaving, arbitrary salts and shuffle composition remain untested.

Runtime: 509.3 seconds. Run `C:/Users/ASUS/AppData/Local/Programs/Python/Python311/python.exe -I scripts/explore/41_codex_generator_leak.py` in a fresh output state; the script refuses to overwrite this report.
## Additional ID structure, provider and hash tests

Combined distinct IDs: 1824642 out of 1824642; observed within-file duplicates and cross-file overlap are zero. Independent unconstrained 32-bit draws would have approximately 387.6 colliding pairs. This is evidence of an ID-uniqueness mechanism or injective construction, not recovered generation order: transaction IDs are required to be unique and rejection sampling suffices.

| test | ids | successor_in_set | chance |
| --- | --- | --- | --- |
| NumericalRecipes | 1824642 | 749 | 775.1673525581136 |
| ANSI_C | 1824642 | 789 | 775.1673525581136 |
| MSVC | 1824642 | 730 | 775.1673525581136 |
| Borland | 1824642 | 742 | 775.1673525581136 |
| identity_increment | 1824642 | 809 | 775.1673525581136 |
| xorshift32_13_17_5 | 1824642 | 814 | 775.1673525581136 |

These are seed-free necessary tests for a consecutive full-state stream. Interleaving/missing outputs can break adjacency; ordinary chance successor hits do not identify a seed.

| file | inverse | rows | decoded_below_combined_N | anchor_rho |
| --- | --- | --- | --- | --- |
| train | CRC32 little endian counter | 1564060 | 683 | 0.000312510427075962 |
| train | CRC32 big endian counter | 1564060 | 702 | 0.0009192494974744928 |
| train | Murmur3 fmix32 inverse | 1564060 | 682 | 0.011527229071267993 |
| test | CRC32 little endian counter | 260582 | 129 | -0.015621783985658482 |
| test | CRC32 big endian counter | 260582 | 116 | 0.01391683464351748 |
| test | Murmur3 fmix32 inverse | 260582 | 119 | 0.003013853149411976 |

Inverse transforms map every ID to some integer by construction. Only compact counter support or monotone creation-device ranks would indicate recovered order; universal invertibility itself is not a positive.

| provider | part | seeds | draws | max_hits | best_seed | chance |
| --- | --- | --- | --- | --- | --- | --- |
| md5 | first8 | 121 | 4096 | 7 | 29 | 1.7401142120361328 |
| md5 | last8 | 121 | 4096 | 7 | 63 | 1.7401142120361328 |
| sha1 | first8 | 121 | 4096 | 5 | 40 | 1.7401142120361328 |
| sha1 | last8 | 121 | 4096 | 5 | 68 | 1.7401142120361328 |
| sha256 | first8 | 121 | 4096 | 5 | 34 | 1.7401142120361328 |
| sha256 | last8 | 121 | 4096 | 6 | 48 | 1.7401142120361328 |

Faker is unavailable locally. The tested hash-provider recipes are reproduced directly from [Faker misc provider source](https://raw.githubusercontent.com/joke2k/faker/master/faker/providers/misc/__init__.py): md5/sha1/sha256 hash the string form of random.random(); uuid4 wraps random.getrandbits(128), already tested above. No claim is made to cover all providers, versions or earlier shared RNG consumption.

| hash | salt | position | counters | first8_hits | last8_hits | chance |
| --- | --- | --- | --- | --- | --- | --- |
| md5 | '42:' | prefix | 65536 | 28 | 24 | 27.841827392578125 |
| md5 | '42:' | suffix | 65536 | 25 | 33 | 27.841827392578125 |
| md5 | '2026:' | prefix | 65536 | 29 | 35 | 27.841827392578125 |
| md5 | '2026:' | suffix | 65536 | 26 | 18 | 27.841827392578125 |
| md5 | '123:' | prefix | 65536 | 34 | 29 | 27.841827392578125 |
| md5 | '123:' | suffix | 65536 | 31 | 19 | 27.841827392578125 |
| md5 | '1337:' | prefix | 65536 | 31 | 34 | 27.841827392578125 |
| md5 | '1337:' | suffix | 65536 | 20 | 26 | 27.841827392578125 |
| md5 | 'train:' | prefix | 65536 | 30 | 33 | 27.841827392578125 |
| md5 | 'train:' | suffix | 65536 | 33 | 30 | 27.841827392578125 |
| md5 | 'test:' | prefix | 65536 | 19 | 35 | 27.841827392578125 |
| md5 | 'test:' | suffix | 65536 | 29 | 34 | 27.841827392578125 |
| md5 | 'TX_' | prefix | 65536 | 28 | 25 | 27.841827392578125 |
| md5 | 'TX_' | suffix | 65536 | 38 | 26 | 27.841827392578125 |
| md5 | 'txn_' | prefix | 65536 | 19 | 31 | 27.841827392578125 |
| md5 | 'txn_' | suffix | 65536 | 23 | 19 | 27.841827392578125 |
| md5 | 'transaction_' | prefix | 65536 | 19 | 37 | 27.841827392578125 |
| md5 | 'transaction_' | suffix | 65536 | 23 | 40 | 27.841827392578125 |
| md5 | 'upi:' | prefix | 65536 | 29 | 30 | 27.841827392578125 |
| md5 | 'upi:' | suffix | 65536 | 32 | 33 | 27.841827392578125 |
| sha1 | '42:' | prefix | 65536 | 33 | 25 | 27.841827392578125 |
| sha1 | '42:' | suffix | 65536 | 28 | 35 | 27.841827392578125 |
| sha1 | '2026:' | prefix | 65536 | 28 | 37 | 27.841827392578125 |
| sha1 | '2026:' | suffix | 65536 | 26 | 23 | 27.841827392578125 |
| sha1 | '123:' | prefix | 65536 | 27 | 28 | 27.841827392578125 |
| sha1 | '123:' | suffix | 65536 | 33 | 27 | 27.841827392578125 |
| sha1 | '1337:' | prefix | 65536 | 31 | 30 | 27.841827392578125 |
| sha1 | '1337:' | suffix | 65536 | 23 | 36 | 27.841827392578125 |
| sha1 | 'train:' | prefix | 65536 | 38 | 25 | 27.841827392578125 |
| sha1 | 'train:' | suffix | 65536 | 30 | 32 | 27.841827392578125 |
| sha1 | 'test:' | prefix | 65536 | 33 | 28 | 27.841827392578125 |
| sha1 | 'test:' | suffix | 65536 | 23 | 33 | 27.841827392578125 |
| sha1 | 'TX_' | prefix | 65536 | 33 | 31 | 27.841827392578125 |
| sha1 | 'TX_' | suffix | 65536 | 18 | 36 | 27.841827392578125 |
| sha1 | 'txn_' | prefix | 65536 | 32 | 25 | 27.841827392578125 |
| sha1 | 'txn_' | suffix | 65536 | 26 | 33 | 27.841827392578125 |
| sha1 | 'transaction_' | prefix | 65536 | 24 | 25 | 27.841827392578125 |
| sha1 | 'transaction_' | suffix | 65536 | 27 | 28 | 27.841827392578125 |
| sha1 | 'upi:' | prefix | 65536 | 19 | 30 | 27.841827392578125 |
| sha1 | 'upi:' | suffix | 65536 | 21 | 34 | 27.841827392578125 |
| sha256 | '42:' | prefix | 65536 | 24 | 29 | 27.841827392578125 |
| sha256 | '42:' | suffix | 65536 | 24 | 25 | 27.841827392578125 |
| sha256 | '2026:' | prefix | 65536 | 23 | 22 | 27.841827392578125 |
| sha256 | '2026:' | suffix | 65536 | 42 | 27 | 27.841827392578125 |
| sha256 | '123:' | prefix | 65536 | 31 | 28 | 27.841827392578125 |
| sha256 | '123:' | suffix | 65536 | 32 | 25 | 27.841827392578125 |
| sha256 | '1337:' | prefix | 65536 | 21 | 22 | 27.841827392578125 |
| sha256 | '1337:' | suffix | 65536 | 24 | 34 | 27.841827392578125 |
| sha256 | 'train:' | prefix | 65536 | 31 | 30 | 27.841827392578125 |
| sha256 | 'train:' | suffix | 65536 | 28 | 39 | 27.841827392578125 |
| sha256 | 'test:' | prefix | 65536 | 32 | 25 | 27.841827392578125 |
| sha256 | 'test:' | suffix | 65536 | 29 | 29 | 27.841827392578125 |
| sha256 | 'TX_' | prefix | 65536 | 29 | 18 | 27.841827392578125 |
| sha256 | 'TX_' | suffix | 65536 | 24 | 27 | 27.841827392578125 |
| sha256 | 'txn_' | prefix | 65536 | 26 | 37 | 27.841827392578125 |
| sha256 | 'txn_' | suffix | 65536 | 26 | 30 | 27.841827392578125 |
| sha256 | 'transaction_' | prefix | 65536 | 33 | 27 | 27.841827392578125 |
| sha256 | 'transaction_' | suffix | 65536 | 29 | 44 | 27.841827392578125 |
| sha256 | 'upi:' | prefix | 65536 | 23 | 32 | 27.841827392578125 |
| sha256 | 'upi:' | suffix | 65536 | 26 | 35 | 27.841827392578125 |
| blake2s | '42:' | prefix | 65536 | 27 | 28 | 27.841827392578125 |
| blake2s | '42:' | suffix | 65536 | 31 | 23 | 27.841827392578125 |
| blake2s | '2026:' | prefix | 65536 | 28 | 33 | 27.841827392578125 |
| blake2s | '2026:' | suffix | 65536 | 33 | 25 | 27.841827392578125 |
| blake2s | '123:' | prefix | 65536 | 23 | 35 | 27.841827392578125 |
| blake2s | '123:' | suffix | 65536 | 33 | 24 | 27.841827392578125 |
| blake2s | '1337:' | prefix | 65536 | 32 | 34 | 27.841827392578125 |
| blake2s | '1337:' | suffix | 65536 | 31 | 28 | 27.841827392578125 |
| blake2s | 'train:' | prefix | 65536 | 24 | 28 | 27.841827392578125 |
| blake2s | 'train:' | suffix | 65536 | 37 | 16 | 27.841827392578125 |
| blake2s | 'test:' | prefix | 65536 | 25 | 30 | 27.841827392578125 |
| blake2s | 'test:' | suffix | 65536 | 31 | 34 | 27.841827392578125 |
| blake2s | 'TX_' | prefix | 65536 | 32 | 33 | 27.841827392578125 |
| blake2s | 'TX_' | suffix | 65536 | 31 | 26 | 27.841827392578125 |
| blake2s | 'txn_' | prefix | 65536 | 30 | 35 | 27.841827392578125 |
| blake2s | 'txn_' | suffix | 65536 | 32 | 25 | 27.841827392578125 |
| blake2s | 'transaction_' | prefix | 65536 | 21 | 30 | 27.841827392578125 |
| blake2s | 'transaction_' | suffix | 65536 | 22 | 23 | 27.841827392578125 |
| blake2s | 'upi:' | prefix | 65536 | 32 | 28 | 27.841827392578125 |
| blake2s | 'upi:' | suffix | 65536 | 27 | 17 | 27.841827392578125 |
| blake2b | '42:' | prefix | 65536 | 29 | 28 | 27.841827392578125 |
| blake2b | '42:' | suffix | 65536 | 27 | 26 | 27.841827392578125 |
| blake2b | '2026:' | prefix | 65536 | 32 | 22 | 27.841827392578125 |
| blake2b | '2026:' | suffix | 65536 | 24 | 26 | 27.841827392578125 |
| blake2b | '123:' | prefix | 65536 | 27 | 27 | 27.841827392578125 |
| blake2b | '123:' | suffix | 65536 | 32 | 29 | 27.841827392578125 |
| blake2b | '1337:' | prefix | 65536 | 26 | 39 | 27.841827392578125 |
| blake2b | '1337:' | suffix | 65536 | 26 | 35 | 27.841827392578125 |
| blake2b | 'train:' | prefix | 65536 | 33 | 26 | 27.841827392578125 |
| blake2b | 'train:' | suffix | 65536 | 22 | 21 | 27.841827392578125 |
| blake2b | 'test:' | prefix | 65536 | 25 | 25 | 27.841827392578125 |
| blake2b | 'test:' | suffix | 65536 | 20 | 30 | 27.841827392578125 |
| blake2b | 'TX_' | prefix | 65536 | 26 | 29 | 27.841827392578125 |
| blake2b | 'TX_' | suffix | 65536 | 27 | 23 | 27.841827392578125 |
| blake2b | 'txn_' | prefix | 65536 | 32 | 29 | 27.841827392578125 |
| blake2b | 'txn_' | suffix | 65536 | 37 | 23 | 27.841827392578125 |
| blake2b | 'transaction_' | prefix | 65536 | 21 | 29 | 27.841827392578125 |
| blake2b | 'transaction_' | suffix | 65536 | 27 | 21 | 27.841827392578125 |
| blake2b | 'upi:' | prefix | 65536 | 27 | 23 | 27.841827392578125 |
| blake2b | 'upi:' | suffix | 65536 | 31 | 33 | 27.841827392578125 |

These 200 additional digest/salt/layout candidates test only counters 0..65535, in contrast to the full counter-domain sweep above. Large starting offsets and arbitrary salts remain unresolved.

| file | recipes | rows_per_recipe | max_matches | max_recipe |
| --- | --- | --- | --- | --- |
| train | 400 | 512 | 0 | (('sender_id', 'receiver_id', 'timestamp', 'amount'), '', '', 'md5') |
| test | 400 | 512 | 0 | (('sender_id', 'receiver_id', 'timestamp', 'amount'), '', '', 'md5') |

No significant-result claim is made from these additional searches. If no compact/monotone inverse or excess stream matches are found, no train label-purity table can legitimately be attributed to recovered order.

### Full-length ID-stream partial-order follow-up

| api | seed | draws | matching_ids | chance | creation_anchors | rho_stream_vs_device |
| --- | --- | --- | --- | --- | --- | --- |
| Python getrandbits32 | 2945 | 1824642 | 768 | 775.1673525581136 | 21 | -0.3090909090909091 |
| RandomState uint32 [0,2^32) | 4266 | 1824642 | 772 | 775.1673525581136 | 22 | 0.010728402032749858 |
| Generator uint32 [0,2^32) | 3822 | 1824642 | 769 | 775.1673525581136 | 22 | 0.10671936758893281 |

For each broad-seed stream family, extend its largest-prefix-hit seed to train+test length and compare stream position with creation-device number on matched anchors. Seed selection uses ID overlap only. This follows up apparent prefix outliers; it is not a full-length scan of all seeds.

## All numeric raw-column order checks

| file | column | finite_anchors | rho | lag1 |
| --- | --- | --- | --- | --- |
| train | amount | 48408 | -0.06751337332631398 | 0.1299579575627772 |
| train | sim_age_hours | 48408 | -0.0027366532563696147 | -0.0018111660092305654 |
| train | device_age_days | 48408 | 0.0835240102856891 | 0.06946462492163331 |
| train | sender_account_age_days | 48408 | 0.005938511320261293 | 0.017359480149834617 |
| train | new_payee | 48408 | -0.10451078714007896 | 0.0939515299718519 |
| train | pin_retries | 48408 | -0.08020541114706813 | 0.08596108881391838 |
| train | login_to_txn_sec | 48408 | 0.004777009411824988 | 0.0004481250921494565 |
| train | balance_before | 48408 | 0.037848294906412866 | 0.01149637076744287 |
| train | ip_risk_score | 48408 | -0.07622120426127983 | 0.0877834865882478 |
| train | qr_vpa_mismatch | 17554 | -0.04034011688412114 | 0.07494210435971538 |
| test | amount | 8026 | -0.07698088203664293 | 0.13762561455423108 |
| test | sim_age_hours | 8026 | 0.011636839627684548 | 0.008590452729005096 |
| test | device_age_days | 8026 | 0.0899080475481727 | 0.061408196307248614 |
| test | sender_account_age_days | 8026 | -0.012485255505845474 | 0.02943592994363388 |
| test | new_payee | 8026 | -0.11181232460522246 | 0.11309452895817751 |
| test | pin_retries | 8026 | -0.06948842395922397 | 0.07959008287152938 |
| test | login_to_txn_sec | 8026 | 0.01632479484904099 | -0.011606651372728965 |
| test | balance_before | 8026 | 0.010019304291870541 | 0.010490536470842169 |
| test | ip_risk_score | 8026 | -0.013738482424219134 | 0.006637638137847319 |
| test | qr_vpa_mismatch | 2884 | -0.03618092933802622 | 0.16318355973528378 |

All ten numeric raw columns are checked for rank relationship and lag-1 structure along creation-device order. ip_risk_score is included only as a diagnostic of a known train/test-changing field, never as a fraud rule. Recipe/phase shifts and discrete support can create autocorrelation without recovering generation order. Seeded exact-uniform sequence matching above covers ages and balances only; arbitrary numeric distributions, transformations and skipped draws remain outside that finite sweep.
