# Modelling integration for codex/target-070

All implementation commits and the final push belong to `codex/target-070`
in `https://github.com/zahy294/upi-fraud-ml.git`.

Starting target SHA: `7ae1941900737d34eb0139e0e4295e2e3e597ecc`.
Merged modelling SHA: `d4ecf6caef06c08fc43b965f14fbabb998a3a34a`.

The merge preserves the existing target pipeline, configurations, tests and
Colab notebook. The shared Events implementation retains the target API and
adds modelling's MIN/HOUR/DAY constants, vals keyword and prev_dt/next_dt
methods. Gap/value queries explicitly check group identity.

Verification: Python syntax compilation of the integrated source modules,
small in-memory event compatibility checks, and git diff whitespace checks.
No real data, model artifacts, notebooks, or training runs were inspected or
executed for this integration. These checks do not establish training success
or leaderboard performance.

The three implementation cards remain subsequent work. Their branch references
and future Colab checkout instructions must use `codex/target-070`. Push only
after Card 3 verification, using `HEAD:refs/heads/codex/target-070`; never force
push or push modelling/main.
