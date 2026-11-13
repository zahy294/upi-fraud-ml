"""Compare the supplied scored CSV with the exact existing rules exporter."""
import hashlib
import json
import sys
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
root, submitted = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
path = REPO / "scripts/eda/07_baselines.py"
source = path.read_text()
marker = 'is_tr = df.split == "train"'
assert source.count(marker) == 1
sys.argv = [str(path), str(root)]
ns = {"__file__": str(path), "__name__": "__main__"}
exec(compile(source.split(marker)[0], str(path), "exec"), ns)
df, prediction = ns["df"], ns["pred_rule"]
mask = df.split == "test"
rules = pd.DataFrame({"id": df.loc[mask, "txn_id"].to_numpy(), "rules": prediction[mask].to_numpy()})
s = pd.read_csv(submitted)
assert list(s) == ["id", "label"] and s.id.is_unique
assert set(s.id) == set(rules.id)
joined = s.merge(rules, on="id", how="left", validate="one_to_one")
agreement = joined.label == joined.rules
union = (joined.label != "legit") | (joined.rules != "legit")
result = {"file": str(submitted), "sha256": hashlib.sha256(submitted.read_bytes()).hexdigest(),
    "rows": len(s), "valid_ids": True, "exact_match": bool(agreement.all()),
    "different_rows": int((~agreement).sum()), "overall_agreement": float(agreement.mean()),
    "fraud_union_rows": int(union.sum()), "fraud_union_agreement": float(agreement[union].mean()),
    "submitted_counts": s.label.value_counts().to_dict(), "rules_counts": rules.rules.value_counts().to_dict(),
    "disagreements": joined.loc[~agreement].groupby(["label", "rules"]).size().rename("n").reset_index().to_dict("records")}
(root / "submission_identity.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
