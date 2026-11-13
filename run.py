from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from upi.config import load_config, resolve_path  # noqa: E402
from upi.data import audit_inputs  # noqa: E402
from upi.pipeline import blend_runs, build_features, final_fit_predict, train_validation  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="UPI Fraud Rings reproducible pipeline")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("audit", "features", "train", "smoke"):
        q = sub.add_parser(name)
        q.add_argument("--config", required=True)
        if name == "features":
            q.add_argument("--force", action="store_true")
    q = sub.add_parser("predict")
    q.add_argument("--config", required=True)
    q.add_argument("--run-dir", required=True)
    q.add_argument("--output")
    q = sub.add_parser("blend")
    q.add_argument("--config", required=True)
    q.add_argument("--run-dirs", nargs="+", required=True)
    q.add_argument("--weights", nargs="+", type=float)
    q.add_argument("--output")
    return p


def main() -> int:
    args = parser().parse_args()
    config = load_config(args.config)
    if args.command == "audit":
        raw = resolve_path(config, "raw_dir", "data/raw")
        output = resolve_path(config, "reports_dir", "reports") / "data_audit.json"
        result = audit_inputs(raw, output)
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 2
    if args.command == "features":
        print(build_features(config, force=args.force))
        return 0
    if args.command == "train":
        print(train_validation(config))
        return 0
    if args.command == "predict":
        print(final_fit_predict(config, Path(args.run_dir), Path(args.output) if args.output else None))
        return 0
    if args.command == "blend":
        print(blend_runs(config, [Path(p) for p in args.run_dirs],
                         Path(args.output) if args.output else None, args.weights))
        return 0
    if args.command == "smoke":
        path = build_features(config, force=True, sample_fraction=float(config.get("smoke", {}).get("sample_fraction", .005)))
        print(train_validation(config, features=path))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

