"""Train the ensemble and persist artifacts.

Usage:
    python -m scripts.train --data data/raw/train.csv --out artifacts/
    python -m scripts.train --synthetic --out artifacts/      # no Kaggle file needed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# allow running as `python -m scripts.train` from repo root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from divorceprice import data, pipeline           # noqa: E402
from divorceprice.config import TrainConfig       # noqa: E402
from divorceprice.models import fit_ensemble      # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Train FairSplit ensemble")
    p.add_argument("--data", type=str, default=None,
                   help="Path to Kaggle train.csv. Omit with --synthetic.")
    p.add_argument("--synthetic", action="store_true",
                   help="Use a generated synthetic dataset (smoke test).")
    p.add_argument("--out", type=str, default="artifacts/",
                   help="Directory to write ensemble.joblib + metrics.json")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--catboost-iterations", type=int, default=1500)
    args = p.parse_args()

    if not args.synthetic and args.data is None:
        p.error("Pass --data or --synthetic")

    print(f"[train] loading dataset ({'synthetic' if args.synthetic else args.data})")
    df = data.load(None if args.synthetic else args.data)
    print(f"[train] dataset shape: {df.shape}")

    cfg = TrainConfig(
        cv_folds=args.cv_folds,
        catboost_iterations=args.catboost_iterations,
    )
    print(f"[train] fitting ensemble — {cfg.cv_folds}-fold CV…")
    ens = fit_ensemble(df, cfg)

    print(f"[train] CV RMSLE — ElasticNet : {ens.cv_rmsle_linear:.4f}")
    print(f"[train] CV RMSLE — CatBoost  : {ens.cv_rmsle_catboost:.4f}")
    print(f"[train] CV RMSLE — Ensemble  : {ens.cv_rmsle_blend:.4f} "
          f"(blend = {ens.blend:.2f} linear, {1 - ens.blend:.2f} tree)")

    out = pipeline.save(ens, args.out)
    print(f"[train] artifacts written to {out}/")
    print(f"[train] metrics: {json.loads((out / 'metrics.json').read_text())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
