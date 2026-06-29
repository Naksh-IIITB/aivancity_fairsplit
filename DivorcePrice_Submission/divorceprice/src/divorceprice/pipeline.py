"""Persistence helpers — save/load a fitted ensemble."""
from __future__ import annotations

import json
from pathlib import Path

import joblib

from .models import FittedEnsemble


def save(ensemble: FittedEnsemble, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(ensemble, out_dir / "ensemble.joblib")
    metrics = {
        "blend_weight_linear": ensemble.blend,
        "cv_rmsle_linear": ensemble.cv_rmsle_linear,
        "cv_rmsle_catboost": ensemble.cv_rmsle_catboost,
        "cv_rmsle_blend": ensemble.cv_rmsle_blend,
        "cv_folds": ensemble.cv_folds,
        "n_numeric_features": len(ensemble.num_cols),
        "n_categorical_features": len(ensemble.cat_cols),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return out_dir


def load(in_dir: str | Path) -> FittedEnsemble:
    return joblib.load(Path(in_dir) / "ensemble.joblib")
