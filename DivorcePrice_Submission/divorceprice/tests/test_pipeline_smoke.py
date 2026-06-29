"""End-to-end smoke test on the synthetic dataset.

Verifies:
- the ensemble trains without error
- CV RMSLE is finite and < 0.5 (a very loose bound — the synthetic
  recipe has real signal, so anything more than this means the pipeline
  is broken)
- predict() and the SHAP explainer return sensible values
- the PDF report file is produced and non-empty
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from divorceprice import data, pipeline
from divorceprice.config import TrainConfig
from divorceprice.explain import explain_one
from divorceprice.models import fit_ensemble
from divorceprice.report import ReportContext, render


def test_full_pipeline(tmp_path: Path) -> None:
    df = data.make_synthetic(n=300, seed=7)
    cfg = TrainConfig(cv_folds=3, catboost_iterations=200)
    ens = fit_ensemble(df, cfg)

    assert math.isfinite(ens.cv_rmsle_blend)
    assert ens.cv_rmsle_blend < 0.5
    assert 0.0 <= ens.blend <= 1.0

    out = pipeline.save(ens, tmp_path / "artifacts")
    assert (out / "ensemble.joblib").exists()
    assert (out / "metrics.json").exists()

    loaded = pipeline.load(out)
    sample = df.drop(columns=["SalePrice"]).iloc[[0]]
    price, factors = explain_one(loaded, sample, top_k=5)
    assert price > 0
    assert len(factors) == 5
    # impacts should be finite
    for f in factors:
        assert math.isfinite(f.impact_eur)

    pdf_path = render(
        tmp_path / "report.pdf",
        ReportContext(
            case_id="TEST-001",
            valuation_date="2026-06-04",
            jurisdiction="France",
            property_address="Test address",
        ),
        loaded,
        sample.iloc[0].to_dict(),
        price,
        factors,
    )
    assert pdf_path.exists() and pdf_path.stat().st_size > 0
