"""SHAP explanation for a single property.

We use TreeExplainer on the CatBoost model (the harder-to-interpret part of
the blend) to produce per-feature contributions. The output is rendered as a
human-readable table for the legal report.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from catboost import Pool

from .models import FittedEnsemble
from .preprocessing import to_tree_frame


@dataclass
class FactorContribution:
    feature: str
    value: object       # raw input value as the user provided it
    impact_log: float   # SHAP value in log-price space
    impact_eur: float   # approximate price impact in raw currency

    @property
    def direction(self) -> str:
        return "↑" if self.impact_eur >= 0 else "↓"


def explain_one(ensemble: FittedEnsemble, row: pd.DataFrame, top_k: int = 10
                ) -> tuple[float, list[FactorContribution]]:
    """Return (predicted_price, top_k contributing factors).

    SHAP is computed on the CatBoost branch only — for a divorce report
    we want the most legible per-feature attribution, and TreeExplainer
    on CatBoost is exact and fast. The blended price is what we return.
    """
    if len(row) != 1:
        raise ValueError("explain_one expects a single-row DataFrame")

    predicted_price = float(ensemble.predict(row)[0])

    tree = to_tree_frame(row, ensemble.num_cols, ensemble.cat_cols)
    explainer = shap.TreeExplainer(ensemble.catboost)
    sv = explainer.shap_values(Pool(tree.X, cat_features=tree.cat_idx))[0]

    # convert log-space SHAP → approximate currency impact at the prediction
    # scale (∂price/∂log_price ≈ price)
    eur_per_log = predicted_price
    contribs = []
    for feat, shap_val in zip(tree.X.columns, sv):
        contribs.append(FactorContribution(
            feature=feat,
            value=row.iloc[0][feat],
            impact_log=float(shap_val),
            impact_eur=float(shap_val) * eur_per_log,
        ))

    contribs.sort(key=lambda c: abs(c.impact_eur), reverse=True)
    return predicted_price, contribs[:top_k]
