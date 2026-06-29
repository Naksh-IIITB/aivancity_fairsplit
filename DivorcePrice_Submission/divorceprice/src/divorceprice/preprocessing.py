"""Preprocessing pipeline.

Two flavours:
- linear: median-impute → standardize numeric, mode-impute → one-hot encode
  categorical. Used by ElasticNet (which needs scaled, fully numeric inputs).
- tree:   median-impute numeric, fill "Missing" + label-encode categorical.
  Used by CatBoost (handles raw categoricals natively, but we still want
  imputation for missing values).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ID_COL, KNOWN_CATEGORICAL, TARGET


def split_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols), excluding ID + target."""
    drop = {ID_COL, TARGET}
    cats = [c for c in df.columns
            if c not in drop and (c in KNOWN_CATEGORICAL or df[c].dtype == "object")]
    nums = [c for c in df.columns if c not in drop and c not in cats]
    return nums, cats


def build_linear_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", num, num_cols), ("cat", cat, cat_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


@dataclass
class TreeFrame:
    """CatBoost-friendly view of the dataset.

    CatBoost wants raw categorical columns (as strings) plus their indices.
    """
    X: pd.DataFrame
    cat_idx: list[int]


def to_tree_frame(df: pd.DataFrame, num_cols: list[str], cat_cols: list[str]) -> TreeFrame:
    """Median-impute numerics, fill 'Missing' for categoricals, return df + cat indices."""
    out = pd.DataFrame(index=df.index)
    for c in num_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        out[c] = s.fillna(s.median())
    for c in cat_cols:
        out[c] = df[c].astype("string").fillna("Missing").astype(str)
    cat_idx = [out.columns.get_loc(c) for c in cat_cols]
    return TreeFrame(X=out, cat_idx=cat_idx)


def log1p_target(y: pd.Series | np.ndarray) -> np.ndarray:
    return np.log1p(np.asarray(y, dtype=float))


def expm1_target(y_log: np.ndarray) -> np.ndarray:
    return np.expm1(y_log)
