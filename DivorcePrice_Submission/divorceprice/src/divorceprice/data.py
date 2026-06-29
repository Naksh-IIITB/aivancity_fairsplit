"""Dataset loading.

Production target: Kaggle "House Prices - Advanced Regression Techniques"
(Ames, Iowa). For smoke tests we generate a small synthetic dataset
shaped like the Ames schema so the pipeline runs end-to-end without
the Kaggle file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ID_COL, TARGET


def load_kaggle(csv_path: str | Path) -> pd.DataFrame:
    """Load the Kaggle Ames train.csv from disk."""
    df = pd.read_csv(csv_path)
    if TARGET not in df.columns:
        raise ValueError(
            f"Expected column '{TARGET}' in {csv_path}; got {list(df.columns)[:8]}…"
        )
    return df


def make_synthetic(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """Tiny dataset shaped like Ames — only the columns the model uses.

    Sale price is built from a deterministic-ish recipe so the model has
    real signal to learn (otherwise CV scores look like noise).
    """
    rng = np.random.default_rng(seed)

    neighborhoods = ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "Gilbert"]
    qualities = ["Ex", "Gd", "TA", "Fa", "Po"]
    quality_score = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}

    df = pd.DataFrame({
        ID_COL: np.arange(1, n + 1),
        "GrLivArea": rng.integers(700, 3500, n),
        "LotArea": rng.integers(2000, 20000, n),
        "OverallQual": rng.integers(3, 11, n),
        "OverallCond": rng.integers(2, 10, n),
        "YearBuilt": rng.integers(1920, 2011, n),
        "YearRemodAdd": rng.integers(1950, 2011, n),
        "TotalBsmtSF": rng.integers(0, 2000, n),
        "GarageCars": rng.integers(0, 4, n),
        "FullBath": rng.integers(0, 4, n),
        "BedroomAbvGr": rng.integers(1, 6, n),
        "Neighborhood": rng.choice(neighborhoods, n),
        "KitchenQual": rng.choice(qualities, n),
        "ExterQual": rng.choice(qualities, n),
        "CentralAir": rng.choice(["Y", "N"], n, p=[0.93, 0.07]),
    })

    # synthetic price recipe — deterministic-ish
    base = (
        45_000
        + 55 * df["GrLivArea"]
        + 1.5 * df["LotArea"]
        + 12_000 * df["OverallQual"]
        + 4_500 * df["OverallCond"]
        + 25 * df["TotalBsmtSF"]
        + 8_000 * df["GarageCars"]
        + 6_000 * df["FullBath"]
        + 250 * (df["YearBuilt"] - 1900)
        + 150 * (df["YearRemodAdd"] - 1900)
        + df["KitchenQual"].map(quality_score) * 9_000
        + df["ExterQual"].map(quality_score) * 6_000
        + df["Neighborhood"].map(
            {"NAmes": 0, "CollgCr": 18_000, "OldTown": -8_000,
             "Edwards": -5_000, "Somerst": 22_000, "Gilbert": 12_000}
        )
        + (df["CentralAir"] == "Y") * 5_000
    )
    noise = rng.normal(0, 12_000, n)
    df[TARGET] = np.clip(base + noise, 35_000, None).astype(int)
    return df


def load(source: str | Path | None) -> pd.DataFrame:
    """Convenience: pass a CSV path, or None for synthetic."""
    if source is None:
        return make_synthetic()
    return load_kaggle(source)
