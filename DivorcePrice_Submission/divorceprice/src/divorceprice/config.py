"""Project-wide constants and tunables."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

TARGET = "SalePrice"
ID_COL = "Id"

# Columns we always treat as ordinal/categorical regardless of dtype on disk.
KNOWN_CATEGORICAL = (
    "MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities",
    "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2",
    "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st",
    "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", "Foundation",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "Heating", "HeatingQC", "CentralAir", "Electrical", "KitchenQual",
    "Functional", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual",
    "GarageCond", "PavedDrive", "PoolQC", "Fence", "MiscFeature",
    "SaleType", "SaleCondition",
)


@dataclass(frozen=True)
class TrainConfig:
    cv_folds: int = 5
    random_state: int = 42
    elasticnet_alpha_grid: tuple = (0.0005, 0.001, 0.005, 0.01, 0.05)
    elasticnet_l1_grid: tuple = (0.1, 0.5, 0.9)
    catboost_iterations: int = 1500
    catboost_depth: int = 6
    catboost_lr: float = 0.05
    blend_weight_grid: tuple = field(
        default_factory=lambda: tuple(round(x * 0.05, 2) for x in range(21))
    )
    log_target: bool = True


@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parents[2]
    data_raw: Path = root / "data" / "raw"
    artifacts: Path = root / "artifacts"
    reports: Path = root / "reports"
