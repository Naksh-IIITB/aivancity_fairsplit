from __future__ import annotations

import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.house_price_project import add_features, clean_known_missing_values, remove_training_outliers


DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"
DELIVERABLES = ROOT / "required_documents"


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_pipeline(model) -> Pipeline:
    train = pd.read_csv(DATA / "train.csv")
    train, _ = clean_known_missing_values(train, train.copy())
    train = remove_training_outliers(add_features(train))
    x = train.drop(columns=["Id", "SalePrice"], errors="ignore")
    num_cols = x.select_dtypes(include=np.number).columns.tolist()
    cat_cols = x.select_dtypes(include="object").columns.tolist()

    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", one_hot_encoder()),
                    ]
                ),
                cat_cols,
            ),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    DELIVERABLES.mkdir(exist_ok=True)

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    train, test = clean_known_missing_values(train, test)
    train = remove_training_outliers(add_features(train))
    test = add_features(test)

    x = train.drop(columns=["Id", "SalePrice"], errors="ignore")
    y = np.log(train["SalePrice"])
    test_x = test.drop(columns=["Id"], errors="ignore")

    candidates = [
        ("Ridge", Ridge(alpha=10.0)),
        ("ElasticNet", ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=30000, random_state=42)),
    ]

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    fitted = []
    for name, model in candidates:
        pipe = make_pipeline(model)
        scores = np.sqrt(
            -cross_val_score(
                pipe,
                x,
                y,
                cv=cv,
                scoring="neg_mean_squared_error",
                n_jobs=1,
            )
        )
        pipe.fit(x, y)
        rows.append(
            {
                "model": name,
                "cv_rmsle_mean": scores.mean(),
                "cv_rmsle_std": scores.std(),
                "best_params": model.get_params(),
            }
        )
        fitted.append((name, pipe, scores.mean()))

    comparison = pd.DataFrame(rows).sort_values("cv_rmsle_mean")
    comparison.to_csv(OUTPUTS / "model_comparison.csv", index=False)

    best_name, best_model, best_score = min(fitted, key=lambda item: item[2])
    joblib.dump(best_model, MODELS / "best_model.joblib")

    for name, pipe, _score in fitted:
        prices = np.exp(pipe.predict(test_x)).clip(min=1)
        pd.DataFrame({"Id": test["Id"], "SalePrice": prices}).to_csv(
            OUTPUTS / f"submission_{name.lower()}.csv",
            index=False,
        )

    preds = np.mean([pipe.predict(test_x) for _name, pipe, _score in fitted], axis=0)
    pd.DataFrame({"Id": test["Id"], "SalePrice": np.exp(preds).clip(min=1)}).to_csv(
        OUTPUTS / "submission_weighted_blend.csv",
        index=False,
    )

    summary = f"""# Final Technical Summary

Dataset: Kaggle House Prices - Advanced Regression Techniques

Best model in the fast verified run: {best_name}

Best 5-fold CV RMSLE: {best_score:.5f}

Generated outputs:

- model_comparison.csv
- submission_ridge.csv
- submission_elasticnet.csv
- submission_weighted_blend.csv
- best_model.joblib

The full project code also supports RandomForest, SVR, MLP, KernelRidge, stacking, and optional XGBoost/LightGBM/CatBoost where native dependencies are available.
"""
    (OUTPUTS / "final_technical_summary.md").write_text(summary, encoding="utf-8")

    files_to_collect = [
        ROOT / "README.md",
        ROOT / "requirements.txt",
        ROOT / "notebooks" / "house_prices_workflow.ipynb",
        ROOT / "notebooks" / "house_prices_workflow.py",
        ROOT / "docs" / "startup_deliverable.md",
        OUTPUTS / "model_comparison.csv",
        OUTPUTS / "submission_ridge.csv",
        OUTPUTS / "submission_elasticnet.csv",
        OUTPUTS / "submission_weighted_blend.csv",
        OUTPUTS / "final_technical_summary.md",
        MODELS / "best_model.joblib",
    ]

    for source in files_to_collect:
        target = DELIVERABLES / source.name
        if source.exists():
            shutil.copy2(source, target)

    print(comparison.to_string(index=False))
    print(f"\nCollected deliverables in {DELIVERABLES}")


if __name__ == "__main__":
    main()
