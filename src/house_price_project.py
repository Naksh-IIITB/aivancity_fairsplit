from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVR

RANDOM_STATE = 42

ABSENT_CATEGORICAL_COLUMNS = [
    "PoolQC",
    "MiscFeature",
    "Alley",
    "Fence",
    "MasVnrType",
    "FireplaceQu",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
]


@dataclass
class ModelResult:
    name: str
    cv_rmsle_mean: float
    cv_rmsle_std: float
    best_params: dict[str, Any]
    estimator: Any


def optional_import(module_name: str, class_name: str | None = None) -> Any | None:
    try:
        module = __import__(module_name, fromlist=[class_name] if class_name else [])
    except Exception:
        return None
    return getattr(module, class_name) if class_name else module


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def rmsle_cv(estimator: Any, x: pd.DataFrame, y: pd.Series, cv: KFold) -> tuple[float, float]:
    neg_mse = cross_val_score(
        estimator,
        x,
        y,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=1,
    )
    scores = np.sqrt(-neg_mse)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return float("inf"), float("nan")
    return float(scores.mean()), float(scores.std())


def grid_search(
    name: str,
    estimator: Any,
    param_grid: dict[str, list[Any]],
    x: pd.DataFrame,
    y: pd.Series,
    cv: KFold,
) -> ModelResult:
    grid = GridSearchCV(
        estimator,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=1,
        error_score="raise",
    )
    grid.fit(x, y)
    mean = math.sqrt(-grid.best_score_)
    std = float("nan")
    return ModelResult(name, float(mean), std, dict(grid.best_params_), grid.best_estimator_)


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    missing = [str(path) for path in [train_path, test_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Kaggle data files: "
            + ", ".join(missing)
            + ". Download them from the competition Data tab and place them in data/."
        )
    return pd.read_csv(train_path), pd.read_csv(test_path)


def clean_known_missing_values(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()

    for col in ABSENT_CATEGORICAL_COLUMNS:
        if col in train.columns:
            train[col] = train[col].fillna("None")
        if col in test.columns:
            test[col] = test[col].fillna("None")

    for frame in (train, test):
        if "GarageYrBlt" in frame.columns:
            frame["GarageYrBlt"] = frame["GarageYrBlt"].fillna(0)

    return train, test


def remove_training_outliers(train: pd.DataFrame) -> pd.DataFrame:
    if {"GrLivArea", "SalePrice"}.issubset(train.columns):
        mask = (train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000)
        return train.loc[~mask].copy()
    return train.copy()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]:
        if col not in df.columns:
            df[col] = 0

    df["TotalSF"] = df["TotalBsmtSF"].fillna(0) + df["1stFlrSF"].fillna(0) + df["2ndFlrSF"].fillna(0)
    df["TotalBath"] = (
        df["FullBath"].fillna(0)
        + 0.5 * df["HalfBath"].fillna(0)
        + df["BsmtFullBath"].fillna(0)
        + 0.5 * df["BsmtHalfBath"].fillna(0)
    )

    if {"YrSold", "YearBuilt"}.issubset(df.columns):
        df["HouseAge"] = (df["YrSold"] - df["YearBuilt"]).clip(lower=0)
    if {"YrSold", "YearRemodAdd"}.issubset(df.columns):
        df["YearsSinceRemodel"] = (df["YrSold"] - df["YearRemodAdd"]).clip(lower=0)
    if {"OverallQual", "GrLivArea"}.issubset(df.columns):
        df["QualityArea"] = df["OverallQual"] * df["GrLivArea"]

    return df


def make_xy(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = train.drop(columns=["Id", "SalePrice"], errors="ignore")
    y = np.log(train["SalePrice"])
    return x, y


def column_lists(x: pd.DataFrame) -> tuple[list[str], list[str]]:
    num_cols = x.select_dtypes(include=np.number).columns.tolist()
    cat_cols = x.select_dtypes(include="object").columns.tolist()
    return num_cols, cat_cols


def linear_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        [
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )


def tree_one_hot_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        [
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )


def ordinal_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )


def target_encoder_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer | None:
    target_encoder = optional_import("sklearn.preprocessing", "TargetEncoder")
    if target_encoder is None:
        return None
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("target", target_encoder(smooth="auto", random_state=RANDOM_STATE, cv=5)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )


def evaluate_linear_models(
    x: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    cv: KFold,
    fast: bool,
) -> list[ModelResult]:
    results: list[ModelResult] = []

    baseline = Pipeline([("preprocessor", preprocessor), ("model", LinearRegression())])
    mean, std = rmsle_cv(baseline, x, y, cv)
    baseline.fit(x, y)
    results.append(ModelResult("LinearRegression", mean, std, {}, baseline))

    ridge_grid = [0.1, 1, 10, 30] if fast else [0.01, 0.1, 1, 10, 30, 100]
    lasso_grid = [0.0005, 0.001, 0.005] if fast else [0.0001, 0.0005, 0.001, 0.01, 0.1, 1]
    enet_alpha = [0.0005, 0.001, 0.005] if fast else [0.0001, 0.0005, 0.001, 0.01, 0.1]
    enet_l1 = [0.3, 0.5, 0.7] if fast else [0.1, 0.3, 0.5, 0.7, 0.9]

    results.append(
        grid_search(
            "Ridge",
            Pipeline([("preprocessor", preprocessor), ("model", Ridge())]),
            {"model__alpha": ridge_grid},
            x,
            y,
            cv,
        )
    )
    results.append(
        grid_search(
            "Lasso",
            Pipeline([("preprocessor", preprocessor), ("model", Lasso(max_iter=20000, random_state=RANDOM_STATE))]),
            {"model__alpha": lasso_grid},
            x,
            y,
            cv,
        )
    )
    results.append(
        grid_search(
            "ElasticNet",
            Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("model", ElasticNet(max_iter=30000, random_state=RANDOM_STATE)),
                ]
            ),
            {"model__alpha": enet_alpha, "model__l1_ratio": enet_l1},
            x,
            y,
            cv,
        )
    )
    return results


def evaluate_tree_models(
    x: pd.DataFrame,
    y: pd.Series,
    one_hot_preprocessor: ColumnTransformer,
    te_preprocessor: ColumnTransformer | None,
    num_cols: list[str],
    cat_cols: list[str],
    cv: KFold,
    fast: bool,
) -> list[ModelResult]:
    results: list[ModelResult] = []

    rf = Pipeline(
        [
            ("preprocessor", one_hot_preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200 if fast else 500,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )
    rf_grid = {
        "model__max_depth": [None, 20] if fast else [None, 10, 20, 30],
        "model__min_samples_leaf": [1, 2] if fast else [1, 2, 5],
    }
    results.append(grid_search("RandomForest", rf, rf_grid, x, y, cv))

    xgb_class = optional_import("xgboost", "XGBRegressor")
    if xgb_class is not None and te_preprocessor is not None:
        xgb = Pipeline(
            [
                ("preprocessor", te_preprocessor),
                (
                    "model",
                    xgb_class(
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                        eval_metric="rmse",
                    ),
                ),
            ]
        )
        xgb_grid = {
            "model__n_estimators": [300, 500] if fast else [300, 500, 1000],
            "model__learning_rate": [0.03, 0.05] if fast else [0.03, 0.05, 0.1],
            "model__max_depth": [2, 3] if fast else [2, 3, 5],
        }
        results.append(grid_search("XGBoost", xgb, xgb_grid, x, y, cv))

    lgbm_class = optional_import("lightgbm", "LGBMRegressor")
    if lgbm_class is not None:
        lgbm = Pipeline(
            [
                ("preprocessor", ordinal_preprocessor(num_cols, cat_cols)),
                ("model", lgbm_class(random_state=RANDOM_STATE, n_jobs=1, verbose=-1)),
            ]
        )
        lgbm_grid = {
            "model__n_estimators": [300, 500] if fast else [500, 1000],
            "model__learning_rate": [0.03, 0.05] if fast else [0.03, 0.05, 0.1],
            "model__num_leaves": [15, 31] if fast else [15, 31, 63],
        }
        results.append(grid_search("LightGBM", lgbm, lgbm_grid, x, y, cv))

    catboost_class = optional_import("catboost", "CatBoostRegressor")
    if catboost_class is not None:
        x_cb = x.copy()
        x_cb[cat_cols] = x_cb[cat_cols].fillna("missing")
        cb = catboost_class(
            random_state=RANDOM_STATE,
            verbose=0,
            thread_count=1,
            allow_writing_files=False,
        )
        cb_grid = {
            "iterations": [300, 500] if fast else [500, 1000, 1500],
            "learning_rate": [0.03, 0.05] if fast else [0.03, 0.05, 0.1],
            "depth": [4, 6] if fast else [4, 6, 8],
        }
        grid_cb = GridSearchCV(
            cb,
            param_grid=cb_grid,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=1,
            error_score="raise",
        )
        grid_cb.fit(x_cb, y, cat_features=cat_cols)
        results.append(
            ModelResult(
                "CatBoost",
                float(math.sqrt(-grid_cb.best_score_)),
                float("nan"),
                dict(grid_cb.best_params_),
                grid_cb.best_estimator_,
            )
        )

    return results


def evaluate_session3_models(
    x: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    cv: KFold,
    fast: bool,
) -> list[ModelResult]:
    results: list[ModelResult] = []

    svr_grid = (
        {"model__C": [5, 10], "model__gamma": ["scale"], "model__epsilon": [0.01, 0.03]}
        if fast
        else {"model__C": [1, 5, 10, 30], "model__gamma": ["scale", 0.001, 0.01], "model__epsilon": [0.01, 0.03, 0.05]}
    )
    results.append(
        grid_search(
            "SVR_RBF",
            Pipeline([("preprocessor", preprocessor), ("model", SVR(kernel="rbf"))]),
            svr_grid,
            x,
            y,
            cv,
        )
    )

    mlp = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(128, 64) if not fast else (64,),
                    alpha=0.001,
                    learning_rate_init=0.001,
                    max_iter=700 if not fast else 300,
                    early_stopping=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    mean, std = rmsle_cv(mlp, x, y, cv)
    mlp.fit(x, y)
    results.append(ModelResult("MLPRegressor", mean, std, {}, mlp))

    krr_grid = (
        {"model__alpha": [0.1, 0.3], "model__gamma": [0.001, 0.003]}
        if fast
        else {"model__alpha": [0.03, 0.1, 0.3, 1.0], "model__gamma": [0.0005, 0.001, 0.003, 0.01]}
    )
    results.append(
        grid_search(
            "KernelRidge_RBF",
            Pipeline([("preprocessor", preprocessor), ("model", KernelRidge(kernel="rbf"))]),
            krr_grid,
            x,
            y,
            cv,
        )
    )
    return results


def build_stacking_model(results: list[ModelResult], preprocessor: ColumnTransformer, x: pd.DataFrame, y: pd.Series, cv: KFold) -> ModelResult:
    candidate_names = {"ElasticNet", "Ridge", "Lasso", "SVR_RBF", "KernelRidge_RBF"}
    base_models = [(r.name.lower(), r.estimator) for r in results if r.name in candidate_names][:4]
    if len(base_models) < 2:
        raise ValueError("Need at least two base models for stacking.")
    stack = StackingRegressor(
        estimators=base_models,
        final_estimator=Ridge(alpha=10.0),
        cv=5,
        n_jobs=1,
        passthrough=False,
    )
    mean, std = rmsle_cv(stack, x, y, cv)
    stack.fit(x, y)
    return ModelResult("StackingRegressor", mean, std, {}, stack)


def make_submission(estimator: Any, test: pd.DataFrame, output_path: Path, cat_cols: list[str] | None = None) -> None:
    test_x = test.drop(columns=["Id"], errors="ignore").copy()
    if cat_cols:
        present = [c for c in cat_cols if c in test_x.columns]
        test_x[present] = test_x[present].fillna("missing")
    log_preds = estimator.predict(test_x)
    prices = np.exp(log_preds).clip(min=1)
    pd.DataFrame({"Id": test["Id"], "SalePrice": prices}).to_csv(output_path, index=False)


def weighted_blend_submission(results: list[ModelResult], test: pd.DataFrame, output_path: Path, cat_cols: list[str]) -> None:
    best = sorted(results, key=lambda r: r.cv_rmsle_mean)[:3]
    test_x = test.drop(columns=["Id"], errors="ignore")
    inverse_errors = np.array([1 / max(r.cv_rmsle_mean, 1e-6) for r in best])
    weights = inverse_errors / inverse_errors.sum()
    log_preds = np.zeros(len(test_x))
    for weight, result in zip(weights, best):
        model_x = test_x.copy()
        if result.name == "CatBoost":
            present = [c for c in cat_cols if c in model_x.columns]
            model_x[present] = model_x[present].fillna("missing")
        log_preds += weight * result.estimator.predict(model_x)
    prices = np.exp(log_preds).clip(min=1)
    pd.DataFrame({"Id": test["Id"], "SalePrice": prices}).to_csv(output_path, index=False)


def save_eda(train: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(train["SalePrice"], kde=True, ax=axes[0])
    axes[0].set_title(f"Raw SalePrice skew = {train['SalePrice'].skew():.2f}")
    sns.histplot(np.log(train["SalePrice"]), kde=True, ax=axes[1], color="green")
    axes[1].set_title(f"log(SalePrice) skew = {np.log(train['SalePrice']).skew():.2f}")
    fig.tight_layout()
    fig.savefig(figures_dir / "saleprice_distribution.png", dpi=160)
    plt.close(fig)

    missing = train.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False).head(20)
    if not missing.empty:
        (missing / len(train) * 100).to_frame("missing_percent").to_csv(output_dir / "missing_values_top20.csv")

    num_cols = train.select_dtypes(include=np.number).columns
    corr_target = train[num_cols].corr(numeric_only=True)["SalePrice"].abs().sort_values(ascending=False)
    corr_target.head(20).to_csv(output_dir / "top_saleprice_correlations.csv")

    top_features = corr_target.head(11).index
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(train[top_features].corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    fig.tight_layout()
    fig.savefig(figures_dir / "top_correlations_heatmap.png", dpi=160)
    plt.close(fig)

    if "GrLivArea" in train.columns:
        fig = plt.figure(figsize=(8, 5))
        sns.scatterplot(data=train, x="GrLivArea", y="SalePrice")
        plt.title("GrLivArea vs SalePrice")
        fig.tight_layout()
        fig.savefig(figures_dir / "grlivarea_outliers.png", dpi=160)
        plt.close(fig)


def save_feature_importance(best_result: ModelResult, x: pd.DataFrame, y: pd.Series, output_dir: Path) -> None:
    try:
        importance = permutation_importance(
            best_result.estimator,
            x,
            y,
            n_repeats=5,
            random_state=RANDOM_STATE,
            scoring="neg_mean_squared_error",
            n_jobs=1,
        )
    except Exception as exc:  # noqa: BLE001
        (output_dir / "feature_importance_error.txt").write_text(str(exc), encoding="utf-8")
        return

    df = pd.DataFrame({"feature": x.columns, "importance": importance.importances_mean}).sort_values("importance", ascending=False)
    df.to_csv(output_dir / "permutation_importance.csv", index=False)


def results_to_frame(results: list[ModelResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": r.name,
                "cv_rmsle_mean": r.cv_rmsle_mean,
                "cv_rmsle_std": r.cv_rmsle_std,
                "best_params": json.dumps(r.best_params, default=str),
            }
            for r in sorted(results, key=lambda item: item.cv_rmsle_mean if np.isfinite(item.cv_rmsle_mean) else float("inf"))
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete House Prices regression workflow.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--fast", action="store_true", help="Use smaller grids for workshop-speed runs.")
    parser.add_argument("--skip-eda-plots", action="store_true", help="Skip Matplotlib plots in restricted environments.")
    parser.add_argument("--skip-optional-boosters", action="store_true", help="Skip XGBoost, LightGBM, and CatBoost.")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    train, test = load_data(args.data_dir)
    train, test = clean_known_missing_values(train, test)
    train = remove_training_outliers(train)
    train = add_features(train)
    test = add_features(test)
    if args.skip_eda_plots:
        missing = train.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False).head(20)
        if not missing.empty:
            (missing / len(train) * 100).to_frame("missing_percent").to_csv(args.output_dir / "missing_values_top20.csv")
        num_cols_for_corr = train.select_dtypes(include=np.number).columns
        train[num_cols_for_corr].corr(numeric_only=True)["SalePrice"].abs().sort_values(ascending=False).head(20).to_csv(
            args.output_dir / "top_saleprice_correlations.csv"
        )
    else:
        save_eda(train, args.output_dir)

    x, y = make_xy(train)
    num_cols, cat_cols = column_lists(x)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    preprocessor = linear_preprocessor(num_cols, cat_cols)
    tree_preprocessor = tree_one_hot_preprocessor(num_cols, cat_cols)
    te_preprocessor = target_encoder_preprocessor(num_cols, cat_cols)

    results: list[ModelResult] = []
    results.extend(evaluate_linear_models(x, y, preprocessor, cv, args.fast))
    if args.skip_optional_boosters:
        results.append(
            grid_search(
                "RandomForest",
                Pipeline(
                    [
                        ("preprocessor", tree_preprocessor),
                        (
                            "model",
                            RandomForestRegressor(
                                n_estimators=200 if args.fast else 500,
                                random_state=RANDOM_STATE,
                                n_jobs=1,
                            ),
                        ),
                    ]
                ),
                {
                    "model__max_depth": [None, 20] if args.fast else [None, 10, 20, 30],
                    "model__min_samples_leaf": [1, 2] if args.fast else [1, 2, 5],
                },
                x,
                y,
                cv,
            )
        )
    else:
        results.extend(evaluate_tree_models(x, y, tree_preprocessor, te_preprocessor, num_cols, cat_cols, cv, args.fast))
    results.extend(evaluate_session3_models(x, y, preprocessor, cv, args.fast))

    try:
        results.append(build_stacking_model(results, preprocessor, x, y, cv))
    except Exception as exc:  # noqa: BLE001
        (args.output_dir / "stacking_error.txt").write_text(str(exc), encoding="utf-8")

    comparison = results_to_frame(results)
    comparison.to_csv(args.output_dir / "model_comparison.csv", index=False)
    print(comparison.to_string(index=False))

    best = min(results, key=lambda r: r.cv_rmsle_mean if np.isfinite(r.cv_rmsle_mean) else float("inf"))
    joblib.dump(best.estimator, args.models_dir / "best_model.joblib")
    save_feature_importance(best, x, y, args.output_dir)

    for result in results:
        safe_name = result.name.lower().replace(" ", "_")
        if result.name == "CatBoost":
            make_submission(result.estimator, test, args.output_dir / f"submission_{safe_name}.csv", cat_cols)
        else:
            make_submission(result.estimator, test, args.output_dir / f"submission_{safe_name}.csv")
    weighted_blend_submission(results, test, args.output_dir / "submission_weighted_blend.csv", cat_cols)

    print(f"\nBest model: {best.name} | CV RMSLE: {best.cv_rmsle_mean:.4f}")
    print(f"Outputs saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
