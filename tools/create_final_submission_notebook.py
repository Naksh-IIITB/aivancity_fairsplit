from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "ipynb_submission"
OUT = OUT_DIR / "House_Prices_AI_Project_Submission.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(keepends=True),
    }


cells = [
    md(
        """
        # House Prices - Advanced Regression Techniques

        **IIIT Bangalore x Aivancity - Hands-On AI Projects**

        This notebook is the final submission file. It contains:

        - EDA and data-cleaning strategy
        - Feature engineering
        - Leakage-safe preprocessing pipelines
        - Cross-validation with RMSLE
        - Ridge and ElasticNet model training
        - Kaggle-ready submission CSV generation
        - Summary of extended model experiments
        - Startup deliverable: Business Model Canvas, competitors, PESTEL, SWOT, ethics, governance, and regulation
        """
    ),
    md(
        """
        ## 1. Problem Framing

        The Kaggle task is a supervised regression problem: predict `SalePrice` for houses in Ames, Iowa using 79 explanatory variables.

        Kaggle evaluates submissions with RMSLE. Training on `log(SalePrice)` makes ordinary RMSE in cross-validation match the Kaggle metric and reduces the effect of extreme luxury-home prices.
        """
    ),
    code(
        """
        from pathlib import Path
        import numpy as np
        import pandas as pd

        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import ElasticNet, Ridge, Lasso, LinearRegression
        from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.ensemble import RandomForestRegressor, StackingRegressor

        RANDOM_STATE = 42
        DATA_DIR = Path("data")
        OUTPUT_DIR = Path("outputs")
        OUTPUT_DIR.mkdir(exist_ok=True)
        """
    ),
    md(
        """
        ## 2. Load Data

        Place `train.csv`, `test.csv`, `sample_submission.csv`, and `data_description.txt` in a folder named `data` next to this notebook.
        """
    ),
    code(
        """
        train = pd.read_csv(DATA_DIR / "train.csv")
        test = pd.read_csv(DATA_DIR / "test.csv")

        print(train.shape)
        print(test.shape)
        train.head()
        """
    ),
    md(
        """
        ## 3. Missing Values and Cleaning

        Some `NaN` values mean the feature is absent, not unknown. For example, `PoolQC = NaN` usually means the house has no pool. These are filled with `"None"` before modeling. `GarageYrBlt` is filled with `0` when there is no garage.
        """
    ),
    code(
        """
        absent_cols = [
            "PoolQC", "MiscFeature", "Alley", "Fence", "MasVnrType",
            "FireplaceQu",
            "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
            "GarageType", "GarageFinish", "GarageQual", "GarageCond",
        ]

        def clean_known_missing_values(train_df, test_df):
            train_df = train_df.copy()
            test_df = test_df.copy()
            for col in absent_cols:
                if col in train_df.columns:
                    train_df[col] = train_df[col].fillna("None")
                if col in test_df.columns:
                    test_df[col] = test_df[col].fillna("None")
            for df in (train_df, test_df):
                if "GarageYrBlt" in df.columns:
                    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0)
            return train_df, test_df

        train, test = clean_known_missing_values(train, test)

        missing = train.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        pd.DataFrame({"count": missing, "percent": (missing / len(train) * 100).round(1)}).head(20)
        """
    ),
    md(
        """
        ## 4. EDA Findings

        Key observations from the assignment workflow:

        - Raw `SalePrice` is right-skewed.
        - `log(SalePrice)` is much closer to a Gaussian distribution.
        - Strong predictors include `OverallQual`, `GrLivArea`, `GarageCars`, `GarageArea`, `TotalBsmtSF`, and `1stFlrSF`.
        - `GrLivArea` has two known outlier homes with very large living area but low sale price; removing them improves model robustness.
        """
    ),
    code(
        """
        print("Raw SalePrice skew:", round(train["SalePrice"].skew(), 3))
        print("Log SalePrice skew:", round(np.log(train["SalePrice"]).skew(), 3))

        num_cols = train.select_dtypes(include=np.number).columns
        corr_target = train[num_cols].corr(numeric_only=True)["SalePrice"].abs().sort_values(ascending=False)
        corr_target.head(12)
        """
    ),
    code(
        """
        # Remove the two famous GrLivArea outliers from the Kaggle training set.
        outlier_mask = (train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000)
        print("Outliers removed:", int(outlier_mask.sum()))
        train = train.loc[~outlier_mask].copy()
        """
    ),
    md(
        """
        ## 5. Feature Engineering

        The engineered features below are simple, interpretable, and useful for a real estate startup product:

        - `TotalSF`: total usable square footage
        - `TotalBath`: total bathroom equivalent count
        - `HouseAge`: age at sale
        - `YearsSinceRemodel`: time since last remodel
        - `QualityArea`: interaction between quality and living area
        """
    ),
    code(
        """
        def add_features(df):
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

        train = add_features(train)
        test = add_features(test)
        """
    ),
    md(
        """
        ## 6. Preprocessing Pipeline

        The pipeline is leakage-safe because imputation, scaling, and one-hot encoding are fitted inside each cross-validation fold.

        Numerical features:

        - Median imputation
        - Standard scaling

        Categorical features:

        - Mode imputation
        - One-hot encoding with `handle_unknown="ignore"`
        """
    ),
    code(
        """
        X = train.drop(columns=["Id", "SalePrice"], errors="ignore")
        y = np.log(train["SalePrice"])
        test_X = test.drop(columns=["Id"], errors="ignore")

        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(include="object").columns.tolist()

        try:
            onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

        preprocessor = ColumnTransformer(
            [
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
                ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", onehot)]), categorical_features),
            ]
        )

        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        """
    ),
    md(
        """
        ## 7. Model Training and Tuning

        The main verified models are Ridge and ElasticNet. ElasticNet is a good fit for this dataset because it combines:

        - L1 feature selection
        - L2 shrinkage for correlated features
        """
    ),
    code(
        """
        def cv_rmsle(model):
            scores = np.sqrt(
                -cross_val_score(
                    model,
                    X,
                    y,
                    cv=cv,
                    scoring="neg_mean_squared_error",
                    n_jobs=1,
                )
            )
            return scores.mean(), scores.std()

        models = {}

        ridge = Pipeline([("preprocessor", preprocessor), ("model", Ridge(alpha=10.0))])
        ridge_mean, ridge_std = cv_rmsle(ridge)
        ridge.fit(X, y)
        models["Ridge"] = ridge

        elastic = Pipeline([
            ("preprocessor", preprocessor),
            ("model", ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=30000, random_state=RANDOM_STATE)),
        ])
        elastic_mean, elastic_std = cv_rmsle(elastic)
        elastic.fit(X, y)
        models["ElasticNet"] = elastic

        comparison = pd.DataFrame(
            [
                {"model": "Ridge", "cv_rmsle_mean": ridge_mean, "cv_rmsle_std": ridge_std},
                {"model": "ElasticNet", "cv_rmsle_mean": elastic_mean, "cv_rmsle_std": elastic_std},
            ]
        ).sort_values("cv_rmsle_mean")

        comparison
        """
    ),
    md(
        """
        ## 8. Extended Experiments Summary

        The full project runner was also used to compare additional models from Sessions 2 and 3. In the verified run, the ranking was:

        | Model | CV RMSLE |
        | --- | ---: |
        | StackingRegressor | 0.111693 |
        | Lasso | 0.112828 |
        | ElasticNet | 0.113068 |
        | Ridge | 0.115107 |
        | CatBoost | 0.116463 |
        | LinearRegression | 0.131199 |
        | RandomForest | 0.133363 |
        | SVR_RBF | 0.140132 |
        | MLPRegressor | 0.175206 |
        | KernelRidge_RBF | 0.283853 |

        Conclusion: regularized linear models and stacking perform best on this mostly additive tabular dataset after log-transforming the target and one-hot encoding categoricals.
        """
    ),
    md(
        """
        ## 9. Kaggle Submission Files

        The following cell generates Kaggle-ready submission files. Kaggle expects exactly two columns: `Id` and `SalePrice`.
        """
    ),
    code(
        """
        for name, model in models.items():
            log_pred = model.predict(test_X)
            sale_price = np.exp(log_pred).clip(min=1)
            submission = pd.DataFrame({"Id": test["Id"], "SalePrice": sale_price})
            path = OUTPUT_DIR / f"submission_{name.lower()}.csv"
            submission.to_csv(path, index=False)
            print(path)

        # Simple average of Ridge and ElasticNet in log-space
        avg_log_pred = np.mean([model.predict(test_X) for model in models.values()], axis=0)
        blend = pd.DataFrame({"Id": test["Id"], "SalePrice": np.exp(avg_log_pred).clip(min=1)})
        blend.to_csv(OUTPUT_DIR / "submission_weighted_blend.csv", index=False)
        blend.head()
        """
    ),
    md(
        """
        # Startup Project: FairPrice AI

        ## Concept

        **FairPrice AI** is a buyer-facing property valuation assistant. A user enters listing details or pastes a listing summary, and the product returns:

        - Predicted fair sale price
        - Verdict: underpriced, fair, or overpriced
        - Confidence band
        - Main value drivers such as quality, living area, garage capacity, neighborhood, house age, and remodel status

        The AI model is the technical engine. The product translates model output into a clear buyer decision.
        """
    ),
    md(
        """
        ## Business Model Canvas

        | Block | Plan |
        | --- | --- |
        | Customer segments | First-time buyers, relocating professionals, buyer agents, mortgage pre-approval advisors |
        | Value proposition | Reduce overpayment risk with an independent fair-value estimate and explanation |
        | Channels | Mobile web app, browser extension, real estate agent partnerships, fintech/lender APIs |
        | Customer relationships | Freemium self-service, saved watchlists, alerts, premium valuation reports |
        | Revenue streams | Subscription, paid reports, B2B agent seats, API usage |
        | Key resources | Trained valuation model, property data, feature engineering, explainability layer |
        | Key activities | Data cleaning, model retraining, monitoring, UX, integrations, governance |
        | Key partners | Listing portals, real estate agencies, lenders, local data providers |
        | Cost structure | Data licensing, cloud inference, engineering, compliance, customer acquisition |
        """
    ),
    md(
        """
        ## Competitors and Positioning

        | Competitor | Strength | Weakness | FairPrice AI Positioning |
        | --- | --- | --- | --- |
        | Zillow Zestimate | Strong brand and data scale | Black-box perception, US-focused | Explainable buyer assistant with clear drivers |
        | Redfin Estimate | Integrated with brokerage workflow | Limited to supported markets | Independent tool across portals |
        | Realtor.com calculators | Simple consumer reach | Limited interpretability | Converts valuation into negotiation guidance |

        **Unique positioning:** explainable fair-value intelligence for buyers before they make an offer.
        """
    ),
    md(
        """
        ## PESTEL Analysis

        | Factor | Implications |
        | --- | --- |
        | Political | Housing affordability is sensitive; transparency can support public trust |
        | Economic | High rates increase buyer caution and demand for valuation confidence |
        | Social | First-time buyers need simple and trustworthy decision support |
        | Technological | Tabular ML, explainability, and AI agents make valuation tools more useful |
        | Environmental | Climate risk can affect long-term value and insurance cost |
        | Legal | Must avoid misleading claims, bias, and misuse of personal data |
        """
    ),
    md(
        """
        ## SWOT Analysis

        | Strengths | Weaknesses |
        | --- | --- |
        | Clear technical MVP, explainable outputs, low marginal prediction cost | Kaggle data is limited to Ames, Iowa and historical sales |

        | Opportunities | Threats |
        | --- | --- |
        | Climate-aware valuation, renovation ROI, lender APIs, international markets | Incumbents have better data access; regulation and model bias risks |
        """
    ),
    md(
        """
        ## Ethics, Data Governance, and Regulation

        Short-term safeguards:

        - Present predictions as estimates, not official appraisals.
        - Show uncertainty and top drivers.
        - Avoid protected attributes and obvious proxies where possible.
        - Audit model error by neighborhood and price segment.
        - Log model version, data version, and prediction inputs.

        Long-term governance:

        - Monitor drift and geographic bias.
        - Use legally licensed and current market data.
        - Add human review for lending or other high-stakes uses.
        - Publish plain-language model documentation.
        - Comply with privacy rules such as GDPR where user data is collected.
        """
    ),
    md(
        """
        ## Final Conclusion

        The project delivers the complete technical foundation for a house-price prediction startup:

        - Clean preprocessing and feature engineering
        - Cross-validated model comparison
        - Kaggle-ready submissions
        - Business model and strategy
        - Competitor, PESTEL, SWOT, ethics, governance, and regulation analysis

        The recommended production direction is a FairPrice AI web or mobile app that uses the trained valuation model as the backend engine and exposes explainable pricing advice to home buyers.
        """
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
