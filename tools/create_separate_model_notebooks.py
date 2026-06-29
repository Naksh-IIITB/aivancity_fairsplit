from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "ipynb_submission"


def lines(text: str) -> list[str]:
    return text.strip().splitlines(keepends=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SETUP = """
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

RANDOM_STATE = 42
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
"""


COMMON_DATA = """
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

absent_cols = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "MasVnrType",
    "FireplaceQu",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
]

for col in absent_cols:
    if col in train.columns:
        train[col] = train[col].fillna("None")
    if col in test.columns:
        test[col] = test[col].fillna("None")

for df in (train, test):
    if "GarageYrBlt" in df.columns:
        df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0)

outlier_mask = (train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000)
train = train.loc[~outlier_mask].copy()

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

X = train.drop(columns=["Id", "SalePrice"], errors="ignore")
y = np.log(train["SalePrice"])
test_X = test.drop(columns=["Id"], errors="ignore")

numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include="object").columns.tolist()

try:
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

linear_preprocessor = ColumnTransformer(
    [
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", onehot)]), categorical_features),
    ]
)

tree_preprocessor = ColumnTransformer(
    [
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", onehot)]), categorical_features),
    ]
)

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def rmsle_cv(model):
    scores = np.sqrt(-cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=1))
    return scores.mean(), scores.std()

def save_submission(model, name):
    log_pred = model.predict(test_X)
    submission = pd.DataFrame({"Id": test["Id"], "SalePrice": np.exp(log_pred).clip(min=1)})
    path = OUTPUT_DIR / f"submission_{name}.csv"
    submission.to_csv(path, index=False)
    return path, submission.head()
"""


MODELS: dict[str, dict[str, str]] = {
    "LinearRegression": {
        "filename": "01_LinearRegression.ipynb",
        "title": "LinearRegression Baseline",
        "text": "Baseline ordinary least squares model. This is useful as a reference point, but regularized models are more stable on one-hot encoded housing data.",
        "code": """
model = Pipeline([
    ("preprocessor", linear_preprocessor),
    ("model", LinearRegression()),
])
mean, std = rmsle_cv(model)
model.fit(X, y)
print(f"LinearRegression CV RMSLE = {mean:.5f} +/- {std:.5f}")
save_submission(model, "linearregression")
""",
    },
    "Ridge": {
        "filename": "02_Ridge.ipynb",
        "title": "Ridge Regression",
        "text": "Ridge adds L2 regularization, which stabilizes coefficients when features are correlated.",
        "code": """
model = Pipeline([
    ("preprocessor", linear_preprocessor),
    ("model", Ridge(alpha=10.0)),
])
mean, std = rmsle_cv(model)
model.fit(X, y)
print(f"Ridge CV RMSLE = {mean:.5f} +/- {std:.5f}")
save_submission(model, "ridge")
""",
    },
    "Lasso": {
        "filename": "03_Lasso.ipynb",
        "title": "Lasso Regression",
        "text": "Lasso adds L1 regularization and can push weak feature coefficients to zero.",
        "code": """
model = Pipeline([
    ("preprocessor", linear_preprocessor),
    ("model", Lasso(alpha=0.0005, max_iter=30000, random_state=RANDOM_STATE)),
])
mean, std = rmsle_cv(model)
model.fit(X, y)
print(f"Lasso CV RMSLE = {mean:.5f} +/- {std:.5f}")
save_submission(model, "lasso")
""",
    },
    "ElasticNet": {
        "filename": "04_ElasticNet.ipynb",
        "title": "ElasticNet",
        "text": "ElasticNet combines L1 feature selection and L2 shrinkage. It is one of the strongest single-model choices here.",
        "code": """
model = Pipeline([
    ("preprocessor", linear_preprocessor),
    ("model", ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=30000, random_state=RANDOM_STATE)),
])
mean, std = rmsle_cv(model)
model.fit(X, y)
print(f"ElasticNet CV RMSLE = {mean:.5f} +/- {std:.5f}")
save_submission(model, "elasticnet")
""",
    },
    "RandomForest": {
        "filename": "05_RandomForest.ipynb",
        "title": "RandomForest",
        "text": "RandomForest is a tree-based bagging model. It needs less scaling, so the tree preprocessor skips StandardScaler.",
        "code": """
model = Pipeline([
    ("preprocessor", tree_preprocessor),
    ("model", RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_leaf=1, random_state=RANDOM_STATE, n_jobs=1)),
])
mean, std = rmsle_cv(model)
model.fit(X, y)
print(f"RandomForest CV RMSLE = {mean:.5f} +/- {std:.5f}")
save_submission(model, "randomforest")
""",
    },
    "SVR_RBF": {
        "filename": "06_SVR_RBF.ipynb",
        "title": "SVR RBF",
        "text": "Support Vector Regression with an RBF kernel models nonlinear relationships after scaling and one-hot encoding.",
        "code": """
model = Pipeline([
    ("preprocessor", linear_preprocessor),
    ("model", SVR(kernel="rbf", C=5, epsilon=0.01, gamma="scale")),
])
mean, std = rmsle_cv(model)
model.fit(X, y)
print(f"SVR RBF CV RMSLE = {mean:.5f} +/- {std:.5f}")
save_submission(model, "svr_rbf")
""",
    },
    "MLPRegressor": {
        "filename": "07_MLPRegressor.ipynb",
        "title": "MLPRegressor",
        "text": "MLPRegressor is a neural-network baseline for tabular regression. It is included for Session 3 comparison.",
        "code": """
model = Pipeline([
    ("preprocessor", linear_preprocessor),
    ("model", MLPRegressor(hidden_layer_sizes=(64,), alpha=0.001, max_iter=300, early_stopping=True, random_state=RANDOM_STATE)),
])
mean, std = rmsle_cv(model)
model.fit(X, y)
print(f"MLPRegressor CV RMSLE = {mean:.5f} +/- {std:.5f}")
save_submission(model, "mlpregressor")
""",
    },
    "KernelRidge_RBF": {
        "filename": "08_KernelRidge_RBF.ipynb",
        "title": "KernelRidge RBF",
        "text": "Kernel Ridge combines ridge regression with an RBF kernel. It is useful as another nonlinear baseline.",
        "code": """
model = Pipeline([
    ("preprocessor", linear_preprocessor),
    ("model", KernelRidge(kernel="rbf", alpha=0.1, gamma=0.001)),
])
mean, std = rmsle_cv(model)
model.fit(X, y)
print(f"KernelRidge RBF CV RMSLE = {mean:.5f} +/- {std:.5f}")
save_submission(model, "kernelridge_rbf")
""",
    },
    "CatBoost": {
        "filename": "09_CatBoost.ipynb",
        "title": "CatBoost",
        "text": "CatBoost handles categorical variables natively. If CatBoost or libomp is unavailable, run the other notebooks and use ElasticNet or Stacking.",
        "code": """
from catboost import CatBoostRegressor

X_cb = X.copy()
test_X_cb = test_X.copy()
X_cb[categorical_features] = X_cb[categorical_features].fillna("missing")
test_X_cb[categorical_features] = test_X_cb[categorical_features].fillna("missing")

model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    random_state=RANDOM_STATE,
    verbose=0,
    thread_count=1,
    allow_writing_files=False,
)

scores = []
for train_idx, valid_idx in cv.split(X_cb):
    fold_model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        random_state=RANDOM_STATE,
        verbose=0,
        thread_count=1,
        allow_writing_files=False,
    )
    fold_model.fit(X_cb.iloc[train_idx], y.iloc[train_idx], cat_features=categorical_features)
    pred = fold_model.predict(X_cb.iloc[valid_idx])
    scores.append(np.sqrt(np.mean((pred - y.iloc[valid_idx]) ** 2)))

model.fit(X_cb, y, cat_features=categorical_features)
print(f"CatBoost CV RMSLE = {np.mean(scores):.5f} +/- {np.std(scores):.5f}")
log_pred = model.predict(test_X_cb)
submission = pd.DataFrame({"Id": test["Id"], "SalePrice": np.exp(log_pred).clip(min=1)})
path = OUTPUT_DIR / "submission_catboost.csv"
submission.to_csv(path, index=False)
path, submission.head()
""",
    },
    "StackingRegressor": {
        "filename": "10_StackingRegressor.ipynb",
        "title": "StackingRegressor",
        "text": "Stacking combines several strong regularized/nonlinear base models and uses Ridge as the final estimator.",
        "code": """
base_models = [
    ("ridge", Pipeline([("preprocessor", linear_preprocessor), ("model", Ridge(alpha=10.0))])),
    ("lasso", Pipeline([("preprocessor", linear_preprocessor), ("model", Lasso(alpha=0.0005, max_iter=30000, random_state=RANDOM_STATE))])),
    ("elasticnet", Pipeline([("preprocessor", linear_preprocessor), ("model", ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=30000, random_state=RANDOM_STATE))])),
]

model = StackingRegressor(
    estimators=base_models,
    final_estimator=Ridge(alpha=10.0),
    cv=5,
    n_jobs=1,
)
mean, std = rmsle_cv(model)
model.fit(X, y)
print(f"StackingRegressor CV RMSLE = {mean:.5f} +/- {std:.5f}")
save_submission(model, "stackingregressor")
""",
    },
    "WeightedBlend": {
        "filename": "11_WeightedBlend.ipynb",
        "title": "Weighted Blend",
        "text": "A simple ensemble that averages Ridge, Lasso, and ElasticNet predictions in log-price space.",
        "code": """
models = [
    Pipeline([("preprocessor", linear_preprocessor), ("model", Ridge(alpha=10.0))]),
    Pipeline([("preprocessor", linear_preprocessor), ("model", Lasso(alpha=0.0005, max_iter=30000, random_state=RANDOM_STATE))]),
    Pipeline([("preprocessor", linear_preprocessor), ("model", ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=30000, random_state=RANDOM_STATE))]),
]

for model in models:
    model.fit(X, y)

log_pred = np.mean([model.predict(test_X) for model in models], axis=0)
submission = pd.DataFrame({"Id": test["Id"], "SalePrice": np.exp(log_pred).clip(min=1)})
path = OUTPUT_DIR / "submission_weighted_blend.csv"
submission.to_csv(path, index=False)
path, submission.head()
""",
    },
}


BUSINESS = notebook(
    [
        md(
            """
            # Business Startup Deliverable - FairPrice AI

            This notebook contains the business part of the IIIT Bangalore x Aivancity House Prices project.
            """
        ),
        md(
            """
            ## Concept

            **FairPrice AI** is a buyer-facing property valuation assistant. A user enters listing details or pastes a listing summary, and the product returns a predicted fair sale price, a verdict of underpriced/fair/overpriced, a confidence band, and the main value drivers.

            The trained House Prices model is the AI engine. The product turns model predictions into practical buying and negotiation guidance.
            """
        ),
        md(
            """
            ## Business Model Canvas

            | Block | Plan |
            | --- | --- |
            | Customer segments | First-time buyers, relocating professionals, buyer agents, mortgage pre-approval advisors |
            | Value proposition | Reduce overpayment risk with an independent fair-value estimate and clear explanation |
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
            | Zillow Zestimate | Strong brand and data scale | Black-box perception, US-focused | Explainable buyer assistant with transparent drivers |
            | Redfin Estimate | Integrated with brokerage workflow | Limited to supported markets | Independent tool that works across portals |
            | Realtor.com calculators | Simple consumer reach | Limited interpretability | Converts valuation into negotiation guidance |

            **Unique positioning:** explainable fair-value intelligence for buyers before they make an offer.
            """
        ),
        md(
            """
            ## PESTEL Analysis

            | Factor | Implications |
            | --- | --- |
            | Political | Housing affordability is sensitive; transparency can build public trust |
            | Economic | Higher rates increase buyer caution and demand for valuation confidence |
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
            - Add human review for lending or other high-stakes use cases.
            - Publish plain-language model documentation.
            - Comply with privacy rules such as GDPR where user data is collected.
            """
        ),
    ]
)


def model_notebook(name: str, spec: dict[str, str]) -> dict:
    return notebook(
        [
            md(f"# {spec['title']}\n\n{spec['text']}"),
            md("## Shared Setup"),
            code(SETUP),
            md("## Load, Clean, Feature Engineer, and Build Pipelines"),
            code(COMMON_DATA),
            md("## Train, Evaluate, and Generate Submission"),
            code(spec["code"]),
        ]
    )


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    old_combined = OUT_DIR / "House_Prices_AI_Project_Submission.ipynb"
    if old_combined.exists():
        old_combined.unlink()

    (OUT_DIR / "00_Business_Startup_Deliverable.ipynb").write_text(json.dumps(BUSINESS, indent=1), encoding="utf-8")
    for name, spec in MODELS.items():
        path = OUT_DIR / spec["filename"]
        path.write_text(json.dumps(model_notebook(name, spec), indent=1), encoding="utf-8")
        print(path)

    print(OUT_DIR / "00_Business_Startup_Deliverable.ipynb")


if __name__ == "__main__":
    main()
