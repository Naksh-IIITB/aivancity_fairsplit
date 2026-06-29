# %%
"""
# House Prices - Advanced Regression Techniques

IIIT Bangalore x Aivancity hands-on AI project.

This notebook-style file mirrors the PDF workflow:

1. EDA and missing-value audit
2. Log-target regression setup
3. sklearn preprocessing pipelines
4. Linear, tree, SVR, MLP, and ensemble models
5. Kaggle-ready submissions

Open this file in VS Code/Jupyter as a percent-format notebook, or run the
complete CLI workflow from the project root:

```bash
python src/house_price_project.py --data-dir data --output-dir outputs --models-dir models --fast
```
"""

# %%
from pathlib import Path

import numpy as np
import pandas as pd

from src.house_price_project import (
    add_features,
    clean_known_missing_values,
    column_lists,
    evaluate_linear_models,
    evaluate_session3_models,
    evaluate_tree_models,
    linear_preprocessor,
    load_data,
    make_submission,
    make_xy,
    remove_training_outliers,
    save_eda,
    target_encoder_preprocessor,
    tree_one_hot_preprocessor,
    weighted_blend_submission,
)

from sklearn.model_selection import KFold

# %%
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
MODELS_DIR = Path("models")
FAST = True
RANDOM_STATE = 42

OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# %%
train, test = load_data(DATA_DIR)
train, test = clean_known_missing_values(train, test)
train = remove_training_outliers(train)
train = add_features(train)
test = add_features(test)

train.shape, test.shape

# %%
save_eda(train, OUTPUT_DIR)

missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
pd.DataFrame(
    {
        "count": missing,
        "percent": (missing / len(train) * 100).round(1),
    }
).head(20)

# %%
num_cols_for_corr = train.select_dtypes(include=np.number).columns
train[num_cols_for_corr].corr(numeric_only=True)["SalePrice"].abs().sort_values(ascending=False).head(15)

# %%
X, y = make_xy(train)
num_cols, cat_cols = column_lists(X)
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

linear_prep = linear_preprocessor(num_cols, cat_cols)
tree_prep = tree_one_hot_preprocessor(num_cols, cat_cols)
target_prep = target_encoder_preprocessor(num_cols, cat_cols)

len(num_cols), len(cat_cols)

# %%
results = []
results.extend(evaluate_linear_models(X, y, linear_prep, cv, fast=FAST))

pd.DataFrame(
    {
        "model": [r.name for r in results],
        "cv_rmsle": [r.cv_rmsle_mean for r in results],
        "params": [r.best_params for r in results],
    }
).sort_values("cv_rmsle")

# %%
results.extend(
    evaluate_tree_models(
        X,
        y,
        tree_prep,
        target_prep,
        num_cols,
        cat_cols,
        cv,
        fast=FAST,
    )
)

pd.DataFrame(
    {
        "model": [r.name for r in results],
        "cv_rmsle": [r.cv_rmsle_mean for r in results],
        "params": [r.best_params for r in results],
    }
).sort_values("cv_rmsle")

# %%
results.extend(evaluate_session3_models(X, y, linear_prep, cv, fast=FAST))

comparison = pd.DataFrame(
    {
        "model": [r.name for r in results],
        "cv_rmsle": [r.cv_rmsle_mean for r in results],
        "params": [r.best_params for r in results],
    }
).sort_values("cv_rmsle")
comparison.to_csv(OUTPUT_DIR / "model_comparison_notebook.csv", index=False)
comparison

# %%
for result in results:
    if result.name == "CatBoost":
        make_submission(result.estimator, test, OUTPUT_DIR / "submission_catboost_notebook.csv", cat_cols)
    elif result.name == "ElasticNet":
        make_submission(result.estimator, test, OUTPUT_DIR / "submission_elasticnet_notebook.csv")

weighted_blend_submission(results, test, OUTPUT_DIR / "submission_weighted_blend_notebook.csv", cat_cols)

print("Notebook submissions saved in outputs/.")
