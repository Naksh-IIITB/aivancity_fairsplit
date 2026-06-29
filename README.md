# House Prices AI Startup Project

Complete technical project for the IIIT Bangalore x Aivancity hands-on AI workshop using Kaggle's **House Prices - Advanced Regression Techniques** competition.

## What is included

- End-to-end tabular regression workflow: EDA, preprocessing, cross-validation, tuning, predictions, and submissions.
- Linear models: LinearRegression, Ridge, Lasso, ElasticNet.
- Tree and boosting models: RandomForest, optional XGBoost, LightGBM, CatBoost.
- Session 3 extensions: SVR, MLPRegressor, weighted blending, stacking, feature engineering, and optional SHAP explainability.
- Startup deliverable: business model canvas, competitors, positioning, PESTEL, SWOT, ethics, governance, and regulation notes.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the Kaggle files from:

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

Place these files in `data/`:

- `train.csv`
- `test.csv`
- `sample_submission.csv`
- `data_description.txt`

## Run the full workflow

```bash
python src/house_price_project.py --data-dir data --output-dir outputs --models-dir models --fast
```

Remove `--fast` for broader grids:

```bash
python src/house_price_project.py --data-dir data --output-dir outputs --models-dir models
```

Generated files include:

- `outputs/model_comparison.csv`
- `outputs/submission_elasticnet.csv`
- `outputs/submission_catboost.csv` if CatBoost is installed
- `outputs/submission_weighted_blend.csv`
- `outputs/submission_stackingregressor.csv`
- `models/best_model.joblib`
- EDA plots in `outputs/figures/`

## Notebook

Open `notebooks/house_prices_workflow.py` in Jupyter or VS Code. It is a percent-format notebook with runnable cells and mirrors the workshop deliverables.

## Startup concept

The included business deliverable uses **FairPrice AI**: a buyer-facing valuation assistant that predicts fair property value, flags listings as underpriced/fair/overpriced, and explains which property features drive the estimate.
