# FairSplit

AI-powered neutral property valuation for divorce disputes.

ElasticNet + CatBoost ensemble, SHAP-explained, with a court-admissible PDF report
co-signable by a certified appraiser.

> Capstone project — IIIT Bangalore × Aivancity. Instructor: Pr Fayçal Braham.

## What this repo contains

- **`src/divorceprice/`** — library code
  - `data.py` — load the Kaggle Ames House Prices dataset (with synthetic fallback for smoke tests)
  - `preprocessing.py` — impute / scale / one-hot pipeline
  - `models.py` — ElasticNet, CatBoost, weighted ensemble (CV-tuned)
  - `metrics.py` — RMSLE + cross-validated scoring helpers
  - `explain.py` — SHAP values formatted for the legal report
  - `report.py` — court-admissible PDF generator
  - `pipeline.py` — train / predict orchestration
- **`scripts/train.py`** — fit the ensemble, save artifacts, print CV RMSLE
- **`scripts/predict.py`** — value a single property, write the PDF report

## Quick start

```bash
# 1. Install
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Get the data
#    Place Kaggle "House Prices - Advanced Regression Techniques" train.csv in:
#      data/raw/train.csv
#    (https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

# 3. Train
python -m scripts.train --data data/raw/train.csv --out artifacts/

# 4. Score one property + emit court-admissible PDF
python -m scripts.predict \
    --artifacts artifacts/ \
    --row data/raw/sample_property.json \
    --case-id "Doe v. Doe — 2026-FR-1142" \
    --out reports/doe_v_doe.pdf
```

The CLI also runs end-to-end against a synthetic dataset (no Kaggle file required) — see
`--synthetic` flag on `train.py`. This is what the unit tests use.

## Reproducing the deck's headline number

The deck quotes **CV RMSLE 0.112** for the ensemble on the Kaggle benchmark. Re-run with:

```bash
python -m scripts.train --data data/raw/train.csv --cv-folds 5
```

The CLI prints per-model RMSLE, the ensemble blend weights, and the CV mean ± std.

## Disclaimer

This is academic / pre-production code. The PDF says so. Production deployment
requires legal review per jurisdiction (France: expert judiciaire registration;
India: IBBI valuer co-signature; EU AI Act high-risk registration).
