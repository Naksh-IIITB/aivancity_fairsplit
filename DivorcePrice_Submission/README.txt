DivorcePrice — Capstone Submission
===================================

Course      : House Price Prediction Startup Project
Programme   : IIIT Bangalore × Aivancity
Instructor  : Pr Fayçal Braham
Date        : 2026-06-04

Files in this submission
-------------------------

1. DivorcePrice_Pitch.pptx   — 13-slide pitch deck (16:9)
                               problem · solution · unit economics ·
                               business model · markets · competitors ·
                               SWOT · PESTEL · ethics & EU AI Act ·
                               roadmap · ask · project code

2. DivorcePrice_Project.zip  — working Python project
                               ElasticNet + CatBoost ensemble
                               SHAP explainer
                               court-admissible PDF report generator
                               unit test (passing)


How to run the code
-------------------

    unzip DivorcePrice_Project.zip
    cd divorceprice
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

    # Train (uses a built-in synthetic dataset — no Kaggle file required):
    python -m scripts.train --synthetic --out artifacts/

    # Score one property and produce the court-admissible PDF:
    python -m scripts.predict \
        --artifacts artifacts/ \
        --row data/raw/sample_property.json \
        --case-id "Doe v. Doe — 2026-FR-1142" \
        --jurisdiction France \
        --address "12 rue de la Paix, 75002 Paris" \
        --out reports/doe_v_doe.pdf

    # Run tests:
    PYTHONPATH=src python -m pytest tests/ -v


To reproduce the deck's headline number (CV RMSLE 0.112)
--------------------------------------------------------

Place the Kaggle "House Prices - Advanced Regression Techniques"
train.csv at  data/raw/train.csv  and run:

    python -m scripts.train --data data/raw/train.csv --cv-folds 5


Verified on synthetic data
--------------------------

CV RMSLE — ElasticNet : 0.0344
CV RMSLE — CatBoost   : 0.0511
CV RMSLE — Ensemble   : 0.0343   (blend = 0.95 linear / 0.05 tree)
Test suite            : 1 passed in 2.65s
