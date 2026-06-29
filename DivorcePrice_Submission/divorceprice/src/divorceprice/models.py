"""ElasticNet + CatBoost ensemble.

Both models are trained in log-target space (log1p). The blend weight is
chosen by grid search on out-of-fold predictions to minimise RMSLE.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from .config import TrainConfig
from .metrics import rmsle_from_log
from .preprocessing import (
    TreeFrame,
    build_linear_preprocessor,
    expm1_target,
    log1p_target,
    split_columns,
    to_tree_frame,
)


@dataclass
class FittedEnsemble:
    """Everything we need to predict + audit a single property."""
    linear: Pipeline
    catboost: CatBoostRegressor
    blend: float                 # weight on linear; (1-blend) on catboost
    feature_names: list[str]     # names AFTER linear preprocessing (for SHAP)
    num_cols: list[str]
    cat_cols: list[str]
    cv_rmsle_linear: float
    cv_rmsle_catboost: float
    cv_rmsle_blend: float
    cv_folds: int

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        lin_log = self.linear.predict(df)
        tree = to_tree_frame(df, self.num_cols, self.cat_cols)
        cat_log = self.catboost.predict(Pool(tree.X, cat_features=tree.cat_idx))
        blend_log = self.blend * lin_log + (1 - self.blend) * cat_log
        return expm1_target(blend_log)


def _fit_linear(X: pd.DataFrame, y_log: np.ndarray, cfg: TrainConfig,
                num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    pre = build_linear_preprocessor(num_cols, cat_cols)
    model = ElasticNetCV(
        alphas=cfg.elasticnet_alpha_grid,
        l1_ratio=list(cfg.elasticnet_l1_grid),
        cv=cfg.cv_folds,
        random_state=cfg.random_state,
        max_iter=20_000,
        n_jobs=-1,
    )
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X, y_log)
    return pipe


def _fit_catboost(tree: TreeFrame, y_log: np.ndarray, cfg: TrainConfig) -> CatBoostRegressor:
    model = CatBoostRegressor(
        iterations=cfg.catboost_iterations,
        depth=cfg.catboost_depth,
        learning_rate=cfg.catboost_lr,
        loss_function="RMSE",
        random_seed=cfg.random_state,
        cat_features=tree.cat_idx,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(Pool(tree.X, label=y_log, cat_features=tree.cat_idx))
    return model


def _oof_predictions(X: pd.DataFrame, y_log: np.ndarray, cfg: TrainConfig,
                     num_cols: list[str], cat_cols: list[str]
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold log-space predictions for both base models."""
    kf = KFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
    oof_lin = np.zeros_like(y_log, dtype=float)
    oof_cat = np.zeros_like(y_log, dtype=float)

    for fold, (tr, va) in enumerate(kf.split(X), 1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr = y_log[tr]

        # Linear branch — full pipeline so it preprocesses inside the fold.
        pre = build_linear_preprocessor(num_cols, cat_cols)
        from sklearn.linear_model import ElasticNetCV
        lin = Pipeline([
            ("pre", pre),
            ("model", ElasticNetCV(
                alphas=cfg.elasticnet_alpha_grid,
                l1_ratio=list(cfg.elasticnet_l1_grid),
                cv=3,  # inner CV for alpha pick — small to keep folds fast
                random_state=cfg.random_state,
                max_iter=20_000,
                n_jobs=-1,
            )),
        ])
        lin.fit(X_tr, y_tr)
        oof_lin[va] = lin.predict(X_va)

        # CatBoost branch
        tree_tr = to_tree_frame(X_tr, num_cols, cat_cols)
        cat = CatBoostRegressor(
            iterations=cfg.catboost_iterations,
            depth=cfg.catboost_depth,
            learning_rate=cfg.catboost_lr,
            loss_function="RMSE",
            random_seed=cfg.random_state,
            cat_features=tree_tr.cat_idx,
            verbose=False,
            allow_writing_files=False,
        )
        cat.fit(Pool(tree_tr.X, label=y_tr, cat_features=tree_tr.cat_idx))
        tree_va = to_tree_frame(X_va, num_cols, cat_cols)
        oof_cat[va] = cat.predict(Pool(tree_va.X, cat_features=tree_va.cat_idx))

    return oof_lin, oof_cat


def _pick_blend(oof_lin: np.ndarray, oof_cat: np.ndarray, y_log: np.ndarray,
                grid) -> tuple[float, float]:
    """Grid-search the blend weight on OOF preds. Return (best_w, best_rmsle)."""
    best_w, best = 0.5, float("inf")
    for w in grid:
        score = rmsle_from_log(y_log, w * oof_lin + (1 - w) * oof_cat)
        if score < best:
            best, best_w = score, float(w)
    return best_w, best


def fit_ensemble(df: pd.DataFrame, cfg: TrainConfig | None = None) -> FittedEnsemble:
    cfg = cfg or TrainConfig()
    num_cols, cat_cols = split_columns(df)
    X = df[num_cols + cat_cols]
    y_log = log1p_target(df["SalePrice"])

    oof_lin, oof_cat = _oof_predictions(X, y_log, cfg, num_cols, cat_cols)
    rmsle_lin = rmsle_from_log(y_log, oof_lin)
    rmsle_cat = rmsle_from_log(y_log, oof_cat)
    blend, rmsle_blend = _pick_blend(oof_lin, oof_cat, y_log, cfg.blend_weight_grid)

    # Refit both models on the full training set with the chosen settings.
    linear = _fit_linear(X, y_log, cfg, num_cols, cat_cols)
    tree = to_tree_frame(X, num_cols, cat_cols)
    catb = _fit_catboost(tree, y_log, cfg)

    feature_names = list(linear.named_steps["pre"].get_feature_names_out())

    return FittedEnsemble(
        linear=linear,
        catboost=catb,
        blend=blend,
        feature_names=feature_names,
        num_cols=num_cols,
        cat_cols=cat_cols,
        cv_rmsle_linear=rmsle_lin,
        cv_rmsle_catboost=rmsle_cat,
        cv_rmsle_blend=rmsle_blend,
        cv_folds=cfg.cv_folds,
    )
