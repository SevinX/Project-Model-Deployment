"""
tuning.py
=========
Hyperparameter tuning XGBoost dengan Optuna. Replikasi persis notebook
section "Eksperimen" (search space + skema CV) -- termasuk prinsip
"cloning dan fitting preprocessor secara terpisah tiap fold" untuk mencegah
data leakage, yang di pipeline ini diperluas: bukan cuma FallbackImputer +
ColumnTransformer yang di-clone per fold, tapi SELURUH preprocessing
pipeline (raw_cleaner -> outlier_clipper -> fallback_imputer ->
feature_engineer -> column_transformer), karena semuanya sekarang fit-safe.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .logging_utils import get_logger

logger = get_logger("tuning")


class OptunaTuner:
    """Tuning XGBoost dengan Optuna TPE sampler, objective = mean macro F1
    across 5-fold Stratified CV (skala training set saja).
    """

    def __init__(self, preprocessor: Pipeline, n_trials: int = 150, cv_folds: int = 5,
                 random_state: int = 42, study_name: str = "xgboost_optuna_macro_f1") -> None:
        self.base_preprocessor = preprocessor
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.study_name = study_name
        self.study: Optional[optuna.Study] = None

    @staticmethod
    def _search_space(trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
            "early_stopping_rounds": 30,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }

    def _objective(self, trial: optuna.Trial, X_train: pd.DataFrame, y_train: np.ndarray) -> float:
        param = self._search_space(trial)

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        fold_scores = []

        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            fold_preprocessor = clone(self.base_preprocessor)
            X_tr_proc = fold_preprocessor.fit_transform(X_tr)
            X_val_proc = fold_preprocessor.transform(X_val)

            weights = compute_sample_weight("balanced", y_tr)
            model = XGBClassifier(**param)
            model.fit(
                X_tr_proc, y_tr,
                sample_weight=weights,
                eval_set=[(X_val_proc, y_val)],
                verbose=False,
            )

            y_pred = model.predict(X_val_proc)
            fold_scores.append(f1_score(y_val, y_pred, average="macro"))

        return float(np.mean(fold_scores))

    def tune(self, X_train: pd.DataFrame, y_train: np.ndarray) -> optuna.Study:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.study = optuna.create_study(
            direction="maximize",
            study_name=self.study_name,
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        logger.info(f"Mulai Optuna tuning -> {self.n_trials} trials, {self.cv_folds}-fold CV")
        self.study.optimize(
            lambda trial: self._objective(trial, X_train, y_train),
            n_trials=self.n_trials,
            show_progress_bar=True,
        )
        logger.info(f"Tuning selesai -> best CV macro_f1={self.study.best_value:.4f}")
        logger.info(f"Best params: {self.study.best_params}")
        return self.study

    def build_best_estimator(self) -> XGBClassifier:
        """Estimator XGBoost final dengan best_params + parameter tetap.
        TIDAK menyertakan `early_stopping_rounds` (dan tanpa eval_set saat
        fit final) -- persis notebook section 'Training Model Ulang'.
        """
        if self.study is None:
            raise RuntimeError("Panggil .tune() terlebih dahulu sebelum build_best_estimator().")
        return XGBClassifier(
            **self.study.best_params,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
