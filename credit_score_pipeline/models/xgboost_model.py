"""
xgboost_model.py
================
Persis hyperparameter & cara fit notebook section 4.2.

Catatan penting: XGBClassifier TIDAK punya parameter `class_weight` seperti
LogisticRegression/RandomForest. Balancing kelas dilakukan lewat
`sample_weight` yang dihitung dengan `compute_sample_weight('balanced', y)`
lalu dilewatkan ke `.fit()` -- karena itu class ini meng-override `_fit()`.

Class ini juga dipakai ulang untuk XGBoost hasil tuning Optuna: cukup
instantiate dengan `hyperparams=study.best_params` (lihat training_pipeline.py).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .base_model import BaseModelPipeline


class XGBoostModel(BaseModelPipeline):
    model_name = "XGBoost"

    def build_estimator(self) -> Any:
        params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "eval_metric": "mlogloss",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
            **self.hyperparams,
        }
        return XGBClassifier(**params)

    def _fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> None:
        sample_weight = compute_sample_weight("balanced", y_train)
        self.pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weight)
