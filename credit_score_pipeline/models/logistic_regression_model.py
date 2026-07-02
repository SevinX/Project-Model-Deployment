"""logistic_regression_model.py -- persis hyperparameter notebook section 4.2."""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression

from .base_model import BaseModelPipeline


class LogisticRegressionModel(BaseModelPipeline):
    model_name = "Logistic_Regression"

    def build_estimator(self) -> Any:
        params = {
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": 42,
            "solver": "lbfgs",
            **self.hyperparams,
        }
        return LogisticRegression(**params)
