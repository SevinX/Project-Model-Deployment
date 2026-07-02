"""random_forest_model.py -- persis hyperparameter notebook section 4.2."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier

from .base_model import BaseModelPipeline


class RandomForestModel(BaseModelPipeline):
    model_name = "Random_Forest"

    def build_estimator(self) -> Any:
        params = {
            "n_estimators": 200,
            "max_depth": 12,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
            **self.hyperparams,
        }
        return RandomForestClassifier(**params)
