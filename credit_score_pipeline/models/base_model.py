"""
base_model.py
=============
Abstract base class yang mewajibkan SETIAP model punya kemampuan
preprocessing, training, dan evaluation sendiri -- sesuai requirement UAS:

    "Code model machine learning ... harus berbasis OOP dengan minimal class
    mencakup preprocessing, training dan evaluation untuk setiap model
    machine learning yang anda training."

Desain: preprocessing (ColumnTransformer + custom transformer) IDENTIK untuk
ketiga model di notebook aslinya, jadi logic-nya ditaruh sekali di
`preprocessing.build_preprocessing_pipeline()` dan tiap subclass model
mewarisinya lewat `build_preprocessor()`. Ini menghindari copy-paste 3x
sekaligus tetap memenuhi requirement: setiap class model (LogisticRegressionModel,
RandomForestModel, XGBoostModel) MEMILIKI method preprocessing/training/
evaluation sendiri (lewat inheritance), dan tiap instance meng-handle
preprocessing-nya sendiri secara independen (di-fit ulang per model, persis
seperti pola `Pipeline([('preprocessor', ...), ('classifier', clf)])` yang
dipakai berulang di notebook untuk tiap model).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ..evaluation import Evaluator
from ..logging_utils import get_logger
from ..preprocessing import build_preprocessing_pipeline

logger = get_logger("models.base")


class BaseModelPipeline(ABC):
    """Kontrak dasar untuk setiap model: build_estimator -> preprocess -> train -> evaluate."""

    #: Nama model, dipakai sebagai run_name di MLflow & key hasil evaluasi.
    model_name: str = "base_model"

    def __init__(self, numeric_features: List[str], categorical_features: List[str],
                 hyperparams: Optional[Dict[str, Any]] = None, evaluator: Optional[Evaluator] = None) -> None:
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.hyperparams = hyperparams or {}
        self.evaluator = evaluator or Evaluator()

        self.preprocessor: Pipeline = self.build_preprocessor()
        self.estimator = self.build_estimator()
        self.pipeline: Optional[Pipeline] = None  # terisi setelah .train()
        self.metrics_: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # PREPROCESSING -- setiap model punya preprocessor sendiri (independen,
    # di-fit ulang saat train dipanggil), walau resep transformernya sama.
    # ------------------------------------------------------------------
    def build_preprocessor(self) -> Pipeline:
        return build_preprocessing_pipeline(self.numeric_features, self.categorical_features)

    # ------------------------------------------------------------------
    # Setiap subclass WAJIB mendefinisikan estimator & (opsional) override
    # cara fit jika butuh perlakuan khusus (mis. XGBoost butuh sample_weight).
    # ------------------------------------------------------------------
    @abstractmethod
    def build_estimator(self) -> Any:
        """Mengembalikan estimator sklearn/xgboost yang belum di-fit."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------
    def train(self, X_train: pd.DataFrame, y_train: np.ndarray) -> "BaseModelPipeline":
        logger.info(f"[{self.model_name}] Mulai training ...")
        self.pipeline = Pipeline([
            ("preprocessor", self.preprocessor),
            ("classifier", self.estimator),
        ])
        self._fit(X_train, y_train)
        logger.info(f"[{self.model_name}] Training selesai.")
        return self

    def _fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> None:
        """Default fit. Di-override oleh model yang butuh argumen fit khusus
        (mis. XGBoostModel dengan sample_weight balanced).
        """
        self.pipeline.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------------
    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray,
                 label_names: Optional[List[str]] = None) -> Dict[str, float]:
        if self.pipeline is None:
            raise RuntimeError(f"[{self.model_name}] Model belum di-train. Panggil .train() dahulu.")
        self.metrics_ = self.evaluator.evaluate_test_set(self.pipeline, X_test, y_test, label_names)
        return self.metrics_

    def cross_validate(self, X_train: pd.DataFrame, y_train: np.ndarray) -> Dict[str, float]:
        """CV memakai pipeline BARU (belum fit) agar preprocessor di-fit ulang
        di tiap fold oleh cross_val_score -- konsisten, tanpa leakage."""
        cv_pipeline = Pipeline([
            ("preprocessor", self.build_preprocessor()),
            ("classifier", self.build_estimator()),
        ])
        return self.evaluator.cross_validate(cv_pipeline, X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError(f"[{self.model_name}] Model belum di-train.")
        return self.pipeline.predict(X)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} model_name={self.model_name!r}>"
