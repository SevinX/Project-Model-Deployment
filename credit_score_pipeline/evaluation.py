"""
evaluation.py
=============
Kelas evaluasi terpusat, dipakai oleh setiap model (lihat models/base_model.py).
Replikasi notebook section 4.2 (metrik test set) dan 4.4 (5-fold CV).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .logging_utils import get_logger

logger = get_logger("evaluation")


class Evaluator:
    """Menghitung metrik evaluasi klasifikasi multi-kelas untuk sebuah model
    yang sudah terlatih (fitted sklearn Pipeline / estimator).
    """

    def __init__(self, cv_folds: int = 5, scoring: str = "f1_macro", random_state: int = 42) -> None:
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state

    def evaluate_test_set(self, fitted_pipeline, X_test: pd.DataFrame, y_test: np.ndarray,
                           label_names: List[str] | None = None) -> Dict[str, float]:
        """Metrik pada test set: accuracy, macro F1, macro precision, macro recall."""
        y_pred = fitted_pipeline.predict(X_test)
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "macro_f1": round(float(f1_score(y_test, y_pred, average="macro")), 4),
            "macro_precision": round(float(precision_score(y_test, y_pred, average="macro")), 4),
            "macro_recall": round(float(recall_score(y_test, y_pred, average="macro")), 4),
        }
        logger.info(f"Evaluasi test set -> {metrics}")

        if label_names is not None:
            report = classification_report(y_test, y_pred, target_names=label_names)
            logger.info(f"Classification report:\n{report}")

        return metrics

    def confusion_matrix(self, fitted_pipeline, X_test: pd.DataFrame, y_test: np.ndarray) -> np.ndarray:
        y_pred = fitted_pipeline.predict(X_test)
        return confusion_matrix(y_test, y_pred)

    def cross_validate(self, estimator, X_train: pd.DataFrame, y_train: np.ndarray) -> Dict[str, float]:
        """5-fold Stratified CV macro F1 (dipakai untuk membandingkan model,
        replikasi notebook section 4.4). `estimator` di sini adalah full
        Pipeline (preprocessor + classifier) yang BELUM di-fit -- cross_val_score
        akan fit ulang preprocessor di setiap fold secara otomatis, sehingga
        aman dari leakage.
        """
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring=self.scoring, n_jobs=-1)
        result = {"cv_mean": round(float(scores.mean()), 4), "cv_std": round(float(scores.std()), 4)}
        logger.info(f"5-Fold CV ({self.scoring}) -> mean={result['cv_mean']}, std={result['cv_std']}")
        return result
