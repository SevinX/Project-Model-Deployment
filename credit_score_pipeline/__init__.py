"""
credit_score_pipeline
======================
Local ML training pipeline (OOP + MLflow) untuk kasus klasifikasi Credit_Score.
Direplikasi dari notebook EDA/Modelling, dengan penambahan agar seluruh transform
statistik (threshold outlier, median, mode) di-fit hanya pada data training --
sehingga pipeline ini aman dipakai ulang untuk retraining tanpa data leakage.
"""

__version__ = "1.0.0"
