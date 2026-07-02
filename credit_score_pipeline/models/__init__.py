from .base_model import BaseModelPipeline
from .logistic_regression_model import LogisticRegressionModel
from .random_forest_model import RandomForestModel
from .xgboost_model import XGBoostModel

__all__ = [
    "BaseModelPipeline",
    "LogisticRegressionModel",
    "RandomForestModel",
    "XGBoostModel",
]

# Registry supaya TrainingPipeline bisa loop tanpa if/elif panjang
MODEL_REGISTRY = {
    "Logistic_Regression": LogisticRegressionModel,
    "Random_Forest": RandomForestModel,
    "XGBoost": XGBoostModel,
}
