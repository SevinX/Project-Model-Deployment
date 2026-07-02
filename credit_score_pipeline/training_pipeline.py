"""
training_pipeline.py
=====================
Orchestrator: menyatukan DataLoader -> 3 model baseline -> Optuna tuning
XGBoost -> MLflow logging -> pemilihan & registrasi model terbaik ->
penyimpanan artifact lokal (best_model.pkl, label_encoder.pkl, model_meta.pkl).

Didesain agar mudah dipanggil ulang untuk retraining:
    TrainingPipeline(config).run()
setiap pemanggilan = 1 percobaan baru yang tercatat rapi sebagai run-run baru
di MLflow (nothing di-hardcode dari run sebelumnya).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict

from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from .data_loader import DataLoader
from .evaluation import Evaluator
from .logging_utils import get_logger, setup_logging
from .models import MODEL_REGISTRY
from .preprocessing import build_preprocessing_pipeline
from .tracking import MLflowTracker
from .tuning import OptunaTuner

logger = get_logger("training_pipeline")


class TrainingPipeline:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        setup_logging(config["artifacts"]["logs_dir"])

        self.numeric_features = config["features"]["numeric"]
        self.categorical_features = config["features"]["categorical"]

        self.data_loader = DataLoader(
            target_column=config["data"]["target_column"],
            test_size=config["data"]["test_size"],
            random_state=config["data"]["random_state"],
        )
        self.evaluator = Evaluator(
            cv_folds=config["evaluation"]["cv_folds"],
            scoring=config["evaluation"]["scoring"],
            random_state=config["data"]["random_state"],
        )
        self.tracker = MLflowTracker(
            tracking_uri=config["mlflow"]["tracking_uri"],
            experiment_name=config["mlflow"]["experiment_name"],
        )

        self.results: Dict[str, Dict[str, float]] = {}
        self.run_ids: Dict[str, str] = {}
        self.model_uris: Dict[str, str] = {}
        self.fitted_pipelines: Dict[str, Pipeline] = {}

    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        X_train, X_test, y_train, y_test, label_names = self._load_and_split()

        self._train_baseline_models(X_train, X_test, y_train, y_test, label_names)

        if self.config["tuning"]["enabled"]:
            self._tune_and_train_xgboost_optuna(X_train, X_test, y_train, y_test, label_names)

        best_name = self._select_best_model()
        self._register_champion(best_name)
        self._save_local_artifacts(best_name, label_names)

        logger.info("=== Pipeline selesai ===")
        logger.info(f"Ringkasan hasil:\n{self._results_table()}")

        return {
            "best_model_name": best_name,
            "results": self.results,
            "artifacts_dir": self.config["artifacts"]["dir"],
        }

    # ------------------------------------------------------------------
    def _load_and_split(self):
        df = self.data_loader.load_raw(self.config["data"]["raw_path"])
        X_train, X_test, y_train, y_test = self.data_loader.split(df)
        label_names = list(self.data_loader.label_encoder.classes_)
        return X_train, X_test, y_train, y_test, label_names

    # ------------------------------------------------------------------
    def _train_baseline_models(self, X_train, X_test, y_train, y_test, label_names) -> None:
        for name, model_cls in MODEL_REGISTRY.items():
            hyperparams = self.config["models"].get(self._to_config_key(name), {})
            model = model_cls(
                numeric_features=self.numeric_features,
                categorical_features=self.categorical_features,
                hyperparams={},  # default notebook hyperparams sudah di dalam tiap class
                evaluator=self.evaluator,
            )
            model.train(X_train, y_train)
            metrics = model.evaluate(X_test, y_test, label_names)
            cv_result = model.cross_validate(X_train, y_train)
            logger.info(f"[{name}] Test metrics={metrics} | CV={cv_result}")

            log_result = self.tracker.log_model_run(
                run_name=name,
                params={"model_type": name, "test_size": self.config["data"]["test_size"],
                        **cv_result},
                metrics=metrics,
                fitted_pipeline=model.pipeline,
                input_example=X_train,
            )

            self.results[name] = metrics
            self.run_ids[name] = log_result["run_id"]
            self.model_uris[name] = log_result["model_uri"]
            self.fitted_pipelines[name] = model.pipeline

    # ------------------------------------------------------------------
    def _tune_and_train_xgboost_optuna(self, X_train, X_test, y_train, y_test, label_names) -> None:
        tuning_cfg = self.config["tuning"]
        base_preprocessor = build_preprocessing_pipeline(self.numeric_features, self.categorical_features)

        tuner = OptunaTuner(
            preprocessor=base_preprocessor,
            n_trials=tuning_cfg["n_trials"],
            cv_folds=tuning_cfg["cv_folds"],
            random_state=tuning_cfg["random_state"],
            study_name=tuning_cfg["study_name"],
        )
        tuner.tune(X_train, y_train)
        best_xgb = tuner.build_best_estimator()

        final_pipeline = Pipeline([
            ("preprocessor", build_preprocessing_pipeline(self.numeric_features, self.categorical_features)),
            ("classifier", best_xgb),
        ])
        sample_weight = compute_sample_weight("balanced", y_train)
        final_pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weight)

        metrics = self.evaluator.evaluate_test_set(final_pipeline, X_test, y_test, label_names)
        logger.info(f"[XGBoost_Optuna] Test metrics={metrics}")

        log_result = self.tracker.log_model_run(
            run_name="XGBoost_Optuna",
            params={"model_type": "XGBoost_Optuna", "test_size": self.config["data"]["test_size"],
                    "n_trials": tuning_cfg["n_trials"], "best_cv_macro_f1": round(tuner.study.best_value, 4),
                    **tuner.study.best_params},
            metrics=metrics,
            fitted_pipeline=final_pipeline,
            input_example=X_train,
        )

        self.results["XGBoost_Optuna"] = metrics
        self.run_ids["XGBoost_Optuna"] = log_result["run_id"]
        self.model_uris["XGBoost_Optuna"] = log_result["model_uri"]
        self.fitted_pipelines["XGBoost_Optuna"] = final_pipeline

    # ------------------------------------------------------------------
    def _select_best_model(self) -> str:
        best_name = max(self.results, key=lambda k: self.results[k]["macro_f1"])
        logger.info(f"Model terbaik: {best_name} | macro_f1={self.results[best_name]['macro_f1']}")
        return best_name

    def _register_champion(self, best_name: str) -> None:
        self.tracker.register_champion(
            model_uri=self.model_uris[best_name],
            registered_model_name=self.config["mlflow"]["registered_model_name"],
            alias=self.config["mlflow"]["champion_alias"],
        )

    def _save_local_artifacts(self, best_name: str, label_names) -> None:
        artifacts_dir = Path(self.config["artifacts"]["dir"])
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        with open(artifacts_dir / "best_model.pkl", "wb") as f:
            pickle.dump(self.fitted_pipelines[best_name], f)
        with open(artifacts_dir / "label_encoder.pkl", "wb") as f:
            pickle.dump(self.data_loader.label_encoder, f)
        with open(artifacts_dir / "model_meta.pkl", "wb") as f:
            pickle.dump({
                "best_model_name": best_name,
                "best_macro_f1": self.results[best_name]["macro_f1"],
                "all_results": self.results,
                "numeric_features": self.numeric_features,
                "categorical_features": self.categorical_features,
                "label_classes": label_names,
            }, f)

        logger.info(f"Artifact tersimpan di '{artifacts_dir}/' "
                    f"(best_model.pkl, label_encoder.pkl, model_meta.pkl)")

    # ------------------------------------------------------------------
    @staticmethod
    def _to_config_key(model_name: str) -> str:
        return model_name.lower()

    def _results_table(self) -> str:
        lines = [f"{'Model':<22}{'Accuracy':>10}{'Macro F1':>10}{'Precision':>11}{'Recall':>9}"]
        for name, m in self.results.items():
            lines.append(
                f"{name:<22}{m['accuracy']:>10.4f}{m['macro_f1']:>10.4f}"
                f"{m['macro_precision']:>11.4f}{m['macro_recall']:>9.4f}"
            )
        return "\n".join(lines)
