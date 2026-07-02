"""
tracking.py
===========
Wrapper MLflow: experiment tracking (params/metrics/model per run) + model
registry (menandai model terbaik sebagai "champion").

Signature `mlflow.sklearn.log_model()` dan API alias (`set_registered_model_alias`)
di modul ini SUDAH diverifikasi langsung terhadap mlflow==3.14.0 yang terpasang
di environment ini (bukan ditulis dari ingatan) -- lihat requirements.txt.

Kenapa `serialization_format="cloudpickle"` di-set eksplisit?
Default MLflow 3.x untuk sklearn model adalah 'skops', yang butuh package
`skops` terpasang. cloudpickle adalah dependency inti MLflow sendiri (selalu
tersedia), jadi dipilih eksplisit agar tidak ada dependency tersembunyi.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature

from .logging_utils import get_logger

logger = get_logger("tracking")


class MLflowTracker:
    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()

        logger.info(f"MLflow tracking_uri={tracking_uri} | experiment='{experiment_name}'")

    def log_model_run(self, run_name: str, params: Dict[str, Any], metrics: Dict[str, float],
                       fitted_pipeline, input_example: Optional[pd.DataFrame] = None) -> Dict[str, str]:
        """Log satu run: params, metrics, dan model (pipeline lengkap
        preprocessing+classifier) sebagai artifact.

        Mengembalikan dict {'run_id': ..., 'model_uri': ...}. `model_uri`
        diambil LANGSUNG dari `ModelInfo.model_uri` yang dikembalikan
        `log_model()` (bentuk 'models:/m-xxxx' di MLflow 3.x) -- bukan
        dikonstruksi manual sebagai 'runs:/<id>/model', karena di MLflow 3.x
        model adalah entity tersendiri (bukan sekadar artifact di bawah run)
        sehingga URI manual bisa tidak match & memicu warning saat registrasi.
        """
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            signature = None
            example = None
            if input_example is not None:
                try:
                    example = input_example.head(2)
                    preds = fitted_pipeline.predict(example)
                    signature = infer_signature(example, preds)
                except Exception as exc:  # pragma: no cover - best effort, tidak boleh gagalkan run
                    logger.warning(f"Gagal infer signature untuk '{run_name}': {exc}")

            model_info = mlflow.sklearn.log_model(
                fitted_pipeline,
                name="model",
                serialization_format="cloudpickle",
                signature=signature,
                input_example=example,
            )
            logger.info(f"Run '{run_name}' logged -> run_id={run.info.run_id} | metrics={metrics}")
            return {"run_id": run.info.run_id, "model_uri": model_info.model_uri}

    def register_champion(self, model_uri: str, registered_model_name: str, alias: str = "champion") -> int:
        """Daftarkan model (dari `model_uri` hasil `log_model_run`) ke Model
        Registry, lalu tandai dengan alias (mis. 'champion'). Mengembalikan
        nomor versi model.
        """
        mv = mlflow.register_model(model_uri, registered_model_name)
        self.client.set_registered_model_alias(registered_model_name, alias, mv.version)
        logger.info(
            f"Model '{registered_model_name}' v{mv.version} terdaftar & diberi alias '{alias}'"
        )
        return int(mv.version)

    def get_best_historical_run(self, metric: str = "macro_f1") -> Optional[Dict[str, Any]]:
        """Cari run terbaik SEPANJANG SEJARAH experiment ini (berguna untuk
        monitoring: apakah hasil retraining hari ini lebih baik dari
        sebelumnya?). Mengembalikan None jika experiment belum punya run.
        """
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            return None
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1,
        )
        if not runs:
            return None
        best = runs[0]
        return {
            "run_id": best.info.run_id,
            "run_name": best.data.tags.get("mlflow.runName"),
            metric: best.data.metrics.get(metric),
        }
