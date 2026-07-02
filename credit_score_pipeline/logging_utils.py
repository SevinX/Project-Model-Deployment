"""
Setup logging terpusat untuk seluruh pipeline.

Kenapa terpisah dari MLflow?
- MLflow mencatat *hasil* (params, metrics, artifact model) -> untuk monitoring
  eksperimen di MLflow UI.
- `logging` module ini mencatat *proses* (progress, warning, error, timing) ke
  console dan ke file `logs/pipeline.log` -> untuk debugging & audit trail saat
  pipeline dijalankan ulang (retraining) secara otomatis/terjadwal.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(logs_dir: str | Path = "logs", level: int = logging.INFO) -> logging.Logger:
    """Konfigurasi root logger project: output ke console + file (append)."""
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "pipeline.log"

    logger = logging.getLogger("credit_score_pipeline")
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        # Sudah pernah di-setup (mis. dipanggil ulang di notebook/interactive) -> skip
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Child logger bernamespace, mewarisi handler dari root 'credit_score_pipeline'."""
    return logging.getLogger(f"credit_score_pipeline.{name}")
