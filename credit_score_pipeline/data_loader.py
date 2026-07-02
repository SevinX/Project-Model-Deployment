"""
data_loader.py
===============
Bertanggung jawab HANYA untuk I/O: membaca CSV mentah dan melakukan split
train/test + encoding target. Tidak melakukan cleaning/feature engineering
apa pun di sini -- itu tanggung jawab `preprocessing.py` (agar tetap fit-safe
dan reusable saat inference).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .logging_utils import get_logger

logger = get_logger("data_loader")


class DataLoader:
    """Load dataset mentah dan siapkan split train/test.

    Parameters
    ----------
    target_column : str
        Nama kolom target (default sesuai notebook: 'Credit_Score').
    test_size : float
        Proporsi data test.
    random_state : int
        Seed untuk reprodusibilitas split.
    """

    def __init__(self, target_column: str = "Credit_Score",
                 test_size: float = 0.2, random_state: int = 42) -> None:
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoder: LabelEncoder = LabelEncoder()

    def load_raw(self, path: str | Path) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset tidak ditemukan di '{path}'. "
                f"Pastikan file data_A.csv sudah diletakkan di folder ini "
                f"(lihat README.md bagian 'Menyiapkan data')."
            )
        logger.info(f"Membaca dataset mentah dari: {path}")
        df = pd.read_csv(path)
        logger.info(f"Dataset dimuat -> shape: {df.shape}")
        return df

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, "pd.Series", "pd.Series"]:
        """Split fitur mentah (belum dibersihkan) dan target ter-encode.

        Encoding target dengan LabelEncoder di-fit pada seluruh label (bukan
        hanya train) -- ini AMAN karena hanya mendaftar kategori kelas target,
        tidak melihat/menggunakan informasi apa pun dari fitur X, sehingga
        tidak menimbulkan data leakage prediktif (identik dengan praktik di
        notebook aslinya).
        """
        if self.target_column not in df.columns:
            raise KeyError(
                f"Kolom target '{self.target_column}' tidak ada di dataset. "
                f"Kolom tersedia: {list(df.columns)}"
            )

        X = df.drop(columns=[self.target_column])
        y_raw = df[self.target_column]
        y_enc = self.label_encoder.fit_transform(y_raw)

        logger.info(f"Kelas target terdeteksi: {list(self.label_encoder.classes_)}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc,
            test_size=self.test_size,
            stratify=y_enc,
            random_state=self.random_state,
        )
        logger.info(
            f"Split selesai -> train: {X_train.shape}, test: {X_test.shape}"
        )
        return X_train, X_test, y_train, y_test
