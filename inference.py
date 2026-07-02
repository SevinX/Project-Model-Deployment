"""
inference.py
=============
Kode inferencing untuk model Credit_Score yang sudah di-deploy.

STRUKTUR FOLDER YANG DIASUMSIKAN (flat, cocok untuk Streamlit Community Cloud):
    repo-root/
    ├── app.py
    ├── inference.py            <- file ini
    ├── test_cases.py
    ├── requirements.txt
    ├── model/
    │   ├── best_model.pkl
    │   ├── label_encoder.pkl
    │   └── model_meta.pkl
    └── credit_score_pipeline/  <- package (folder ini, WAJIB ikut di-upload)
        ├── __init__.py
        ├── preprocessing.py
        └── ...

Dependency penting: `best_model.pkl` adalah sklearn Pipeline yang berisi
custom transformer class (RawDataCleaner, StatisticalOutlierClipper,
FallbackImputer, FeatureEngineer) yang didefinisikan di package
`credit_score_pipeline.preprocessing`. Python's pickle mengingat class lewat
*nama modul asalnya* -- jadi folder `credit_score_pipeline/` di atas WAJIB
ada persis sebagai sibling file ini (bukan di dalam subfolder lain, bukan
"src/credit_score_pipeline"), supaya `import credit_score_pipeline.preprocessing`
resolve ke tempat yang sama seperti saat model di-pickle.

Input yang diterima adalah data MENTAH (raw) -- persis format kolom di
data_A.csv asli (tanpa kolom identitas/target) -- karena seluruh cleaning,
outlier handling, dan feature engineering sudah dibungkus di dalam pipeline
itu sendiri. Pemanggil TIDAK perlu membersihkan data terlebih dahulu.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

# --- Pastikan package credit_score_pipeline (folder sibling file ini) bisa
#     di-import, apa pun current working directory saat Streamlit dijalankan. ---
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

if not (_THIS_DIR / "credit_score_pipeline").exists():
    raise ModuleNotFoundError(
        f"Folder 'credit_score_pipeline/' tidak ditemukan di '{_THIS_DIR}'. "
        f"Folder package ini WAJIB di-upload persis sebagai sibling inference.py "
        f"(lihat docstring di atas file ini) -- bukan di dalam 'src/', bukan di "
        f"folder lain. Salin folder 'credit_score_pipeline/src/credit_score_pipeline/' "
        f"dari project pipeline (bagian 2) ke sini, pertahankan namanya."
    )

# Import ini WAJIB ada sebelum pickle.load() di bawah, walau terlihat tidak
# dipakai langsung -- supaya class-nya terdaftar di sys.modules sebelum
# unpickling mencarinya.
from credit_score_pipeline import preprocessing as _preprocessing  # noqa: F401,E402


# Kolom mentah yang dibutuhkan model (persis skema data_A.csv, tanpa kolom
# identitas ['Unnamed: 0','ID','Customer_ID','Month','Name','SSN'] dan tanpa
# kolom target 'Credit_Score' -- keduanya tidak relevan untuk inferencing).
RAW_INPUT_COLUMNS: List[str] = [
    "Age", "Occupation", "Annual_Income", "Monthly_Inhand_Salary",
    "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
    "Type_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Num_Credit_Inquiries", "Credit_Mix",
    "Outstanding_Debt", "Credit_Utilization_Ratio", "Credit_History_Age",
    "Payment_of_Min_Amount", "Total_EMI_per_month", "Amount_invested_monthly",
    "Payment_Behaviour", "Monthly_Balance",
]


class CreditScoreInferencer:
    """Wrapper OOP untuk load model terdeploy + jalankan prediksi.

    Contoh pemakaian
    -----------------
    >>> inferencer = CreditScoreInferencer(model_dir="model")
    >>> result = inferencer.predict_one({"Age": 35, "Annual_Income": 50000, ...})
    >>> result["predicted_class"]
    'Good'
    """

    def __init__(self, model_dir: Union[str, Path] = "model") -> None:
        self.model_dir = Path(model_dir)
        self.pipeline = None
        self.label_encoder = None
        self.meta: Dict[str, Any] = {}
        self._load_artifacts()

    # ------------------------------------------------------------------
    def _load_artifacts(self) -> None:
        model_path = self.model_dir / "best_model.pkl"
        encoder_path = self.model_dir / "label_encoder.pkl"
        meta_path = self.model_dir / "model_meta.pkl"

        for p in (model_path, encoder_path, meta_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Artifact tidak ditemukan: '{p}'. Salin best_model.pkl, "
                    f"label_encoder.pkl, dan model_meta.pkl dari folder "
                    f"'artifacts/' hasil training pipeline (bagian 2) ke "
                    f"'{self.model_dir}/' sebelum menjalankan inferencing."
                )

        with open(model_path, "rb") as f:
            self.pipeline = pickle.load(f)
        with open(encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)
        with open(meta_path, "rb") as f:
            self.meta = pickle.load(f)

    # ------------------------------------------------------------------
    def _to_dataframe(self, raw_input: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
        """Terima 1 dict (single record) atau DataFrame (batch), kembalikan
        DataFrame dengan kolom lengkap sesuai RAW_INPUT_COLUMNS. Kolom yang
        tidak diisi akan diisi NaN -- aman, karena pipeline sudah menangani
        missing value secara menyeluruh.
        """
        if isinstance(raw_input, dict):
            df = pd.DataFrame([raw_input])
        elif isinstance(raw_input, pd.DataFrame):
            df = raw_input.copy()
        else:
            raise TypeError("raw_input harus berupa dict (1 baris) atau pandas DataFrame (banyak baris).")

        for col in RAW_INPUT_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan

        return df[RAW_INPUT_COLUMNS]

    # ------------------------------------------------------------------
    def predict(self, raw_input: Union[Dict[str, Any], pd.DataFrame]) -> List[Dict[str, Any]]:
        """Prediksi untuk 1 atau banyak baris. Selalu mengembalikan LIST of dict
        (1 dict per baris input), masing-masing berisi predicted_class dan
        probabilitas tiap kelas.
        """
        df = self._to_dataframe(raw_input)

        pred_encoded = self.pipeline.predict(df)
        pred_proba = self.pipeline.predict_proba(df)
        class_names = list(self.label_encoder.classes_)

        results = []
        for i in range(len(df)):
            proba_dict = {cls: round(float(pred_proba[i][j]), 4) for j, cls in enumerate(class_names)}
            results.append({
                "predicted_class": self.label_encoder.inverse_transform([pred_encoded[i]])[0],
                "probabilities": proba_dict,
                "confidence": round(float(max(pred_proba[i])), 4),
            })
        return results

    def predict_one(self, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        """Shortcut untuk 1 record -> 1 dict hasil (bukan list)."""
        return self.predict(raw_input)[0]

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Shortcut untuk batch: kembalikan df asli + kolom hasil prediksi."""
        results = self.predict(df)
        out = df.copy()
        out["predicted_class"] = [r["predicted_class"] for r in results]
        out["confidence"] = [r["confidence"] for r in results]
        return out

    # ------------------------------------------------------------------
    @property
    def model_info(self) -> Dict[str, Any]:
        return {
            "best_model_name": self.meta.get("best_model_name"),
            "best_macro_f1": self.meta.get("best_macro_f1"),
            "label_classes": list(self.label_encoder.classes_),
        }


# ------------------------------------------------------------------------
# CLI kecil untuk quick test dari terminal:
#   python inference.py
# ------------------------------------------------------------------------
if __name__ == "__main__":
    inferencer = CreditScoreInferencer(model_dir=Path(__file__).parent / "model")
    print("Model info:", inferencer.model_info)

    dummy = {
        "Age": 34, "Occupation": "Engineer", "Annual_Income": 65000,
        "Monthly_Inhand_Salary": 5400, "Num_Bank_Accounts": 3, "Num_Credit_Card": 4,
        "Interest_Rate": 12, "Num_of_Loan": 2, "Type_of_Loan": "Auto Loan, Student Loan",
        "Delay_from_due_date": 5, "Num_of_Delayed_Payment": 3, "Changed_Credit_Limit": 4.2,
        "Num_Credit_Inquiries": 2, "Credit_Mix": "Standard", "Outstanding_Debt": 1200,
        "Credit_Utilization_Ratio": 28.5, "Credit_History_Age": "10 Years and 3 Months",
        "Payment_of_Min_Amount": "No", "Total_EMI_per_month": 150, "Amount_invested_monthly": 200,
        "Payment_Behaviour": "High_spent_Medium_value_payments", "Monthly_Balance": 850,
    }
    result = inferencer.predict_one(dummy)
    print("Contoh prediksi:", result)
