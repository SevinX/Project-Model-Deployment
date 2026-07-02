"""
test_cases.py
=============
Menyediakan 1 test case representatif per kelas (Good / Standard / Poor)
untuk memenuhi requirement UAS:
    "testing hasil deployment dengan menggunakan test case yang
    merepresentasikan setiap kelas"

Dua sumber test case:
1. `load_illustrative_examples()` -- 3 contoh buatan tangan (domain reasoning
   kasar) untuk demo cepat SEBELUM data/model asli tersedia. Jelas ditandai
   ILUSTRATIF, bukan hasil validasi model.
2. `extract_real_examples_from_data(...)` -- RECOMMENDED. Mengambil baris
   ASLI dari data_A.csv yang: (a) label aslinya == kelas tsb, (b) diprediksi
   BENAR oleh model, (c) confidence prediksi tertinggi di antara kandidat --
   sehingga contoh yang didapat benar-benar merepresentasikan kelasnya
   menurut model yang sudah dilatih, bukan tebakan.

Jalankan langsung untuk generate `representative_test_cases.json`:
    python test_cases.py --data ../data/data_A.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from inference import RAW_INPUT_COLUMNS, CreditScoreInferencer

_SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))
from credit_score_pipeline.preprocessing import RawDataCleaner, clean_numeric_str  # noqa: E402


def extract_real_examples_from_data(data_path: str, model_dir: str = "model") -> Dict[str, Dict[str, Any]]:
    """Ambil 1 baris asli per kelas dari data_path: label benar + confidence
    tertinggi. Mengembalikan {class_name: raw_input_dict}.
    """
    df = pd.read_csv(data_path)
    if "Credit_Score" not in df.columns:
        raise KeyError("Kolom 'Credit_Score' tidak ada -- pastikan data_path menunjuk ke data_A.csv asli (dengan label).")

    inferencer = CreditScoreInferencer(model_dir=model_dir)
    raw_X = df[[c for c in RAW_INPUT_COLUMNS if c in df.columns]]

    predictions = inferencer.predict(raw_X)
    df = df.reset_index(drop=True)
    df["_predicted_class"] = [p["predicted_class"] for p in predictions]
    df["_confidence"] = [p["confidence"] for p in predictions]

    # Bersihkan kolom numerik yang di raw CSV masih berformat string kotor
    # (mis. "_69.0_") -- supaya test case yang dihasilkan berisi angka bersih,
    # enak dipakai langsung mengisi form Streamlit (widget number_input butuh
    # tipe numerik asli, bukan string).
    for col in RawDataCleaner.DIRTY_NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = clean_numeric_str(df[col])

    # Bersihkan token kategorikal kotor (mis. 'NM', '_______') -> None, supaya
    # kompatibel dengan pilihan tetap di widget selectbox/multiselect app.py.
    # (Pipeline TETAP bisa terima data kotor -- ini murni supaya test case
    # yang ditampilkan di form enak dilihat & tidak memicu error widget.)
    for col in RawDataCleaner.DIRTY_CAT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].where(~df[col].isin(RawDataCleaner.DIRTY_CAT_TOKENS), None)

    examples: Dict[str, Dict[str, Any]] = {}
    for cls in inferencer.label_encoder.classes_:
        candidates = df[(df["Credit_Score"] == cls) & (df["_predicted_class"] == cls)]
        if candidates.empty:
            print(f"[!] Tidak ada baris untuk kelas '{cls}' yang diprediksi benar -- dilewati. "
                  f"Cek kembali performa model.")
            continue
        best_row = candidates.sort_values("_confidence", ascending=False).iloc[0]
        examples[cls] = {
            col: _to_native_type(best_row[col])
            for col in RAW_INPUT_COLUMNS if col in best_row
        }
        print(f"[{cls}] confidence={best_row['_confidence']:.4f}")

    return examples


def _to_native_type(value: Any) -> Any:
    """Konversi nilai numpy/pandas (int64, float64, dst.) ke tipe Python
    native, supaya aman di-serialize ke JSON sebagai angka (bukan string).
    NaN dikonversi ke None.
    """
    if pd.isna(value):
        return None
    if hasattr(value, "item"):  # numpy scalar (int64, float64, bool_, dst.)
        return value.item()
    return value


def load_illustrative_examples() -> Dict[str, Dict[str, Any]]:
    """Contoh ILUSTRATIF (domain reasoning, BUKAN hasil ekstraksi data asli).
    Dipakai hanya untuk demo UI cepat sebelum data/model asli siap.
    """
    return {
        "Good": {
            "Age": 38, "Occupation": "Engineer", "Annual_Income": 95000,
            "Monthly_Inhand_Salary": 7800, "Num_Bank_Accounts": 3, "Num_Credit_Card": 3,
            "Interest_Rate": 8, "Num_of_Loan": 1, "Type_of_Loan": "Auto Loan",
            "Delay_from_due_date": 2, "Num_of_Delayed_Payment": 0, "Changed_Credit_Limit": 2.1,
            "Num_Credit_Inquiries": 1, "Credit_Mix": "Good", "Outstanding_Debt": 380,
            "Credit_Utilization_Ratio": 18.4, "Credit_History_Age": "18 Years and 4 Months",
            "Payment_of_Min_Amount": "No", "Total_EMI_per_month": 90, "Amount_invested_monthly": 650,
            "Payment_Behaviour": "Low_spent_Small_value_payments", "Monthly_Balance": 2100,
        },
        "Standard": {
            "Age": 29, "Occupation": "Teacher", "Annual_Income": 48000,
            "Monthly_Inhand_Salary": 4000, "Num_Bank_Accounts": 4, "Num_Credit_Card": 5,
            "Interest_Rate": 17, "Num_of_Loan": 3, "Type_of_Loan": "Personal Loan, Credit-Builder Loan",
            "Delay_from_due_date": 12, "Num_of_Delayed_Payment": 8, "Changed_Credit_Limit": 6.5,
            "Num_Credit_Inquiries": 4, "Credit_Mix": "Standard", "Outstanding_Debt": 1450,
            "Credit_Utilization_Ratio": 33.0, "Credit_History_Age": "6 Years and 8 Months",
            "Payment_of_Min_Amount": "Yes", "Total_EMI_per_month": 210, "Amount_invested_monthly": 90,
            "Payment_Behaviour": "High_spent_Medium_value_payments", "Monthly_Balance": 620,
        },
        "Poor": {
            "Age": 24, "Occupation": "_______", "Annual_Income": 19000,
            "Monthly_Inhand_Salary": 1450, "Num_Bank_Accounts": 9, "Num_Credit_Card": 10,
            "Interest_Rate": 32, "Num_of_Loan": 7, "Type_of_Loan": "Payday Loan, Personal Loan, Auto Loan",
            "Delay_from_due_date": 35, "Num_of_Delayed_Payment": 18, "Changed_Credit_Limit": 15.8,
            "Num_Credit_Inquiries": 11, "Credit_Mix": "Bad", "Outstanding_Debt": 4200,
            "Credit_Utilization_Ratio": 46.5, "Credit_History_Age": "1 Years and 2 Months",
            "Payment_of_Min_Amount": "Yes", "Total_EMI_per_month": 480, "Amount_invested_monthly": 0,
            "Payment_Behaviour": "High_spent_Large_value_payments", "Monthly_Balance": 110,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate representative test cases per kelas")
    parser.add_argument("--data", type=str, default="../data/data_A.csv", help="Path ke data_A.csv asli (dengan label)")
    parser.add_argument("--model-dir", type=str, default="model", help="Folder berisi best_model.pkl dkk")
    parser.add_argument("--out", type=str, default="representative_test_cases.json", help="Output JSON")
    args = parser.parse_args()

    if Path(args.data).exists():
        print(f"Mengekstrak test case NYATA dari '{args.data}' ...")
        cases = extract_real_examples_from_data(args.data, args.model_dir)
        source = "real_data"
    else:
        print(f"'{args.data}' tidak ditemukan -- pakai contoh ILUSTRATIF sebagai fallback.")
        cases = load_illustrative_examples()
        source = "illustrative"

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"source": source, "cases": cases}, f, indent=2)
    print(f"\nTersimpan -> {args.out} (source={source})")
