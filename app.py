from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from inference import CreditScoreInferencer, RAW_INPUT_COLUMNS
from test_cases import load_illustrative_examples

APP_DIR = Path(__file__).parent
MODEL_DIR = APP_DIR / "model"
TEST_CASES_JSON = APP_DIR / "representative_test_cases.json"

CLASS_COLOR = {"Good": "#1DB954", "Standard": "#F5A623", "Poor": "#E03131"}
CLASS_ICON = {"Good": "✅", "Standard": "⚠️", "Poor": "🚫"}

st.set_page_config(page_title="Credit Score Deployment", page_icon="💳", layout="wide")


# ---------------------------------------------------------------------
# Load model (cached supaya tidak reload tiap interaksi)
# ---------------------------------------------------------------------
@st.cache_resource
def get_inferencer() -> CreditScoreInferencer:
    return CreditScoreInferencer(model_dir=MODEL_DIR)


@st.cache_data
def get_test_cases() -> dict:
    if TEST_CASES_JSON.exists():
        with open(TEST_CASES_JSON, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload["cases"], payload.get("source", "unknown")
    return load_illustrative_examples(), "illustrative (jalankan test_cases.py untuk data asli)"


try:
    inferencer = get_inferencer()
    model_load_error = None
except FileNotFoundError as exc:
    inferencer = None
    model_load_error = str(exc)

test_cases, test_case_source = get_test_cases()

DEFAULT_FIELDS = {
    "Age": 30, "Occupation": "", "Annual_Income": 50000.0, "Monthly_Inhand_Salary": 4000.0,
    "Num_Bank_Accounts": 3, "Num_Credit_Card": 3, "Interest_Rate": 15.0, "Num_of_Loan": 2,
    "Type_of_Loan": [], "Delay_from_due_date": 5, "Num_of_Delayed_Payment": 3,
    "Changed_Credit_Limit": 5.0, "Num_Credit_Inquiries": 3, "Credit_Mix": "Standard",
    "Outstanding_Debt": 1000.0, "Credit_Utilization_Ratio": 30.0,
    "Credit_History_Years": 5, "Credit_History_Months": 0, "Payment_of_Min_Amount": "No",
    "Total_EMI_per_month": 150.0, "Amount_invested_monthly": 200.0,
    "Payment_Behaviour": "Low_spent_Small_value_payments", "Monthly_Balance": 800.0,
}

for key, val in DEFAULT_FIELDS.items():
    st.session_state.setdefault(f"f_{key}", val)
st.session_state.setdefault("history", [])


SELECT_OPTIONS = {
    "Credit_Mix": ["Good", "Standard", "Bad"],
    "Payment_of_Min_Amount": ["No", "Yes"],
    "Payment_Behaviour": ["Low_spent_Small_value_payments", "Low_spent_Medium_value_payments",
                          "Low_spent_Large_value_payments", "High_spent_Small_value_payments",
                          "High_spent_Medium_value_payments", "High_spent_Large_value_payments"],
}
MULTISELECT_OPTIONS = {
    "Type_of_Loan": ["Auto Loan", "Credit-Builder Loan", "Personal Loan", "Home Equity Loan",
                     "Payday Loan", "Mortgage Loan", "Student Loan", "Debt Consolidation Loan"],
}


def load_case_into_form(class_name: str) -> None:
    case = test_cases.get(class_name, {})
    for field, value in case.items():
        target_key = f"f_{field}"

        if field == "Credit_History_Age" and isinstance(value, str):
            try:
                years = int(value.split(" Years")[0])
                months = int(value.split("and ")[1].split(" Months")[0])
                st.session_state["f_Credit_History_Years"] = years
                st.session_state["f_Credit_History_Months"] = months
            except (IndexError, ValueError):
                pass
            continue

        if field == "Type_of_Loan":
            items = [x.strip() for x in str(value or "").split(",") if x.strip()]
            valid_items = [x for x in items if x in MULTISELECT_OPTIONS["Type_of_Loan"]]
            st.session_state["f_Type_of_Loan"] = valid_items
            continue

        if field in SELECT_OPTIONS:
            # Selectbox hanya boleh diisi nilai yang ADA di daftar opsi -- kalau
            # data mentah masih kotor (mis. 'NM', '_______') atau kosong (None),
            # biarkan nilai default form apa adanya daripada memicu error widget.
            if value in SELECT_OPTIONS[field] and target_key in st.session_state:
                st.session_state[target_key] = value
            continue

        if field == "Occupation":
            st.session_state[target_key] = "" if value is None else str(value)
            continue

        if target_key in st.session_state and value is not None:
            st.session_state[target_key] = value

    st.session_state["_pending_predict"] = class_name


# ---------------------------------------------------------------------
# Sidebar: info model
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("💳 Credit Score Deployment")
    st.caption("UAS Model Deployment — Bagian 3 (Web Deployment + Inferencing)")

    if inferencer is not None:
        info = inferencer.model_info
        st.success("Model berhasil dimuat")
        st.metric("Model terbaik", info["best_model_name"])
        st.metric("Macro F1 (test set)", info["best_macro_f1"])
        st.caption(f"Kelas: {', '.join(info['label_classes'])}")
    else:
        st.error("Model belum tersedia")
        st.caption(model_load_error or "")

    st.divider()
    st.caption(f"Sumber test case: **{test_case_source}**")
    st.caption("Test case 'real_data' = diambil dari baris asli data_A.csv yang diprediksi benar dengan confidence tertinggi per kelas.")

    st.divider()
    st.subheader("🧪 Load Test Case per Kelas")
    cols = st.columns(3)
    for i, cls in enumerate(["Good", "Standard", "Poor"]):
        with cols[i]:
            st.button(f"{CLASS_ICON[cls]} {cls}", key=f"btn_{cls}", width='stretch',
                      on_click=load_case_into_form, args=(cls,))


# ---------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------
st.title("Prediksi Kelayakan Skor Kredit Nasabah")

tab_manual, tab_batch, tab_history = st.tabs(["📝 Prediksi Manual", "📂 Prediksi Batch (CSV)", "🕓 Riwayat Sesi Ini"])

# ===================== TAB 1: PREDIKSI MANUAL =====================
with tab_manual:
    if inferencer is None:
        st.warning("Model belum dimuat. Salin `best_model.pkl`, `label_encoder.pkl`, "
                   "`model_meta.pkl` ke folder `model/` lalu refresh halaman ini.")
    else:
        st.caption("Isi manual, atau klik salah satu tombol **Load Test Case** di sidebar "
                   "untuk mengisi otomatis dengan contoh per kelas.")

        with st.expander("👤 Data Personal & Pekerjaan", expanded=True):
            c1, c2 = st.columns(2)
            c1.number_input("Age (usia)", 18, 100, key="f_Age")
            c2.text_input("Occupation (pekerjaan)", key="f_Occupation",
                          placeholder="mis. Engineer, Teacher, Doctor ...")

        with st.expander("💰 Pendapatan", expanded=True):
            c1, c2 = st.columns(2)
            c1.number_input("Annual Income", 0.0, step=1000.0, key="f_Annual_Income")
            c2.number_input("Monthly Inhand Salary", 0.0, step=100.0, key="f_Monthly_Inhand_Salary",
                            help="Kosongkan/0 jika tidak tahu -- akan dihitung otomatis dari Annual Income.")

        with st.expander("🏦 Rekening, Kartu, & Pinjaman", expanded=True):
            c1, c2, c3 = st.columns(3)
            c1.number_input("Num Bank Accounts", 0, step=1, key="f_Num_Bank_Accounts")
            c2.number_input("Num Credit Card", 0, step=1, key="f_Num_Credit_Card")
            c3.number_input("Num of Loan", 0, step=1, key="f_Num_of_Loan")
            st.multiselect(
                "Type of Loan (boleh lebih dari 1)", key="f_Type_of_Loan",
                options=MULTISELECT_OPTIONS["Type_of_Loan"],
            )
            c1, c2 = st.columns(2)
            c1.number_input("Interest Rate (%)", 0.0, step=0.5, key="f_Interest_Rate")
            c2.number_input("Changed Credit Limit", key="f_Changed_Credit_Limit", step=0.5)

        with st.expander("⏰ Riwayat Pembayaran & Keterlambatan", expanded=True):
            c1, c2, c3 = st.columns(3)
            c1.number_input("Delay from due date (hari)", key="f_Delay_from_due_date", step=1)
            c2.number_input("Num of Delayed Payment", 0, step=1, key="f_Num_of_Delayed_Payment")
            c3.number_input("Num Credit Inquiries", 0, step=1, key="f_Num_Credit_Inquiries")
            c1, c2 = st.columns(2)
            c1.selectbox("Payment of Min Amount (bayar minimum saja?)", SELECT_OPTIONS["Payment_of_Min_Amount"], key="f_Payment_of_Min_Amount")
            c2.selectbox(
                "Payment Behaviour", key="f_Payment_Behaviour",
                options=SELECT_OPTIONS["Payment_Behaviour"],
            )

        with st.expander("📊 Profil Kredit", expanded=True):
            c1, c2 = st.columns(2)
            c1.selectbox("Credit Mix", SELECT_OPTIONS["Credit_Mix"], key="f_Credit_Mix")
            c2.slider("Credit Utilization Ratio (%)", 0.0, 100.0, step=0.5, key="f_Credit_Utilization_Ratio")
            c1, c2, c3 = st.columns(3)
            c1.number_input("Credit History -- Years", 0, 60, key="f_Credit_History_Years")
            c2.number_input("Credit History -- Months", 0, 11, key="f_Credit_History_Months")
            c3.number_input("Outstanding Debt", 0.0, step=100.0, key="f_Outstanding_Debt")
            c1, c2 = st.columns(2)
            c1.number_input("Total EMI per month", 0.0, step=10.0, key="f_Total_EMI_per_month")
            c2.number_input("Amount invested monthly", 0.0, step=10.0, key="f_Amount_invested_monthly")
            st.number_input("Monthly Balance", key="f_Monthly_Balance", step=50.0)

        predict_clicked = st.button("🔮 Prediksi Skor Kredit", type="primary", width='stretch')

        auto_predict = st.session_state.pop("_pending_predict", None)

        if predict_clicked or auto_predict:
            raw_input = {
                "Age": st.session_state["f_Age"],
                "Occupation": st.session_state["f_Occupation"] or None,
                "Annual_Income": st.session_state["f_Annual_Income"],
                "Monthly_Inhand_Salary": st.session_state["f_Monthly_Inhand_Salary"] or None,
                "Num_Bank_Accounts": st.session_state["f_Num_Bank_Accounts"],
                "Num_Credit_Card": st.session_state["f_Num_Credit_Card"],
                "Interest_Rate": st.session_state["f_Interest_Rate"],
                "Num_of_Loan": st.session_state["f_Num_of_Loan"],
                "Type_of_Loan": ", ".join(st.session_state["f_Type_of_Loan"]),
                "Delay_from_due_date": st.session_state["f_Delay_from_due_date"],
                "Num_of_Delayed_Payment": st.session_state["f_Num_of_Delayed_Payment"],
                "Changed_Credit_Limit": st.session_state["f_Changed_Credit_Limit"],
                "Num_Credit_Inquiries": st.session_state["f_Num_Credit_Inquiries"],
                "Credit_Mix": st.session_state["f_Credit_Mix"],
                "Outstanding_Debt": st.session_state["f_Outstanding_Debt"],
                "Credit_Utilization_Ratio": st.session_state["f_Credit_Utilization_Ratio"],
                "Credit_History_Age": f"{st.session_state['f_Credit_History_Years']} Years and "
                                       f"{st.session_state['f_Credit_History_Months']} Months",
                "Payment_of_Min_Amount": st.session_state["f_Payment_of_Min_Amount"],
                "Total_EMI_per_month": st.session_state["f_Total_EMI_per_month"],
                "Amount_invested_monthly": st.session_state["f_Amount_invested_monthly"],
                "Payment_Behaviour": st.session_state["f_Payment_Behaviour"],
                "Monthly_Balance": st.session_state["f_Monthly_Balance"],
            }
            result = inferencer.predict_one(raw_input)
            pred = result["predicted_class"]
            color = CLASS_COLOR.get(pred, "#666")

            st.markdown("---")
            st.markdown(
                f"""<div style="padding:1.2rem;border-radius:0.6rem;background:{color}22;
                border:2px solid {color};text-align:center;">
                <span style="font-size:1.1rem;">Hasil Prediksi</span><br>
                <span style="font-size:2.2rem;font-weight:700;color:{color};">
                {CLASS_ICON.get(pred,'')} {pred}</span><br>
                <span style="font-size:0.95rem;">Confidence: {result['confidence']*100:.1f}%</span>
                </div>""",
                unsafe_allow_html=True,
            )

            proba_df = pd.DataFrame({
                "Kelas": list(result["probabilities"].keys()),
                "Probabilitas": list(result["probabilities"].values()),
            }).sort_values("Probabilitas", ascending=False)
            st.bar_chart(proba_df.set_index("Kelas"))

            st.session_state["history"].append({
                "trigger": auto_predict or "manual", "predicted_class": pred,
                "confidence": result["confidence"], **result["probabilities"],
            })

# ===================== TAB 2: BATCH CSV =====================
with tab_batch:
    st.caption("Upload CSV dengan kolom mentah (sama seperti data_A.csv, kolom target boleh ada/tidak).")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None and inferencer is not None:
        batch_df = pd.read_csv(uploaded)
        with st.spinner("Memproses prediksi batch ..."):
            result_df = inferencer.predict_dataframe(batch_df)
        st.success(f"Selesai memprediksi {len(result_df)} baris.")
        st.dataframe(result_df, width='stretch')
        st.download_button(
            "⬇️ Download hasil (CSV)",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="hasil_prediksi_batch.csv",
            mime="text/csv",
        )

# ===================== TAB 3: HISTORY =====================
with tab_history:
    if st.session_state["history"]:
        st.caption("Semua prediksi yang dilakukan pada sesi ini (berguna untuk 1 screenshot mencakup semua kelas).")
        hist_df = pd.DataFrame(st.session_state["history"])
        st.dataframe(hist_df, width='stretch')
        if st.button("🗑️ Bersihkan riwayat"):
            st.session_state["history"] = []
            st.rerun()
    else:
        st.info("Belum ada prediksi pada sesi ini. Coba tombol Load Test Case di sidebar, lalu klik Prediksi.")
