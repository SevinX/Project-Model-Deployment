"""
preprocessing.py
=================
Replikasi lengkap logic EDA/cleaning/feature-engineering dari
`Credit_Score_EDA_Modelling.ipynb` (section B & C), dibungkus sebagai
sklearn-compatible Transformer agar bisa dipakai di dalam Pipeline.

Perbedaan sengaja dari notebook asli (dan alasannya)
------------------------------------------------------
Di notebook, sebagian besar statistik (P99 clip, median fallback, mode
Credit_Mix) dihitung di ATAS SELURUH dataframe SEBELUM train/test split.
Ini wajar untuk notebook eksplorasi, tapi berisiko *data leakage* ringan
dan -- yang lebih penting untuk pipeline ini -- tidak valid dipakai ulang
saat *retraining* pada data baru (statistik akan basi / dihitung dari
campuran train+test).

Semua transformer di bawah ini dipecah sesuai sifatnya:
- Stateless (aturan/rumus tetap, tidak butuh statistik dari data) -> aman
  dijalankan kapan saja: `RawDataCleaner`.
- Stateful (butuh Q1/Q3, P99, median, mode dari data) -> WAJIB fit() hanya
  pada data training, lalu transform() dipakai konsisten ke train & test:
  `StatisticalOutlierClipper`, `FallbackImputer`, `FeatureEngineer`.

Konsekuensinya: metrik akhir pipeline ini bisa sedikit berbeda (biasanya
sangat tipis) dari angka di notebook asli, karena threshold di sini murni
dipelajari dari 80% data training, bukan 100% data. Ini justru perilaku
yang benar untuk pipeline retraining.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .logging_utils import get_logger

logger = get_logger("preprocessing")


# -----------------------------------------------------------------------
# Helper functions (persis dari notebook, section "1. Cleaning data")
# -----------------------------------------------------------------------

def clean_numeric_str(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace("_", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


_CREDIT_HISTORY_PATTERN = re.compile(r"(\d+)\s+Years?\s+and\s+(\d+)\s+Months?", re.I)


def parse_credit_history_age(series: pd.Series) -> pd.Series:
    def to_months(val):
        if pd.isna(val):
            return np.nan
        m = _CREDIT_HISTORY_PATTERN.search(str(val))
        return int(m.group(1)) * 12 + int(m.group(2)) if m else np.nan
    return series.apply(to_months)


# -----------------------------------------------------------------------
# 1) RawDataCleaner -- STATELESS (aman dipanggil sebelum atau di dalam Pipeline)
#    Replikasi notebook section B.1, B.2, B.3 (outlier hard-rule), B.4, B.5
#    (bagian yang berbasis rumus/konstanta, bukan statistik dataset)
# -----------------------------------------------------------------------

class RawDataCleaner(BaseEstimator, TransformerMixin):
    """Cleaning dasar: drop identifier, numeric string kotor -> numeric,
    kategorikal kotor -> NaN, parsing Credit_History_Age, hard-rule outlier
    (batas logis absolut), ekstraksi Type_of_Loan, dan imputasi berbasis
    domain knowledge (rumus/konstanta tetap -- bukan statistik dataset).
    """

    ID_COLUMNS = ["Unnamed: 0", "ID", "Customer_ID", "Month", "Name", "SSN"]

    DIRTY_NUMERIC_COLUMNS = [
        "Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment",
        "Changed_Credit_Limit", "Outstanding_Debt", "Amount_invested_monthly",
        "Monthly_Balance", "Monthly_Inhand_Salary", "Interest_Rate",
        "Credit_Utilization_Ratio", "Total_EMI_per_month",
    ]

    DIRTY_CAT_TOKENS = {"_______", "!@9#%8", "NM", "_", "nan", ""}
    DIRTY_CAT_COLUMNS = ["Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]

    def fit(self, X: pd.DataFrame, y=None) -> "RawDataCleaner":
        return self  # stateless

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # -- Drop identifiers --
        df = df.drop(columns=[c for c in self.ID_COLUMNS if c in df.columns])

        # -- Numeric kotor -> numeric --
        for col in self.DIRTY_NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = clean_numeric_str(df[col])

        # -- Kategorikal kotor -> NaN --
        # Pakai .mask() (bukan .replace()) supaya tidak kena FutureWarning
        # downcasting pandas 2.x/3.x; hasil akhirnya identik dengan notebook.
        for col in self.DIRTY_CAT_COLUMNS:
            if col in df.columns:
                str_col = df[col].astype(str).str.strip()
                is_dirty = str_col.isin(self.DIRTY_CAT_TOKENS) | (str_col == "nan")
                df[col] = str_col.mask(is_dirty, np.nan)

        # -- Credit_History_Age -> Credit_History_Age_months --
        if "Credit_History_Age" in df.columns:
            df["Credit_History_Age_months"] = parse_credit_history_age(df["Credit_History_Age"])
            df = df.drop(columns=["Credit_History_Age"])

        # -- Hard-rule outlier -> NaN (batas logis absolut, bukan statistik) --
        if "Age" in df.columns:
            df.loc[(df["Age"] <= 0) | (df["Age"] > 100), "Age"] = np.nan
        if "Num_of_Loan" in df.columns:
            df.loc[(df["Num_of_Loan"] < 0) | (df["Num_of_Loan"] > 20), "Num_of_Loan"] = np.nan
        if "Num_of_Delayed_Payment" in df.columns:
            df.loc[df["Num_of_Delayed_Payment"] < 0, "Num_of_Delayed_Payment"] = np.nan
        if "Num_Bank_Accounts" in df.columns:
            df.loc[df["Num_Bank_Accounts"] < 0, "Num_Bank_Accounts"] = np.nan
        if "Interest_Rate" in df.columns:
            df["Interest_Rate"] = df["Interest_Rate"].clip(upper=100)
        if "Delay_from_due_date" in df.columns:
            df["Delay_from_due_date"] = df["Delay_from_due_date"].clip(lower=0)

        # -- Type_of_Loan -> Count_Loan_Types, Has_Payday_Loan --
        if "Type_of_Loan" in df.columns:
            df["Type_of_Loan"] = df["Type_of_Loan"].fillna("")
            df["Count_Loan_Types"] = df["Type_of_Loan"].apply(
                lambda x: len(str(x).split(",")) if str(x).strip() != "" else 0
            )
            df["Has_Payday_Loan"] = (
                df["Type_of_Loan"].str.contains("Payday Loan", case=False, na=False).astype(int)
            )
            df = df.drop(columns=["Type_of_Loan"])

        # -- Domain-rule imputation (rumus/konstanta tetap) --
        if "Monthly_Inhand_Salary" in df.columns and "Annual_Income" in df.columns:
            mask = df["Monthly_Inhand_Salary"].isnull() & df["Annual_Income"].notnull()
            df.loc[mask, "Monthly_Inhand_Salary"] = df.loc[mask, "Annual_Income"] / 12

        if "Num_Credit_Inquiries" in df.columns:
            df.loc[df["Num_Credit_Inquiries"].isnull(), "Num_Credit_Inquiries"] = 0

        if "Amount_invested_monthly" in df.columns:
            df.loc[df["Amount_invested_monthly"].isnull(), "Amount_invested_monthly"] = 0

        if "Num_of_Delayed_Payment" in df.columns and "Delay_from_due_date" in df.columns:
            mask = df["Num_of_Delayed_Payment"].isnull() & (df["Delay_from_due_date"] <= 0)
            df.loc[mask, "Num_of_Delayed_Payment"] = 0

        return df


# -----------------------------------------------------------------------
# 2) StatisticalOutlierClipper -- STATEFUL (fit hanya di training)
#    Replikasi clip_iqr_upper() + P99 clip Total_EMI_per_month
# -----------------------------------------------------------------------

class StatisticalOutlierClipper(BaseEstimator, TransformerMixin):
    """Clip outlier berbasis statistik data: Q3 + 1.5*IQR (upper only) untuk
    kolom count, dan P99 untuk Total_EMI_per_month. Threshold di-fit HANYA
    dari data training.
    """

    IQR_COLUMNS = ["Num_Bank_Accounts", "Num_Credit_Card", "Num_Credit_Inquiries"]
    IQR_MULTIPLIER = 1.5
    P99_COLUMNS = ["Total_EMI_per_month"]

    def fit(self, X: pd.DataFrame, y=None) -> "StatisticalOutlierClipper":
        self.iqr_upper_bounds_: Dict[str, float] = {}
        for col in self.IQR_COLUMNS:
            if col in X.columns:
                q1, q3 = X[col].quantile(0.25), X[col].quantile(0.75)
                self.iqr_upper_bounds_[col] = q3 + self.IQR_MULTIPLIER * (q3 - q1)

        self.p99_bounds_: Dict[str, float] = {}
        for col in self.P99_COLUMNS:
            if col in X.columns:
                self.p99_bounds_[col] = X[col].quantile(0.99)

        logger.info(f"OutlierClipper fit -> IQR bounds: {self.iqr_upper_bounds_}, P99: {self.p99_bounds_}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col, upper in self.iqr_upper_bounds_.items():
            if col in df.columns:
                df[col] = df[col].clip(upper=upper)
        for col, upper in self.p99_bounds_.items():
            if col in df.columns:
                df[col] = df[col].clip(upper=upper)
        return df


# -----------------------------------------------------------------------
# 3) FallbackImputer -- STATEFUL (fit hanya di training)
#    Diambil hampir persis dari notebook (di notebook aslinya, class ini
#    SUDAH fit-safe karena memang sudah didesain sebagai sklearn Transformer
#    di dalam Pipeline). Hanya ditambah sedikit pengaman.
# -----------------------------------------------------------------------

class FallbackImputer(BaseEstimator, TransformerMixin):
    """Imputer hierarkis untuk Monthly_Balance:
    1) median per Occupation -> 2) median per bracket Annual_Income (4 kuartil)
    -> 3) median global. Persis logic notebook section 5.1.
    """

    def __init__(self) -> None:
        self.level1_medians_: dict = {}
        self.level2_medians_: dict = {}
        self.level3_median_: Optional[float] = None
        self.income_bins_: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame, y=None) -> "FallbackImputer":
        df_temp = X.copy()

        self.level1_medians_ = (
            df_temp.groupby("Occupation")["Monthly_Balance"].median().to_dict()
        )

        _, bins = pd.qcut(
            df_temp["Annual_Income"], q=4,
            labels=["Low", "Medium", "High", "Very High"],
            retbins=True, duplicates="drop",
        )
        self.income_bins_ = bins
        self.income_bins_[0] = -np.inf
        self.income_bins_[-1] = np.inf

        bracket_train = pd.cut(
            df_temp["Annual_Income"], bins=self.income_bins_,
            labels=["Low", "Medium", "High", "Very High"],
        )
        df_temp["_bracket"] = bracket_train
        self.level2_medians_ = (
            df_temp.groupby("_bracket", observed=True)["Monthly_Balance"].median().to_dict()
        )
        self.level3_median_ = df_temp["Monthly_Balance"].median()
        if pd.isna(self.level3_median_):
            self.level3_median_ = 0.0

        logger.info("FallbackImputer fit selesai (level1/level2/level3 medians disiapkan)")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df_temp = X.copy()
        df_temp["Monthly_Balance"] = df_temp["Monthly_Balance"].astype(float)

        fill1 = df_temp["Occupation"].map(self.level1_medians_)
        df_temp["Monthly_Balance"] = df_temp["Monthly_Balance"].fillna(fill1)

        if df_temp["Monthly_Balance"].isnull().any():
            bracket = pd.cut(
                df_temp["Annual_Income"], bins=self.income_bins_,
                labels=["Low", "Medium", "High", "Very High"],
            )
            fill2 = pd.to_numeric(bracket.map(self.level2_medians_), errors="coerce")
            df_temp["Monthly_Balance"] = df_temp["Monthly_Balance"].fillna(fill2)

        df_temp["Monthly_Balance"] = df_temp["Monthly_Balance"].fillna(self.level3_median_)

        return df_temp


# -----------------------------------------------------------------------
# 4) FeatureEngineer -- STATEFUL (fit hanya di training)
#    Replikasi notebook section 6 (rasio finansial + index), 7 (ordinal
#    Credit_Mix), 7.1 (interaksi Credit_Mix)
# -----------------------------------------------------------------------

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Membuat seluruh fitur turunan: rasio finansial, Financial_Stress_Index,
    Credit_Hunger_Score, Struggling_Flag, High_Risk_Utilization,
    Credit_Start_Age, Credit_Mix_Ordinal, dan fitur interaksi CreditMix_x_*.

    `fit()` menjalankan urutan yang sama seperti `transform()` di atas data
    training untuk mempelajari P99/median/mode yang dibutuhkan, lalu
    menyimpannya sebagai atribut ber-akhiran `_` sesuai konvensi sklearn.
    """

    EPS = 1.0
    CREDIT_MIX_MAP = {"Bad": 0, "Standard": 1, "Good": 2}

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        self._process(X, fitting=True)
        logger.info("FeatureEngineer fit selesai (P99/median/mode dipelajari dari training data)")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._process(X, fitting=False)

    def _process(self, X: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        df = X.copy()

        # --- 1. Rasio finansial ---
        if {"Outstanding_Debt", "Annual_Income"}.issubset(df.columns):
            df["Debt_to_Income_Ratio"] = df["Outstanding_Debt"] / (df["Annual_Income"] + self.EPS)
        if {"Total_EMI_per_month", "Monthly_Inhand_Salary"}.issubset(df.columns):
            df["EMI_to_Income_Ratio"] = df["Total_EMI_per_month"] / (df["Monthly_Inhand_Salary"] + self.EPS)
        if {"Amount_invested_monthly", "Monthly_Inhand_Salary"}.issubset(df.columns):
            df["Invest_to_Income_Ratio"] = df["Amount_invested_monthly"] / (df["Monthly_Inhand_Salary"] + self.EPS)

        for ratio_col in ["EMI_to_Income_Ratio", "Invest_to_Income_Ratio", "Debt_to_Income_Ratio"]:
            if ratio_col in df.columns:
                attr = f"_p99_{ratio_col}"
                if fitting:
                    setattr(self, attr, df[ratio_col].quantile(0.99))
                p99 = getattr(self, attr)
                df[ratio_col] = df[ratio_col].clip(upper=p99)

        # --- 2. Financial_Stress_Index ---
        df["Financial_Stress_Index"] = (
            df["Delay_from_due_date"].fillna(0) * df["Num_of_Delayed_Payment"].fillna(0)
        )
        if fitting:
            self._p99_fsi = df["Financial_Stress_Index"].quantile(0.99)
        df["Financial_Stress_Index"] = df["Financial_Stress_Index"].clip(upper=self._p99_fsi)

        # --- 3. Credit_Hunger_Score ---
        if fitting:
            self._median_credit_history = df["Credit_History_Age_months"].median()
        history_filled = df["Credit_History_Age_months"].fillna(self._median_credit_history)
        df["Credit_Hunger_Score"] = (
            (df["Num_Credit_Card"].fillna(0) + df["Num_Credit_Inquiries"].fillna(0))
            / (history_filled + self.EPS)
        )
        if fitting:
            self._p99_chs = df["Credit_Hunger_Score"].quantile(0.99)
        df["Credit_Hunger_Score"] = df["Credit_Hunger_Score"].clip(upper=self._p99_chs)

        # --- 4. Struggling_Flag (stateless, threshold tetap) ---
        df["Struggling_Flag"] = (
            (df["Payment_of_Min_Amount"].fillna("No").str.upper() == "YES")
            & (df["Num_of_Delayed_Payment"].fillna(0) > 2)
        ).astype(int)

        # --- 5. High_Risk_Utilization (stateless, threshold tetap) ---
        df["High_Risk_Utilization"] = (df["Credit_Utilization_Ratio"].fillna(0) > 35).astype(int)

        # --- 6. Credit_Start_Age (stateless) ---
        df["Credit_Start_Age"] = df["Age"] - (df["Credit_History_Age_months"] / 12)
        df["Credit_Start_Age"] = df["Credit_Start_Age"].clip(lower=0)

        # --- 7. Ordinal encoding Credit_Mix ---
        df["Credit_Mix_Ordinal"] = df["Credit_Mix"].map(self.CREDIT_MIX_MAP)
        df = df.drop(columns=["Credit_Mix"])
        if fitting:
            mode_series = df["Credit_Mix_Ordinal"].mode()
            self._credit_mix_mode = mode_series.iloc[0] if len(mode_series) else 1
        df["Credit_Mix_Ordinal"] = df["Credit_Mix_Ordinal"].fillna(self._credit_mix_mode)

        # --- 7.1 Interaksi Credit_Mix ---
        df["CreditMix_x_Payment"] = (
            df["Credit_Mix_Ordinal"]
            * (df["Payment_of_Min_Amount"].fillna("No").str.upper() == "YES").astype(int)
        )

        if fitting:
            self._median_debt = df["Outstanding_Debt"].median()
        df["CreditMix_x_Debt"] = df["Credit_Mix_Ordinal"] * df["Outstanding_Debt"].fillna(self._median_debt)
        if fitting:
            self._p99_cxd = df["CreditMix_x_Debt"].quantile(0.99)
        df["CreditMix_x_Debt"] = df["CreditMix_x_Debt"].clip(upper=self._p99_cxd)

        if fitting:
            self._median_interest = df["Interest_Rate"].median()
        df["CreditMix_x_Interest"] = df["Credit_Mix_Ordinal"] * df["Interest_Rate"].fillna(self._median_interest)
        if fitting:
            self._p99_cxi = df["CreditMix_x_Interest"].quantile(0.99)
        df["CreditMix_x_Interest"] = df["CreditMix_x_Interest"].clip(upper=self._p99_cxi)

        return df


# -----------------------------------------------------------------------
# 5) ColumnTransformer akhir -- persis notebook section 4.1
# -----------------------------------------------------------------------

def build_column_transformer(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """SimpleImputer(median)+StandardScaler untuk numerik,
    SimpleImputer(most_frequent)+OneHotEncoder untuk kategorikal.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [
            ("num", num_pipeline, numeric_features),
            ("cat", cat_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def build_preprocessing_pipeline(numeric_features: List[str], categorical_features: List[str]) -> Pipeline:
    """Preprocessor lengkap end-to-end: raw dataframe -> matrix siap model.

    Menyatukan seluruh transformer di atas jadi SATU sklearn Pipeline, agar:
    1) Aman di-clone+fit ulang per fold (Optuna CV) tanpa leakage.
    2) Bisa dipakai langsung untuk inference di deployment (bagian 3 UAS),
       cukup panggil `.transform(raw_dataframe_baru)`.
    """
    return Pipeline([
        ("raw_cleaner", RawDataCleaner()),
        ("outlier_clipper", StatisticalOutlierClipper()),
        ("monthly_balance_imputer", FallbackImputer()),
        ("feature_engineer", FeatureEngineer()),
        ("column_transformer", build_column_transformer(numeric_features, categorical_features)),
    ])
