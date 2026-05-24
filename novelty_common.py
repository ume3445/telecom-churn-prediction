from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


def load_data() -> pd.DataFrame:
    for file_name in ["Telco_customer_churn.xlsx", "Telco_customer_churn-2.xlsx"]:
        fp = Path(file_name)
        if fp.exists():
            return pd.read_excel(fp)
    raise FileNotFoundError("Dataset file not found. Expected Telco_customer_churn.xlsx or Telco_customer_churn-2.xlsx")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip().replace(" ", "_").replace("-", "_").replace("/", "_") for c in out.columns]
    return out


def find_existing_column(df: pd.DataFrame, names: List[str]) -> str:
    for name in names:
        if name in df.columns:
            return name
    return ""


def prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    total_charges_col = find_existing_column(df, ["TotalCharges", "Total_Charges"])
    tenure_col = find_existing_column(df, ["tenure", "Tenure_Months", "TenureMonths"])
    if total_charges_col:
        df[total_charges_col] = pd.to_numeric(df[total_charges_col], errors="coerce")
        if tenure_col:
            mask = df[total_charges_col].isna() & (df[tenure_col] == 0)
            df.loc[mask, total_charges_col] = 0

    target_col = find_existing_column(df, ["Churn_Value"])
    if target_col:
        y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)
    else:
        churn_text_col = find_existing_column(df, ["Churn", "Churn_Label"])
        if not churn_text_col:
            raise ValueError("Could not find churn target column.")
        y = df[churn_text_col].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0}).fillna(0).astype(int)

    leakage_cols = [
        c
        for c in ["customerID", "CustomerID", "Customer_ID", "Churn_Label", "Churn_Value", "Churn_Score", "Churn_Reason", "Count", "Lat_Long"]
        if c in df.columns
    ]
    X = df.drop(columns=leakage_cols, errors="ignore").copy()
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))])
    return ColumnTransformer([("num", num_pipe, numeric_features), ("cat", cat_pipe, categorical_features)])


def get_base_splits() -> Dict:
    df = standardize_columns(load_data())
    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    return {"df": df, "X": X, "y": y, "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


def fit_best_calibrated_elastic_net(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Pipeline, CalibratedClassifierCV]:
    preprocessor = build_preprocessor(X_train)
    best_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    penalty="elasticnet",
                    class_weight="balanced",
                    solver="saga",
                    max_iter=8000,
                    random_state=42,
                    C=0.1,
                    l1_ratio=0.2,
                ),
            ),
        ]
    )
    best_pipeline.fit(X_train, y_train)
    calibrated = CalibratedClassifierCV(estimator=best_pipeline, cv=5, method="isotonic")
    calibrated.fit(X_train, y_train)
    return best_pipeline, calibrated
