import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibrationDisplay, CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
import matplotlib.pyplot as plt


def load_data() -> pd.DataFrame:
    candidate_files = [
        "Telco_customer_churn.xlsx",
        "Telco_customer_churn-2.xlsx",
    ]
    for file_name in candidate_files:
        file_path = Path(file_name)
        if file_path.exists():
            print(f"Loading dataset: {file_path}")
            return pd.read_excel(file_path)
    raise FileNotFoundError(
        "Could not find Telco dataset file. Expected one of: "
        + ", ".join(candidate_files)
    )


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        c.strip().replace(" ", "_").replace("-", "_").replace("/", "_") for c in out.columns
    ]
    return out


def find_existing_column(df: pd.DataFrame, names: List[str]) -> str:
    for name in names:
        if name in df.columns:
            return name
    return ""


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
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
            raise ValueError("Could not find target column in dataset.")
        y = df[churn_text_col].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
        y = y.fillna(0).astype(int)

    leakage_cols = [
        c for c in [
            "customerID", "CustomerID", "Customer_ID",
            "Churn_Label", "Churn_Value", "Churn_Score", "Churn_Reason",
            "Count", "Lat_Long",
        ]
        if c in df.columns
    ]
    X = df.drop(columns=leakage_cols, errors="ignore").copy()
    return X, y


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ]
    )
    return preprocessor, numeric_features, categorical_features


def build_result_dict(
    model_name: str,
    cv_roc_auc: float,
    y_test: pd.Series,
    y_pred,
    y_proba,
    fpr,
    tpr,
    precision_curve,
    recall_curve,
    extra: Dict = None,
) -> Dict:
    d = {
        "Model": model_name,
        "CV_ROC_AUC": cv_roc_auc,
        "Test_Accuracy": accuracy_score(y_test, y_pred),
        "Test_Precision": precision_score(y_test, y_pred, zero_division=0),
        "Test_Recall": recall_score(y_test, y_pred, zero_division=0),
        "Test_F1": f1_score(y_test, y_pred, zero_division=0),
        "Test_ROC_AUC": roc_auc_score(y_test, y_proba),
        "Test_PR_AUC": average_precision_score(y_test, y_proba),
        "Brier_Score": brier_score_loss(y_test, y_proba),
        "Confusion_Matrix": confusion_matrix(y_test, y_pred).tolist(),
        "fpr": fpr,
        "tpr": tpr,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve,
    }
    if extra:
        d.update(extra)
    return d


def evaluate_model(
    model_name: str,
    estimator: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv: StratifiedKFold,
) -> Dict:
    cv_scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    estimator.fit(X_train, y_train)
    y_proba = estimator.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
    return build_result_dict(
        model_name, float(np.mean(cv_scores)), y_test, y_pred, y_proba,
        fpr, tpr, precision_curve, recall_curve,
    )


def tune_and_evaluate(
    model_name: str,
    base_pipeline: Pipeline,
    param_grid: Dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv: StratifiedKFold,
) -> Tuple[Dict, Pipeline]:
    search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    y_proba = best.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
    result = build_result_dict(
        model_name, search.best_score_, y_test, y_pred, y_proba,
        fpr, tpr, precision_curve, recall_curve,
        extra={"Best_Params": search.best_params_},
    )
    return result, best


def extract_coefficients(model_name: str, fitted_pipeline: Pipeline, output_dir: Path) -> pd.DataFrame:
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    classifier = fitted_pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    coefs = classifier.coef_.ravel()
    coef_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": coefs,
            "Odds_Ratio": np.exp(coefs),
        }
    )
    coef_df["Absolute_Coefficient"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("Coefficient", ascending=False)

    out_path = output_dir / f"coefficients_{model_name.replace(' ', '_').lower()}.csv"
    coef_df.to_csv(out_path, index=False)

    print(f"\n=== {model_name} Top Positive Churn Drivers ===")
    print(coef_df.head(10)[["Feature", "Coefficient", "Odds_Ratio"]])

    print(f"\n=== {model_name} Top Negative Churn Drivers ===")
    print(coef_df.tail(10).sort_values("Coefficient")[["Feature", "Coefficient", "Odds_Ratio"]])

    if "lasso" in model_name.lower():
        near_zero = (coef_df["Coefficient"].abs() < 1e-4).sum()
        print(f"\n{model_name}: near-zero coefficients (sparse signal): {near_zero}")
    return coef_df


def threshold_analysis(
    model_name: str,
    fitted_model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
) -> pd.DataFrame:
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]
    y_proba = fitted_model.predict_proba(X_test)[:, 1]
    rows = []
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        rows.append(
            {
                "Model": model_name,
                "Threshold": thr,
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1": f1_score(y_test, y_pred, zero_division=0),
                "Confusion_Matrix": confusion_matrix(y_test, y_pred).tolist(),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(output_path, index=False)
    print(f"\n=== Threshold Analysis: {model_name} ===")
    print(table)
    return table


def try_probit(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict:
    try:
        import statsmodels.api as sm

        X_train_d = pd.get_dummies(X_train, drop_first=True)
        X_test_d = pd.get_dummies(X_test, drop_first=True)
        X_train_d, X_test_d = X_train_d.align(X_test_d, join="left", axis=1, fill_value=0)
        X_train_d = X_train_d.fillna(0).astype(float)
        X_test_d = X_test_d.fillna(0).astype(float)

        X_train_sm = sm.add_constant(X_train_d, has_constant="add")
        X_test_sm = sm.add_constant(X_test_d, has_constant="add")

        model = sm.Probit(y_train, X_train_sm)
        fitted = model.fit(disp=False)
        y_proba = fitted.predict(X_test_sm)
        y_pred = (y_proba >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
        return build_result_dict(
            "Probit (statsmodels)", np.nan, y_test, y_pred, y_proba,
            fpr, tpr, precision_curve, recall_curve,
            extra={"Best_Params": {}},
        )
    except Exception as exc:
        # Probit can fail due to singular matrices or perfect separation on sparse high-dimensional one-hot data.
        print(f"\nSkipping Probit model due to stability/practicality issue: {exc}")
        return {}


def plot_curves(results: List[Dict], output_dir: Path) -> None:
    plt.figure(figsize=(9, 7))
    for r in results:
        if r.get("fpr") is not None:
            plt.plot(r["fpr"], r["tpr"], label=f'{r["Model"]} (AUC={r["Test_ROC_AUC"]:.3f})')
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 7))
    for r in results:
        if r.get("precision_curve") is not None:
            plt.plot(
                r["recall_curve"],
                r["precision_curve"],
                label=f'{r["Model"]} (AP={r["Test_PR_AUC"]:.3f})',
            )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curves.png", dpi=200)
    plt.close()


def run_calibration_comparison(
    top_models: List[Tuple[str, Pipeline]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
) -> pd.DataFrame:
    rows = []
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in top_models:
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        CalibrationDisplay.from_predictions(y_test, proba, n_bins=10, name=f"{name} (raw)", ax=ax)
        rows.append(
            {
                "Model": name,
                "Variant": "raw",
                "Brier_Score": brier_score_loss(y_test, proba),
            }
        )

    if top_models:
        name, model = top_models[0]
        calibrated = CalibratedClassifierCV(estimator=clone(model), cv=5, method="isotonic")
        calibrated.fit(X_train, y_train)
        cal_proba = calibrated.predict_proba(X_test)[:, 1]
        CalibrationDisplay.from_predictions(
            y_test, cal_proba, n_bins=10, name=f"{name} (calibrated)", ax=ax
        )
        rows.append(
            {
                "Model": name,
                "Variant": "calibrated",
                "Brier_Score": brier_score_loss(y_test, cal_proba),
            }
        )

    ax.set_title("Calibration Curves")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "calibration_curves.png", dpi=200)
    plt.close(fig)
    cal_df = pd.DataFrame(rows).sort_values("Brier_Score")
    cal_df.to_csv(output_dir / "calibration_results.csv", index=False)
    return cal_df


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    raw = load_data()
    df = standardize_columns(raw)
    X, y = clean_data(df)
    preprocessor, numeric_features, categorical_features = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    MODEL_CONFIGS = [
        ("Logistic Regression (balanced)", {"solver": "liblinear", "max_iter": 5000}, None),
        ("Ridge Logistic (L2)",  {"penalty": "l2",        "solver": "liblinear", "max_iter": 5000}, {"model__C": [0.01, 0.1, 1, 10, 100]}),
        ("Lasso Logistic (L1)", {"penalty": "l1",         "solver": "liblinear", "max_iter": 5000}, {"model__C": [0.01, 0.1, 1, 10, 100]}),
        ("Elastic Net Logistic", {"penalty": "elasticnet", "solver": "saga",     "max_iter": 8000}, {"model__C": [0.01, 0.1, 1, 10, 100], "model__l1_ratio": [0.2, 0.5, 0.8]}),
    ]

    results: List[Dict] = []
    fitted_models: Dict[str, Pipeline] = {}

    for name, lr_kwargs, param_grid in MODEL_CONFIGS:
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(class_weight="balanced", random_state=42, **lr_kwargs)),
        ])
        if param_grid is None:
            result = evaluate_model(name, pipe, X_train, y_train, X_test, y_test, cv)
            fitted_models[name] = pipe
        else:
            result, best = tune_and_evaluate(name, pipe, param_grid, X_train, y_train, X_test, y_test, cv)
            fitted_models[name] = best
        results.append(result)

    probit_result = try_probit(X_train, y_train, X_test, y_test)
    if probit_result:
        results.append(probit_result)

    comparison_df = pd.DataFrame(results).sort_values(
        by=["Test_ROC_AUC", "Test_PR_AUC"], ascending=False
    )
    export_df = comparison_df.drop(
        columns=["fpr", "tpr", "precision_curve", "recall_curve"], errors="ignore"
    )
    export_df.to_csv(output_dir / "model_comparison_results.csv", index=False)
    print("\n=== Model Comparison ===")
    print(
        export_df[
            [
                "Model",
                "CV_ROC_AUC",
                "Test_ROC_AUC",
                "Test_PR_AUC",
                "Test_Precision",
                "Test_Recall",
                "Test_F1",
                "Brier_Score",
            ]
        ]
    )

    plot_curves(results, output_dir)

    best_model = comparison_df.iloc[0]["Model"]
    best_pipeline = fitted_models.get(best_model, fitted_models["Ridge Logistic (L2)"])
    interp_model = "Ridge Logistic (L2)"
    interp_pipeline = fitted_models[interp_model]

    threshold_analysis(
        best_model,
        best_pipeline,
        X_test,
        y_test,
        output_dir / "threshold_analysis_best_model.csv",
    )
    threshold_analysis(
        interp_model,
        interp_pipeline,
        X_test,
        y_test,
        output_dir / "threshold_analysis_interpretable_model.csv",
    )

    for model_name in [
        "Logistic Regression (balanced)",
        "Ridge Logistic (L2)",
        "Lasso Logistic (L1)",
        "Elastic Net Logistic",
    ]:
        if model_name in fitted_models:
            extract_coefficients(model_name, fitted_models[model_name], output_dir)

    top2_names = comparison_df["Model"].tolist()[:2]
    top2_pipelines = [(n, fitted_models[n]) for n in top2_names if n in fitted_models]

    calibration_df = run_calibration_comparison(
        top2_pipelines, X_train, y_train, X_test, y_test, output_dir
    )
    print("\n=== Calibration Results (lower Brier is better) ===")
    print(calibration_df)


if __name__ == "__main__":
    main()
