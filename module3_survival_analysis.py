from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from novelty_common import fit_best_calibrated_elastic_net, get_base_splits


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    out_dir = Path("outputs/survival")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from lifelines import CoxPHFitter, KaplanMeierFitter, WeibullAFTFitter
        from lifelines.utils import concordance_index
    except ImportError:
        print("Missing dependency: lifelines. Install with: pip install lifelines")
        return

    data = get_base_splits()
    df = data["df"].copy()
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    _, calibrated_model = fit_best_calibrated_elastic_net(X_train, y_train)

    tenure_col = "Tenure_Months" if "Tenure_Months" in df.columns else "tenure"
    contract_col = "Contract" if "Contract" in df.columns else "Contract_Type"
    if tenure_col not in df.columns:
        raise ValueError("Tenure column not found for survival analysis.")
    if contract_col not in df.columns:
        raise ValueError("Contract column not found for stratified KM analysis.")

    # Rebuild train/test row-aligned survival frames.
    train_idx = X_train.index
    test_idx = X_test.index
    train_raw = df.loc[train_idx].copy()
    test_raw = df.loc[test_idx].copy()
    train_raw["Event"] = y_train.values
    test_raw["Event"] = y_test.values
    train_raw["Duration"] = pd.to_numeric(train_raw[tenure_col], errors="coerce").fillna(0).clip(lower=0.1)
    test_raw["Duration"] = pd.to_numeric(test_raw[tenure_col], errors="coerce").fillna(0).clip(lower=0.1)

    kmf = KaplanMeierFitter()
    kmf.fit(durations=train_raw["Duration"], event_observed=train_raw["Event"], label="Overall")
    plt.figure(figsize=(8, 5))
    kmf.plot_survival_function(ci_show=True)
    plt.title("Kaplan-Meier Survival Curve (Overall)")
    plt.xlabel("Months")
    plt.ylabel("Survival probability")
    plt.tight_layout()
    plt.savefig(out_dir / "km_overall.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    for c in ["Month-to-month", "One year", "Two year"]:
        grp = train_raw[train_raw[contract_col] == c]
        if grp.empty:
            continue
        km = KaplanMeierFitter()
        km.fit(grp["Duration"], event_observed=grp["Event"], label=c)
        km.plot_survival_function(ci_show=False)
    plt.title("Kaplan-Meier Survival by Contract Type")
    plt.xlabel("Months")
    plt.ylabel("Survival probability")
    plt.tight_layout()
    plt.savefig(out_dir / "km_by_contract.png", dpi=220)
    plt.close()

    # Build Cox/AFT dataset using one-hot encoded features.
    cov_train = pd.get_dummies(X_train, drop_first=True).replace([np.inf, -np.inf], np.nan).fillna(0)
    cov_test = pd.get_dummies(X_test, drop_first=True).replace([np.inf, -np.inf], np.nan).fillna(0)
    cov_train, cov_test = cov_train.align(cov_test, join="left", axis=1, fill_value=0)
    surv_train = cov_train.copy()
    surv_train["Duration"] = train_raw["Duration"].values
    surv_train["Event"] = train_raw["Event"].values
    surv_test = cov_test.copy()
    surv_test["Duration"] = test_raw["Duration"].values
    surv_test["Event"] = test_raw["Event"].values

    cox = CoxPHFitter(penalizer=0.1)
    cox.fit(surv_train, duration_col="Duration", event_col="Event")
    cox_summary = cox.summary.copy()
    cox_summary["hazard_ratio"] = np.exp(cox_summary["coef"])
    cox_top = cox_summary.reindex(cox_summary["coef"].abs().sort_values(ascending=False).head(10).index)
    cox_top_export = cox_top[["coef", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].reset_index()
    first_col = cox_top_export.columns[0]
    cox_top_export.rename(columns={first_col: "feature"}, inplace=True)
    cox_top_export.to_csv(out_dir / "cox_top10_hazard_ratios.csv", index=False)

    plt.figure(figsize=(8, 6))
    plot_df = cox_top_export.iloc[::-1]
    y_pos = np.arange(len(plot_df))
    plt.errorbar(
        x=plot_df["exp(coef)"],
        y=y_pos,
        xerr=[
            plot_df["exp(coef)"] - plot_df["exp(coef) lower 95%"],
            plot_df["exp(coef) upper 95%"] - plot_df["exp(coef)"],
        ],
        fmt="o",
        capsize=4,
    )
    plt.axvline(1.0, color="red", linestyle="--")
    plt.yticks(y_pos, plot_df["feature"])
    plt.xlabel("Hazard Ratio (95% CI)")
    plt.title("Cox PH Top 10 Hazard Ratios")
    plt.tight_layout()
    plt.savefig(out_dir / "cox_hazard_forest_top10.png", dpi=220)
    plt.close()

    aft = WeibullAFTFitter(penalizer=0.1)
    aft.fit(surv_train, duration_col="Duration", event_col="Event")

    # C-index on test using Cox risk scores
    test_risk = cox.predict_partial_hazard(surv_test.drop(columns=["Duration", "Event"]))
    c_index = concordance_index(
        event_times=surv_test["Duration"],
        predicted_scores=-test_risk.values.ravel(),
        event_observed=surv_test["Event"],
    )

    churn_risk = calibrated_model.predict_proba(X_test)[:, 1]
    top5_idx = np.argsort(churn_risk)[-5:][::-1]
    top5_cov = surv_test.drop(columns=["Duration", "Event"]).iloc[top5_idx]
    pred_median = aft.predict_median(top5_cov)
    top5_table = pd.DataFrame(
        {
            "customer_index": X_test.iloc[top5_idx].index.astype(str),
            "churn_probability": churn_risk[top5_idx],
            "predicted_median_survival_months": pred_median.values,
        }
    )
    top5_table.to_csv(out_dir / "top5_predicted_median_survival.csv", index=False)

    with (out_dir / "cox_test_c_index.txt").open("w", encoding="utf-8") as f:
        f.write(f"C-index: {c_index:.4f}\n")

    print("\nTop 10 Cox hazard ratios (with confidence intervals):")
    print(cox_top_export.to_string(index=False))
    print(f"\nCox test C-index: {c_index:.4f}")
    print("\nTop 5 highest-risk customers and predicted median survival (months):")
    print(top5_table.to_string(index=False))
    print(f"\nSaved survival outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
