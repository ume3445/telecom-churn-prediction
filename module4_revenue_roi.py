from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from novelty_common import fit_best_calibrated_elastic_net, get_base_splits


def simulate_roi(
    y_true: pd.Series,
    y_prob: pd.Series,
    threshold: float,
    offer_cost: float,
    success_rate: float,
    avg_monthly_revenue: float,
    months_retained: int,
) -> dict:
    targeted = y_prob >= threshold
    y_pred = targeted.astype(int)
    n_targeted = int(targeted.sum())
    n_true_churners_caught = int(((y_true == 1) & (y_pred == 1)).sum())
    n_saved = n_true_churners_caught * success_rate
    revenue_saved = n_saved * avg_monthly_revenue * months_retained
    total_offer_cost = n_targeted * offer_cost
    net_roi = revenue_saved - total_offer_cost
    roi_percent = (net_roi / total_offer_cost * 100.0) if total_offer_cost > 0 else 0.0
    return {
        "threshold": threshold,
        "n_targeted": n_targeted,
        "n_true_churners_caught": n_true_churners_caught,
        "n_saved": n_saved,
        "revenue_saved": revenue_saved,
        "total_offer_cost": total_offer_cost,
        "net_roi": net_roi,
        "roi_percent": roi_percent,
    }


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    out_dir = Path("outputs/roi")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = get_base_splits()
    X_test, y_test = data["X_test"], data["y_test"]
    _, calibrated_model = fit_best_calibrated_elastic_net(data["X_train"], data["y_train"])

    probs = pd.Series(calibrated_model.predict_proba(X_test)[:, 1], index=X_test.index, name="churn_probability")
    monthly_charges_col = "Monthly_Charges" if "Monthly_Charges" in X_test.columns else "MonthlyCharges"
    if monthly_charges_col not in X_test.columns:
        raise ValueError("Monthly charges column not found.")
    monthly_charges = pd.to_numeric(X_test[monthly_charges_col], errors="coerce").fillna(0.0)
    revenue_at_risk = probs * monthly_charges * 12.0

    risk_df = X_test.copy()
    risk_df["churn_probability"] = probs
    risk_df["monthly_charges"] = monthly_charges
    risk_df["revenue_at_risk"] = revenue_at_risk
    risk_df["y_true"] = y_test.values
    risk_df = risk_df.sort_values("revenue_at_risk", ascending=False)
    risk_df.to_csv(out_dir / "revenue_at_risk_ranked.csv", index=True)

    top20 = risk_df[["churn_probability", "monthly_charges", "revenue_at_risk"]].head(20).copy()
    top20.to_csv(out_dir / "top20_revenue_at_risk.csv", index=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        risk_df["churn_probability"],
        risk_df["monthly_charges"],
        c=risk_df["revenue_at_risk"],
        s=(risk_df["revenue_at_risk"] / (risk_df["revenue_at_risk"].max() + 1e-9)) * 280 + 20,
        cmap="viridis",
        alpha=0.75,
        edgecolor="k",
        linewidth=0.2,
    )
    cb = plt.colorbar()
    cb.set_label("Revenue at Risk")
    plt.xlabel("Calibrated Churn Probability")
    plt.ylabel("Monthly Charges")
    plt.title("Revenue-at-Risk Prioritization Map")
    plt.tight_layout()
    plt.savefig(out_dir / "revenue_at_risk_scatter.png", dpi=220)
    plt.close()

    threshold_04_total = float(risk_df.loc[risk_df["churn_probability"] >= 0.4, "revenue_at_risk"].sum())
    with (out_dir / "total_revenue_at_risk_threshold_0_4.txt").open("w", encoding="utf-8") as f:
        f.write(f"Total revenue at risk for churn probability >= 0.4: {threshold_04_total:.2f}\n")

    roi_rows = []
    for thr in [0.2, 0.3, 0.4, 0.5, 0.6]:
        roi_rows.append(
            simulate_roi(
                y_true=y_test.reset_index(drop=True),
                y_prob=probs.reset_index(drop=True),
                threshold=thr,
                offer_cost=50.0,
                success_rate=0.30,
                avg_monthly_revenue=65.0,
                months_retained=12,
            )
        )
    roi_df = pd.DataFrame(roi_rows)
    roi_df.to_csv(out_dir / "roi_simulation_grid.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(roi_df["threshold"].astype(str), roi_df["net_roi"], color="#4C78A8")
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Net ROI vs Decision Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Net ROI")
    plt.tight_layout()
    plt.savefig(out_dir / "net_roi_by_threshold.png", dpi=220)
    plt.close()

    print("\nTop 20 highest revenue-at-risk customers:")
    print(top20.to_string())
    print(f"\nTotal revenue at risk at threshold 0.4: {threshold_04_total:,.2f}")
    print("\nROI simulation table:")
    print(roi_df.to_string(index=False))
    print(f"\nSaved ROI outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
