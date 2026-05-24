from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from novelty_common import build_preprocessor, get_base_splits


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    out_dir = Path("outputs/uplift")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    data = get_base_splits()
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    treatment = rng.binomial(1, 0.5, size=len(X_train))

    base_rate = y_train.to_numpy().astype(float)
    effect = rng.normal(loc=0.25, scale=0.05, size=len(X_train))
    effect = np.clip(effect, 0.05, 0.45)
    treated_prob = np.clip(base_rate * (1.0 - effect), 0.001, 0.999)
    control_prob = np.clip(base_rate, 0.001, 0.999)
    churn_sim = np.where(treatment == 1, rng.binomial(1, treated_prob), rng.binomial(1, control_prob))

    train_df = X_train.copy()
    train_df["treatment"] = treatment
    y_sim = pd.Series(churn_sim, index=X_train.index, name="churn_sim")

    treated_idx = train_df["treatment"] == 1
    control_idx = train_df["treatment"] == 0
    X_treat, y_treat = X_train.loc[treated_idx], y_sim.loc[treated_idx]
    X_ctrl, y_ctrl = X_train.loc[control_idx], y_sim.loc[control_idx]

    pre_t = build_preprocessor(X_treat)
    pre_c = build_preprocessor(X_ctrl)
    model_t1 = Pipeline(
        [("preprocessor", pre_t), ("model", LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=5000, random_state=42))]
    )
    model_t0 = Pipeline(
        [("preprocessor", pre_c), ("model", LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=5000, random_state=42))]
    )
    model_t1.fit(X_treat, y_treat)
    model_t0.fit(X_ctrl, y_ctrl)

    p_treat = model_t1.predict_proba(X_test)[:, 1]
    p_control = model_t0.predict_proba(X_test)[:, 1]
    uplift_score = p_treat - p_control
    uplift_help = -uplift_score  # larger value => bigger expected churn reduction if treated

    uplift_df = X_test.copy()
    uplift_df["y_true"] = y_test.values
    uplift_df["p_treat"] = p_treat
    uplift_df["p_control"] = p_control
    uplift_df["uplift_score"] = uplift_score
    uplift_df["uplift_help"] = uplift_help
    uplift_df.to_csv(out_dir / "uplift_scores_test.csv", index=True)

    plt.figure(figsize=(8, 5))
    plt.hist(uplift_score, bins=35, color="#4472C4", alpha=0.85, edgecolor="black")
    plt.axvline(0, color="red", linestyle="--", linewidth=1.2)
    plt.title("Uplift Score Distribution (P_treat - P_control)")
    plt.xlabel("Uplift score (negative means treatment helps)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "uplift_distribution_hist.png", dpi=220)
    plt.close()

    # Qini-like cumulative gains curve using modeled incremental benefit proxy.
    ranked = uplift_df.sort_values("uplift_help", ascending=False).reset_index(drop=True)
    ranked["cum_gain"] = ranked["uplift_help"].cumsum()
    n = len(ranked)
    x = np.arange(1, n + 1)
    random_gain = (ranked["uplift_help"].sum() / n) * x

    plt.figure(figsize=(8, 5))
    plt.plot(x, ranked["cum_gain"], label="Model-targeted uplift gains", linewidth=2)
    plt.plot(x, random_gain, label="Random targeting baseline", linestyle="--")
    plt.title("Qini-Style Cumulative Gains Curve")
    plt.xlabel("Customers targeted (sorted by uplift help)")
    plt.ylabel("Cumulative expected churn reduction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "qini_curve.png", dpi=220)
    plt.close()

    # Segmentation matrix: high/low churn risk x high/low uplift help
    churn_risk = p_control
    risk_cut = np.quantile(churn_risk, 0.5)
    uplift_cut = np.quantile(uplift_help, 0.5)
    risk_level = np.where(churn_risk >= risk_cut, "High Risk", "Low Risk")
    uplift_level = np.where(uplift_help >= uplift_cut, "High Uplift", "Low Uplift")

    segment = []
    for r, u in zip(risk_level, uplift_level):
        if r == "High Risk" and u == "Low Uplift":
            segment.append("Sure Things")
        elif r == "High Risk" and u == "High Uplift":
            segment.append("Persuadables")
        elif r == "Low Risk" and u == "Low Uplift":
            segment.append("Lost Causes")
        else:
            segment.append("Do Not Disturb")

    uplift_df["risk_level"] = risk_level
    uplift_df["uplift_level"] = uplift_level
    uplift_df["segment"] = segment
    seg_counts = uplift_df["segment"].value_counts().rename_axis("segment").reset_index(name="count")
    seg_counts.to_csv(out_dir / "segment_counts.csv", index=False)

    plt.figure(figsize=(8, 6))
    color_map = {
        "Sure Things": "#d62728",
        "Persuadables": "#2ca02c",
        "Lost Causes": "#9467bd",
        "Do Not Disturb": "#ff7f0e",
    }
    for seg_name, group in uplift_df.groupby("segment"):
        plt.scatter(
            group["p_control"],
            group["uplift_help"],
            s=16,
            alpha=0.65,
            label=seg_name,
            color=color_map.get(seg_name, "gray"),
        )
    plt.axvline(risk_cut, color="black", linestyle="--", linewidth=1)
    plt.axhline(uplift_cut, color="black", linestyle="--", linewidth=1)
    plt.title("2x2 Segmentation: Churn Risk vs Uplift Potential")
    plt.xlabel("Predicted Churn Risk (Control Model)")
    plt.ylabel("Uplift Potential (-[P_treat - P_control])")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "segmentation_2x2.png", dpi=220)
    plt.close()

    print("\nUplift segment counts:")
    print(seg_counts.to_string(index=False))
    print(f"\nSaved uplift outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
