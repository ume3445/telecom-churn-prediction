from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

from novelty_common import fit_best_calibrated_elastic_net, get_base_splits


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    out_dir = Path("outputs/shap")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import shap
    except ImportError:
        print("Missing dependency: shap. Install with: pip install shap")
        return

    data = get_base_splits()
    X_train, X_test = data["X_train"], data["X_test"]
    y_test = data["y_test"]

    best_pipeline, calibrated_model = fit_best_calibrated_elastic_net(X_train, data["y_train"])
    # LinearExplainer requires a linear model; we use the calibrated model's fitted base estimator.
    fitted_base = calibrated_model.estimator
    preprocessor = fitted_base.named_steps["preprocessor"]
    linear_model = fitted_base.named_steps["model"]

    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    if sparse.issparse(X_train_t):
        X_train_t = X_train_t.toarray()
    if sparse.issparse(X_test_t):
        X_test_t = X_test_t.toarray()
    feature_names = preprocessor.get_feature_names_out().tolist()
    X_train_df = pd.DataFrame(X_train_t, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_t, columns=feature_names, index=X_test.index)

    explainer = shap.LinearExplainer(linear_model, X_train_df)
    shap_values = explainer.shap_values(X_test_df)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap.summary_plot(shap_values, X_test_df, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary_beeswarm_top15.png", dpi=220)
    plt.close()

    calibrated_proba = calibrated_model.predict_proba(X_test)[:, 1]
    top3_idx = np.argsort(calibrated_proba)[-3:][::-1]
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = np.array(expected_value).ravel()[0]

    for rank, row_idx in enumerate(top3_idx, start=1):
        exp = shap.Explanation(
            values=shap_values[row_idx],
            base_values=expected_value,
            data=X_test_df.iloc[row_idx].values,
            feature_names=feature_names,
        )
        shap.plots.waterfall(exp, max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(out_dir / f"waterfall_top_risk_{rank}.png", dpi=220)
        plt.close()

    mean_abs = np.abs(shap_values).mean(axis=0)
    top3_feat_idx = np.argsort(mean_abs)[-3:][::-1]
    for pos, feat_idx in enumerate(top3_feat_idx, start=1):
        feat_name = feature_names[feat_idx]
        shap.dependence_plot(feat_name, shap_values, X_test_df, show=False)
        plt.tight_layout()
        safe_name = feat_name.replace("/", "_").replace(" ", "_")
        plt.savefig(out_dir / f"dependence_top{pos}_{safe_name}.png", dpi=220)
        plt.close()

    direction = np.sign(shap_values.mean(axis=0))
    direction_label = np.where(direction >= 0, "positive", "negative")
    summary = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": mean_abs,
                "mean_shap": shap_values.mean(axis=0),
                "direction": direction_label,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .head(10)
    )
    summary.to_csv(out_dir / "top10_mean_abs_shap.csv", index=False)

    print("\nTop 10 features by mean |SHAP|:")
    print(summary.to_string(index=False))
    print(f"\nSaved SHAP outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
