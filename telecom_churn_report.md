# Interpretable Churn Prediction for Telecom Decision Support: A Comparative Study of Regularized Logistic Models and Probability Calibration

## 1. Introduction

Customer churn remains one of the most consequential challenges for subscription-based telecom providers. Even moderate increases in customer attrition can materially reduce recurring revenue, increase acquisition costs, and weaken long-term customer lifetime value. In this context, predictive analytics offers a practical path from reactive retention efforts to proactive intervention.

In this study, we develop and compare interpretable churn prediction models using the Telco Customer Churn dataset. Our goals are to (i) deliver strong predictive performance, (ii) preserve model transparency for managerial trust and auditability, (iii) evaluate probability calibration for risk-based targeting, and (iv) translate model outputs into decision-ready business actions. We emphasize interpretable AI not only as a technical preference, but as an operational requirement for deployment in customer-facing workflows.

## 2. Dataset Description

We use the publicly available Telco Customer Churn dataset, a widely used benchmark in churn analytics. The dataset contains 7,043 customer records and 33 raw columns spanning:

- **Demographics:** gender, senior status, partner/dependents, geography
- **Service attributes:** internet service type, security/backup/support options, streaming services, phone line characteristics
- **Billing and contract behavior:** monthly charges, total charges, payment method, paperless billing, contract type
- **Target outcome:** churn status

The target variable is binary churn (`Churn_Value`: 1 = churned, 0 = retained). Class balance indicates meaningful but manageable imbalance (approximately 26.5% churners, 73.5% non-churners), appropriate for cost-sensitive churn modeling.

## 3. Data Preparation

Data preparation was designed to maximize robustness while preventing information leakage.

- **Column standardization:** names were normalized (trimmed and converted to underscore format) for reproducible processing.
- **`Total_Charges` cleaning:** converted from string/object to numeric with coercion; missing values were handled according to tenure logic (tenure = 0 implies zero cumulative charges).
- **Leakage control:** identifier and post-outcome columns were removed from predictors, including `CustomerID`, `Churn_Label`, `Churn_Value` (as predictor), `Churn_Score`, `Churn_Reason`, `Count`, and `Lat_Long`.
- **Feature encoding and preprocessing:**
  - Numeric features: median imputation + standardization
  - Categorical features: most-frequent imputation + one-hot encoding (`drop='first'`)
- **Validation protocol:** stratified 80/20 train-test split and cross-validation performed within pipelines.
- **Leakage prevention by design:** preprocessing and model fitting were encapsulated in scikit-learn pipelines to ensure transformations were learned on training folds only.

This setup supports fair model comparison and reduces optimism bias in reported performance.

## 4. Methodology

We evaluated four logistic-family models:

- **Baseline Logistic Regression** (`class_weight='balanced'`)
- **Ridge Logistic Regression (L2)**
- **Lasso Logistic Regression (L1)**
- **Elastic Net Logistic Regression (L1 + L2)**

Logistic regression was selected as the core model family because it balances predictive utility with direct interpretability through coefficients and odds ratios. For churn management, this interpretability is strategically important: stakeholders need to understand *why* a customer is high risk, not only *that* they are high risk.

### Hyperparameter tuning and validation

- 5-fold stratified cross-validation
- ROC AUC as primary tuning metric
- Grid search:
  - `C ∈ {0.01, 0.1, 1, 10, 100}`
  - `l1_ratio ∈ {0.2, 0.5, 0.8}` for Elastic Net

### Evaluation metrics

We report:

- ROC AUC
- PR AUC (Average Precision)
- Accuracy
- Precision
- Recall
- F1-score
- Brier Score
- Confusion Matrix

This combination captures ranking quality, minority-class retrieval quality, thresholded classification behavior, and probability reliability.

## 5. Results

### 5.1 Comparative model performance

Across models, performance is consistently strong and tightly clustered, indicating that regularized logistic approaches are robust for this problem.

- **Elastic Net Logistic**: Test ROC AUC = **0.8484**, PR AUC = **0.6452**
- **Ridge Logistic (L2)**: Test ROC AUC = **0.8477**, PR AUC = **0.6466**
- **Lasso Logistic (L1)**: Test ROC AUC = **0.8476**, PR AUC = **0.6466**
- **Baseline Logistic**: Test ROC AUC = **0.8366**, PR AUC = **0.6257**

Interpretation: Elastic Net is marginally strongest in ROC ranking, while Ridge/Lasso are extremely close and slightly stronger on PR AUC. The differences are small enough that operational considerations (interpretability/calibration) become decisive.

### 5.2 Curves and discrimination behavior

- **ROC curves (Figure 1):** Curves for Elastic Net, Ridge, and Lasso are nearly overlapping and clearly above the diagonal baseline, confirming strong discrimination.
- **Precision-Recall curves (Figure 2):** Curves remain materially above baseline prevalence, supporting practical ability to identify churners under class imbalance.

These plots show that gains from regularization are real but incremental; model choice should therefore reflect deployment priorities, not only peak AUC differences.

### 5.3 Confusion matrices (threshold = 0.5)

- **Elastic Net:** `[[759, 276], [82, 292]]`
- **Ridge:** `[[770, 265], [82, 292]]`
- **Lasso:** `[[754, 281], [82, 292]]`
- **Baseline:** `[[776, 259], [91, 283]]`

At the default threshold, regularized models improve churn capture over baseline (fewer false negatives), with manageable precision trade-offs.

## 6. Threshold Analysis

A default threshold of 0.5 is rarely optimal in churn operations because retention programs are cost-sensitive and often prioritize recall.

For **Elastic Net**:

- Threshold 0.2: Precision 0.396, Recall 0.971
- Threshold 0.3: Precision 0.433, Recall 0.912
- Threshold 0.4: Precision 0.475, Recall 0.869
- Threshold 0.5: Precision 0.514, Recall 0.781
- Threshold 0.6: Precision 0.564, Recall 0.711

This profile illustrates the core business trade-off:

- Lower thresholds capture more churners (high recall) but trigger more outreach (higher false positives).
- Higher thresholds improve targeting efficiency (precision) but miss more at-risk customers.

**Recommended threshold (business-conditional):**

- **0.4** is a strong operational compromise when the business seeks higher churn capture without the substantial alert volume seen at 0.2–0.3.
- **0.3** is appropriate when churn miss-cost is very high and retention contact capacity is sufficient.

## 7. Calibration Analysis

Calibration assesses whether predicted probabilities correspond to actual event frequencies (e.g., customers scored at 0.70 should churn about 70% of the time in aggregate). This is critical when probabilities drive budget allocation and campaign tiering.

Brier scores:

- **Elastic Net (raw):** 0.1641
- **Ridge (raw):** 0.1637
- **Elastic Net (calibrated):** **0.1361**

As shown in **Figure 3 (calibration curves)**, post-calibration probabilities are substantially more reliable. This matters directly for decision-making: calibrated risk scores enable more accurate estimation of expected retention uplift and better ROI targeting across customer segments.

## 8. Interpretability Analysis

Logistic coefficients and odds ratios enable actionable interpretation of churn drivers.

### 8.1 Consistent churn risk factors (positive coefficients)

Across regularized models, higher churn likelihood is associated with:

- **Fiber optic internet service** (Ridge OR approx. 1.74; Elastic Net OR approx. 1.76; Lasso OR approx. 1.80)
- **Electronic check payment method** (OR approx. 1.47)
- **Paperless billing** (OR approx. 1.35–1.37)
- Higher monthly/total charges (moderate positive effect)

Business reading: customers with high-intensity service and potentially friction-prone payment/billing behaviors appear more volatile.

### 8.2 Protective factors (negative coefficients)

Strong retention-associated factors include:

- **Dependents = Yes** (Ridge OR approx. 0.23; Elastic Net OR approx. 0.23; Lasso OR approx. 0.22)
- **Longer contract commitments**:
  - One-year contract (OR approx. 0.50–0.51)
  - Two-year contract (OR approx. 0.29–0.32)
- **Longer tenure** (OR approx. 0.36–0.40)
- Security/support add-ons (moderate protective effects)

Business reading: contract lock-in and relationship maturity remain dominant churn suppressors.

### 8.3 Sparsity and feature selection

Lasso shrank a large number of coefficients to zero or near-zero, producing a more compact explanation space. This is useful for communication and policy simplification, though predictive gains over Ridge/Elastic Net were minimal.

## 9. Business Implications

The modeling outputs support concrete telecom retention actions:

- **Risk-tiered targeting:** use calibrated churn probabilities to define intervention tiers (e.g., high/medium/low).
- **Contract strategy:** prioritize month-to-month customers for conversion offers to longer terms.
- **Payment experience interventions:** proactively support customers on electronic check pathways with friction-reduction campaigns.
- **Offer economics:** apply threshold-based targeting based on customer value and intervention cost; lower thresholds for high-CLTV segments, higher thresholds where outreach cost is high.
- **Automation integration:** deploy model scoring in CRM workflows with scheduled batch scoring, threshold-based trigger rules, and campaign attribution tracking.

In practice, this enables disciplined retention allocation instead of broad, cost-inefficient outreach.

## 10. Model Comparison Insight and Final Recommendation

- **Best predictive model:** **Elastic Net Logistic** (highest ROC AUC)
- **Most interpretable model:** **Ridge Logistic (L2)** (stable coefficients, clear driver patterns)
- **Best calibrated model:** **Calibrated Elastic Net** (best Brier score)

### Final recommendation

For production deployment, we recommend **Elastic Net with calibration**, paired with a business-adjusted operating threshold (starting point: 0.4). This configuration offers the best balance of ranking quality, probability reliability, and interpretability sufficient for stakeholder trust.

## 11. Limitations

Several limitations should inform interpretation:

- The dataset is a benchmark dataset and may not fully reflect current market dynamics or provider-specific customer behavior.
- External validation on a temporally separated or geographically distinct cohort was not performed.
- Logistic models capture linear effects in log-odds; nonlinear interactions may remain under-modeled.
- Some location-level dummy variables may encode local artifacts not generalizable to other telecom contexts.
- Causal inference is out of scope; coefficients indicate association, not causal impact.

## 12. Conclusion

This study demonstrates that interpretable logistic-family models can achieve strong churn prediction performance while remaining operationally transparent. Regularized variants outperform the baseline model, and calibration materially improves probability quality for decision-making. The resulting framework is not only predictive but deployable: it supports threshold tuning, ROI-aware targeting, and clear communication of churn drivers to business stakeholders.

Future work should include temporal external validation, uplift modeling for intervention effectiveness, and monitoring frameworks for drift and recalibration in production environments.
