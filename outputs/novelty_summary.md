# Novelty Extension Summary

## 1) What Each New Module Adds and Why It Is Novel

### Module 1 — SHAP-Based Instance-Level Explainability
- Adds local and global explanation capability beyond coefficient-level interpretation.
- Introduces customer-specific attribution (waterfall plots) so retention teams can see *why* each high-risk customer is predicted to churn.
- Provides feature-level interaction visibility via dependence plots, improving actionability for operations and campaign design.

### Module 2 — Uplift Modeling (Treatment Response)
- Adds treatment-effect thinking to churn modeling: not just "who will churn," but "who can be influenced."
- Uses a T-Learner strategy to estimate incremental benefit from retention intervention.
- Introduces strategic segmentation into Persuadables, Sure Things, Lost Causes, and Do Not Disturb cohorts.

### Module 3 — Survival Analysis (Time-to-Churn)
- Adds a temporal layer to churn risk with Kaplan-Meier and Cox/Weibull survival models.
- Moves decision-making from binary classification to expected churn timing.
- Enables retention teams to prioritize both likelihood and urgency.

### Module 4 — Revenue-at-Risk and ROI Simulation
- Connects churn probability to financial exposure through annualized revenue-at-risk scoring.
- Adds decision simulation for campaign economics across multiple thresholds.
- Enables threshold selection based on business return rather than model metrics alone.

## 2) Key Findings from Each Module

### Module 1 Findings (SHAP)
- Most influential features by mean absolute SHAP value:
  - `num__Tenure_Months` (0.9162)
  - `cat__Dependents_Yes` (0.5327)
  - `cat__Contract_Two year` (0.3946)
  - `cat__Internet_Service_Fiber optic` (0.2816)
  - `cat__Contract_One year` (0.2556)
- SHAP outputs confirm that contract structure, tenure, and service/payment characteristics are dominant churn drivers.
- All required SHAP artifacts were generated in `outputs/shap/`:
  - Beeswarm summary (top 15)
  - 3 high-risk waterfall plots
  - 3 dependence plots

### Module 2 Findings (Uplift)
- Mean uplift score (`P(churn|treated)-P(churn|control)`) on test set: **-0.0267**, implying positive average intervention effect under the simulation setup.
- Segment distribution:
  - Persuadables: **474**
  - Lost Causes: **473**
  - Sure Things: **231**
  - Do Not Disturb: **231**
- Top 10% uplift-help cohort has much higher expected incremental gain than overall population (0.2718 vs 0.0267), supporting uplift-ranked targeting.
- Required plots generated in `outputs/uplift/`:
  - Uplift histogram
  - Qini-style cumulative gains curve
  - 2x2 segmentation plot

### Module 3 Findings (Survival)
- Cox model test concordance index: **0.9028**, indicating strong ranking performance in time-to-event ordering.
- Kaplan-Meier curves show strong contract-based survival separation, with month-to-month customers churning earlier than longer-contract customers.
- Predicted median survival among the top 5 highest-risk customers ranges from **3.24 to 11.05 months**, giving direct intervention timing signals.
- Required survival artifacts generated in `outputs/survival/`:
  - Overall and contract-stratified KM plots
  - Cox top-10 hazard ratio forest plot
  - Top-5 median survival table

### Module 4 Findings (Revenue + ROI)
- Total revenue at risk for customers with churn probability >= 0.4: **$242,525.92**.
- ROI simulation under default assumptions (`$50` offer cost, `30%` success, `12` months retained) is positive at all tested thresholds:
  - Threshold 0.2: Net ROI **$41,932**
  - Threshold 0.3: Net ROI **$39,518**
  - Threshold 0.4: Net ROI **$36,362**
  - Threshold 0.5: Net ROI **$33,072**
  - Threshold 0.6: Net ROI **$25,336**
- ROI percentage increases at higher thresholds due lower contact cost intensity, while absolute retained-revenue capture is larger at lower thresholds.
- Required outputs generated in `outputs/roi/`, including top-20 revenue-at-risk ranking and ROI bar chart.

## 3) Revised Final Recommendation

We recommend deploying the **calibrated Elastic Net model** as the scoring backbone, then layering **uplift-aware targeting** and **revenue-at-risk prioritization** for campaign execution. Operationally, a two-stage policy is strongest: (1) rank customers by uplift-help among high predicted churn-risk customers, then (2) prioritize by revenue-at-risk within the persuadable group. This approach improves intervention efficiency, aligns retention spend with incremental impact, and ties model outputs directly to financial decision-making. In the current simulation settings, thresholds in the **0.3–0.4** range provide a practical balance between capture volume and campaign economics.

## 4) Limitation of Simulated Treatment and Production Replacement Plan

The uplift module currently relies on **simulated treatment assignment and synthetic treatment effect**, which is useful for method development but does not establish real causal impact. Simulated uplift can overstate or misstate true intervention response because treatment assignment is not tied to real-world policy, channel effects, or customer self-selection. In production, this should be replaced with a randomized controlled retention experiment (A/B test): randomize eligible customers to treatment/control, log intervention exposure and outcomes, estimate heterogeneous treatment effects from observed data, and then retrain/validate uplift models on real causal labels. This converts the current proof-of-concept into a decision system with defensible causal inference.
