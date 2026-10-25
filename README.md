# Bank Customer Churn Prediction & Risk Scoring

## Overview
An educational machine-learning project that estimates the probability of bank customer churn, compares Logistic Regression, Decision Tree, Random Forest and Gradient Boosting classifiers, and presents customer-level predictions through a Streamlit dashboard.

## Project objectives
- Inspect the customer dataset and summarize churn patterns.
- Build features that represent customer profile, product usage, and engagement.
- Compare four classifiers using accuracy, precision, recall, F1-score, ROC-AUC and 5-fold cross-validated ROC-AUC.
- Estimate a churn probability for a hypothetical customer profile.
- Explore global feature importance, a SHAP summary plot and partial dependence plots.
- Demonstrate probability distribution and what-if scenario analysis in Streamlit.

## Dataset
The working dataset has 10,000 rows and 14 original columns. The target is `Exited` (0 = stayed, 1 = exited). The observed target counts are 7,963 stayed (79.63%) and 2,037 exited (20.37%). No missing values were found in the supplied CSV during the project inspection.

**Data rights:** Before publishing, verify the dataset's source, licence, and redistribution permission. The repository instructions below exclude `data/` by default. Do not upload data unless you are authorized to redistribute it.

## Workflow
1. Inspect the dataset (`inspect_data.py`).
2. Review churn distribution (`eda.py`) and descriptive group rates (`feature_analysis.py`).
3. Prepare the feature set: remove `CustomerId`, `Surname`, and `Year`; one-hot encode `Geography` and `Gender`; scale numerical fields within a scikit-learn pipeline.
4. Engineer `BalanceSalaryRatio`, `ProductDensity`, `EngagementProduct`, and `AgeTenureInteraction`.
5. Use an 80/20 stratified train/test split with `random_state=42`.
6. Compare Logistic Regression, Decision Tree, Random Forest and Gradient Boosting with 5-fold cross-validation (`compare_models.py`).
7. Generate Random Forest feature importance (`feature_importance.py`), SHAP explanations (`shap_explainability.py`) and partial dependence plots (`partial_dependence.py`).
8. Run the Streamlit dashboard (`app.py`).

## Reported model results
Test-set results on the 2,000-row stratified holdout (default 0.50 threshold). Re-run `compare_models.py` to reproduce them.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | CV ROC-AUC (5-fold) |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8060 | 0.5704 | 0.1892 | 0.2841 | 0.7735 | 0.7625 ± 0.0203 |
| Decision Tree | 0.7620 | 0.4513 | 0.7862 | 0.5735 | 0.8410 | 0.8323 ± 0.0080 |
| Random Forest | 0.8410 | 0.6121 | 0.5971 | 0.6045 | 0.8522 | 0.8470 ± 0.0124 |
| Gradient Boosting | 0.8705 | 0.7913 | 0.4939 | 0.6082 | 0.8674 | 0.8631 ± 0.0093 |

Gradient Boosting scored highest on accuracy, precision, F1 and ROC-AUC. Random Forest has higher recall than Gradient Boosting and is the model used in the dashboard. The Decision Tree has the highest recall but low precision. The Random Forest confusion matrix on the 2,000-row test split was `[[1439, 154], [164, 243]]` (rows = actual class; columns = predicted class; class 0 = stayed, class 1 = exited). Thus, 164 customers who exited were missed at the default classification threshold, and 154 stayed customers were flagged as churners. Threshold selection should depend on retention costs and should be evaluated on validation data.

## Descriptive observations
- Overall churn in the supplied dataset was 20.37%.
- Observed churn rate by geography: France 16.15%, Germany 32.44%, Spain 16.67%.
- Observed churn rate by active-member status: inactive 26.85%, active 14.27%.
- Observed churn rate by gender: Female 25.07%, Male 16.46%.
- Observed churn rate by number of products: 1 product 27.71%, 2 products 7.58%, 3 products 82.71%, 4 products 100.00%.

These are descriptive associations in this dataset, not causal effects or guaranteed patterns in other banks. The very high rates for 3 and 4 products rest on small groups (266 and 60 customers) and should be checked against data provenance before drawing operational conclusions.

## Install and run
Use Python 3.12 or another compatible Python version. From the project root:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run app.py
```

The dashboard expects `models/churn_model.pkl`, `images/feature_importance.png`, `images/shap_summary.png`, `images/model_comparison.png`, `images/partial_dependence.png` and (for the probability-distribution chart) `models/scored_customers.csv` at the paths shown. The scores file is created by `python make_scores.py` and contains only the model score and outcome, not customer details. The supplied trained model is about 37 MB. Only load a pickle model you created or otherwise trust; pickle files can execute code when loaded.

To regenerate the model and feature-importance chart, make sure you have authorized data at `data/churn.csv`, then run:

```bash
python feature_importance.py
```

For the comparison and confusion matrix:

```bash
python compare_models.py
```

For SHAP:

```bash
python shap_explainability.py
```

## Dashboard features verified from the supplied source
- Customer profile inputs and predicted churn probability.
- A classification label using a fixed 0.50 threshold.
- Random Forest feature-importance chart.
- Probability distribution across the supplied dataset, split by actual status.
- What-if scenario comparison for active-member status and number of products.

## Limitations and responsible use
- This is a portfolio/educational prototype, not a production banking decision system.
- Reported performance depends on the supplied dataset, random split, and chosen threshold.
- The target is imbalanced; accuracy alone is not sufficient.
- No cross-validation, threshold tuning, calibration study, fairness audit, or external validation is documented in the supplied scripts.
- `Year` was excluded because its meaning was not verified; document this assumption if source documentation becomes available.
- Feature importance and SHAP describe model behaviour; they do not prove causality.
- Do not make adverse customer decisions solely from this prototype. Human review, privacy protections, fairness testing, and governance would be required before any real-world use.

## Repository contents
The project scripts include `app.py`, `inspect_data.py`, `eda.py`, `feature_analysis.py`, `preprocessing.py`, `train_model.py`, `compare_models.py`, `feature_importance.py`, and `shap_explainability.py`. Generated plots are in `images/`.

## Disclaimer
For educational and portfolio purposes only. It is not financial advice and is not validated for production use.