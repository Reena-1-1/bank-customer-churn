# Bank Customer Churn Prediction

## Project Overview

This project uses machine learning to estimate the probability that a bank customer will leave the bank. It compares Logistic Regression and Random Forest classifiers and provides an interactive Streamlit dashboard for customer risk assessment.

## Objectives

- Explore customer data and identify patterns associated with churn.
- Prepare features for machine learning.
- Compare classification models using appropriate evaluation metrics.
- Estimate churn probability for customer profiles.
- Visualize Random Forest feature importance.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Streamlit

## Machine Learning Workflow

1. Inspect and explore the dataset.
2. Remove customer identifiers and the unverified Year column.
3. Engineer additional numerical features.
4. Split the data into training and test sets using stratification.
5. Preprocess numerical and categorical features.
6. Train Logistic Regression and Random Forest models.
7. Evaluate performance using accuracy, precision, recall, F1-score, and ROC-AUC.
8. Save the trained Random Forest pipeline.
9. Use Streamlit to display predictions and feature importance.

## Model Results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.806 | 0.841 |
| Precision | 0.5704 | 0.6121 |
| Recall | 0.1892 | 0.5971 |
| F1-score | 0.2841 | 0.6045 |
| ROC-AUC | 0.7735 | 0.8522 |

These are results from the project's existing test evaluation. They may vary if the dataset or training procedure changes.

## Run Locally

1. Install Python 3.12 or another compatible Python version.
2. Clone this repository.
3. Create and activate a virtual environment.
4. Install dependencies using `pip install -r requirements.txt`.
5. Ensure the required trained model exists at `models/churn_model.pkl`.
6. Run `python -m streamlit run app.py`.

The dataset is not included in the repository. Retraining the model requires access to an authorized copy of the dataset and the appropriate project files.

## Limitations

- Predictions are estimates, not guarantees of customer behaviour.
- Model performance depends on the dataset and the test split.
- Feature importance indicates how the model uses features; it does not prove causation.
- A suitable churn-risk threshold should be selected according to the intended business use.
- The dataset's representativeness and the model's performance on new customers require further validation.

## Disclaimer

This project is for educational and portfolio purposes. It is not a production-ready banking decision system.
