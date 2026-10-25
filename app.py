import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
 
st.set_page_config(
    page_title="Bank Churn Intelligence",
    page_icon="🏦",
    layout="wide"
)
 
# Load the trained model
MODEL_PATH = Path("models/churn_model.pkl")
DATA_PATH = Path("models/scored_customers.csv")
 
st.title("🏦 Bank Customer Churn Intelligence")
st.write(
    "Predict customer churn risk and explore "
    "factors associated with customer retention."
)
 
if not MODEL_PATH.exists():
    st.error("Model not found. Run feature_importance.py first.")
    st.stop()
 
model = joblib.load(MODEL_PATH)
 
st.sidebar.header("Customer Details")
 
credit_score = st.sidebar.slider(
    "Credit Score", 300, 900, 650
)
geography = st.sidebar.selectbox(
    "Geography", ["France", "Germany", "Spain"]
)
gender = st.sidebar.selectbox(
    "Gender", ["Female", "Male"]
)
age = st.sidebar.slider("Age", 18, 100, 40)
tenure = st.sidebar.slider("Tenure (years)", 0, 10, 5)
balance = st.sidebar.number_input(
    "Account Balance", min_value=0.0, value=50000.0
)
products = st.sidebar.selectbox(
    "Number of Products", [1, 2, 3, 4]
)
has_card = st.sidebar.selectbox(
    "Has Credit Card?", [1, 0],
    format_func=lambda x: "Yes" if x == 1 else "No"
)
active = st.sidebar.selectbox(
    "Active Member?", [1, 0],
    format_func=lambda x: "Yes" if x == 1 else "No"
)
salary = st.sidebar.number_input(
    "Estimated Salary", min_value=0.0, value=60000.0
)
 
# Build the input in the same format used during training
input_data = pd.DataFrame([{
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": products,
    "HasCrCard": has_card,
    "IsActiveMember": active,
    "EstimatedSalary": salary,
    "BalanceSalaryRatio": balance / max(salary, 1),
    "ProductDensity": products / (tenure + 1),
    "EngagementProduct": active * products,
    "AgeTenureInteraction": age * tenure
}])
 
st.subheader("Customer Risk Assessment")
 
if st.button("Predict Churn Risk", type="primary"):
    probability = model.predict_proba(input_data)[0, 1]
    prediction = int(probability >= 0.5)
 
    col1, col2 = st.columns(2)
 
    with col1:
        st.metric("Predicted Churn Probability", f"{probability:.1%}")
 
    with col2:
        st.metric(
            "Prediction at 50% Threshold",
            "Likely to churn" if prediction else "Likely to stay"
        )
 
    st.progress(float(probability))
 
    if probability >= 0.5:
        st.warning(
            "The model flags this customer as higher risk. "
            "Consider reviewing appropriate retention options."
        )
    else:
        st.success(
            "The model does not flag this customer above "
            "the current 50% threshold."
        )
 
st.divider()
st.subheader("Model Feature Importance")
 
importance_path = Path("images/feature_importance.png")
 
if importance_path.exists():
    st.image(
        str(importance_path),
        caption="Random Forest feature importance"
    )
else:
    st.info("Run feature_importance.py to generate the chart.")
 
st.caption(
    "This is an experimental machine learning model. "
    "Predictions are estimates, not guarantees of customer behaviour."
)
 
# Churn Probability Distribution
st.divider()
st.subheader("📊 Churn Probability Distribution")
st.write(
    "Explore the model's predicted churn probabilities across the customer dataset."
)
 
if DATA_PATH.exists():
    try:
        scored = pd.read_csv(DATA_PATH)
 
        fig, ax = plt.subplots(figsize=(9, 5))
 
        for label, name in [(0, "Stayed"), (1, "Exited")]:
            values = scored.loc[scored["Exited"] == label, "Probability"]
            if not values.empty:
                ax.hist(values, bins=20, alpha=0.6, label=name)
 
        ax.axvline(0.5, linestyle="--", label="50% decision threshold")
        ax.set_xlabel("Predicted Churn Probability")
        ax.set_ylabel("Number of Customers")
        ax.set_title("Predicted Churn Risk Distribution")
        ax.legend()
        fig.tight_layout()
 
        st.pyplot(fig)
        plt.close(fig)
 
        st.caption(
            "Scores come from the trained model on the full dataset, including "
            "records used for training, so separation looks stronger than on "
            "unseen data. Actual status is shown for comparison only; this "
            "chart is not a guarantee of future churn."
        )
 
    except Exception as e:
        st.error(f"Could not create the probability chart: {e}")
else:
    st.warning("Scored data not found. Run make_scores.py first.")
    
# What-If Scenario Simulator
st.divider()
st.subheader("🔍 What-If Scenario Simulator")
 
st.write(
    "Change a customer's activity status or number of products "
    "to compare hypothetical churn-risk predictions."
)
 
col1, col2 = st.columns(2)
 
with col1:
    scenario_active = st.selectbox(
        "Scenario: Customer Active?",
        options=[0, 1],
        index=active,
        format_func=lambda x: "Yes" if x == 1 else "No",
        key="scenario_active"
    )
 
with col2:
    scenario_products = st.selectbox(
        "Scenario: Number of Products",
        options=[1, 2, 3, 4],
        index=products - 1,
        key="scenario_products"
    )
 
# Original customer's predicted risk
baseline_probability = model.predict_proba(input_data)[0, 1]
 
# Copy the original customer and modify scenario values
scenario_data = input_data.copy()
scenario_data["IsActiveMember"] = scenario_active
scenario_data["NumOfProducts"] = scenario_products
 
# Recalculate engineered features after the changes
scenario_data["BalanceSalaryRatio"] = (
    scenario_data["Balance"]
    / scenario_data["EstimatedSalary"].clip(lower=1)
)
scenario_data["ProductDensity"] = (
    scenario_data["NumOfProducts"]
    / (scenario_data["Tenure"] + 1)
)
scenario_data["EngagementProduct"] = (
    scenario_data["IsActiveMember"]
    * scenario_data["NumOfProducts"]
)
scenario_data["AgeTenureInteraction"] = (
    scenario_data["Age"] * scenario_data["Tenure"]
)
 
# Scenario prediction
scenario_probability = model.predict_proba(scenario_data)[0, 1]
difference = (scenario_probability - baseline_probability) * 100
 
st.markdown("### Risk Comparison")
 
col1, col2, col3 = st.columns(3)
 
with col1:
    st.metric(
        "Original Risk",
        f"{baseline_probability:.1%}"
    )
 
with col2:
    st.metric(
        "Scenario Risk",
        f"{scenario_probability:.1%}"
    )
 
with col3:
    st.metric(
        "Risk Difference",
        f"{difference:+.1f} percentage points"
    )
 
st.progress(float(scenario_probability))
 
if difference < -0.1:
    st.success("The model predicts lower churn risk in this scenario.")
elif difference > 0.1:
    st.warning("The model predicts higher churn risk in this scenario.")
else:
    st.info("The model predicts almost no change in churn risk.")
 
st.caption(
    "This is a hypothetical model comparison, not proof of causation. "
    "Only activity status and product count are changed; all other "
    "customer details remain the same. Predictions are estimates."
)
 
# SHAP Explainability
st.divider()
st.subheader("🧠 Explainable AI — SHAP Analysis")
 
st.write(
    "This chart shows which features most influence the "
    "Random Forest model's churn predictions across the sampled customers."
)
 
shap_chart_path = Path("images/shap_summary.png")
 
if shap_chart_path.exists():
    st.image(
        str(shap_chart_path),
        caption="SHAP Summary Plot — Global Feature Contributions",
        use_container_width=True
    )
else:
    st.info(
        "SHAP chart not found. Run shap_explainability.py first."
    )
 
st.caption(
    "SHAP describes how the model uses input features. "
    "These associations do not prove cause and effect."
)
 
# Confusion Matrix
st.divider()
st.subheader("📉 Random Forest Confusion Matrix")
 
confusion_path = Path("images/confusion_matrix.png")
 
if confusion_path.exists():
    st.image(
        str(confusion_path),
        caption="Random Forest performance on the test dataset",
        use_container_width=True
    )
else:
    st.warning("Run compare_models.py to generate the confusion matrix.")
 
# Model Comparison and Partial Dependence
st.divider()
st.subheader("📈 Model Comparison")
 
comparison_path = Path("images/model_comparison.png")
 
if comparison_path.exists():
    st.image(
        str(comparison_path),
        caption="Model comparison on the test set",
        use_container_width=True
    )
else:
    st.info("Run compare_models.py to generate the comparison chart.")
 
st.subheader("📉 Partial Dependence")
 
pdp_path = Path("images/partial_dependence.png")
 
if pdp_path.exists():
    st.image(
        str(pdp_path),
        caption="Average effect of key features on predicted churn probability",
        use_container_width=True
    )
else:
    st.info("Run partial_dependence.py to generate the chart.")
 
st.caption(
    "Partial dependence shows average model behaviour, not cause and effect."
)

