import joblib
import pandas as pd

df = pd.read_csv("data/churn.csv")
model = joblib.load("models/churn_model.pkl")

X = df.drop(columns=["CustomerId", "Surname", "Year", "Exited"], errors="ignore").copy()
X["BalanceSalaryRatio"] = X["Balance"] / X["EstimatedSalary"].clip(lower=1)
X["ProductDensity"] = X["NumOfProducts"] / (X["Tenure"] + 1)
X["EngagementProduct"] = X["IsActiveMember"] * X["NumOfProducts"]
X["AgeTenureInteraction"] = X["Age"] * X["Tenure"]

out = pd.DataFrame({
    "Probability": model.predict_proba(X)[:, 1],
    "Exited": df["Exited"],
})
out.to_csv("models/scored_customers.csv", index=False)
print(out.shape)