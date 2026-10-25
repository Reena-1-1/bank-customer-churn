import joblib
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.inspection import PartialDependenceDisplay

df = pd.read_csv("data/churn.csv")
df = df.drop(columns=["CustomerId", "Surname", "Year"], errors="ignore")

# Same engineered features used in training
df["BalanceSalaryRatio"] = df["Balance"] / df["EstimatedSalary"].clip(lower=1)
df["ProductDensity"] = df["NumOfProducts"] / (df["Tenure"] + 1)
df["EngagementProduct"] = df["IsActiveMember"] * df["NumOfProducts"]
df["AgeTenureInteraction"] = df["Age"] * df["Tenure"]

X = df.drop(columns=["Exited"])
X_sample = X.sample(n=min(2000, len(X)), random_state=42).copy()

# Partial dependence needs floating-point numeric columns
numeric_cols = X_sample.select_dtypes(include="number").columns
X_sample[numeric_cols] = X_sample[numeric_cols].astype(float)

model = joblib.load("models/churn_model.pkl")

features = ["Age", "NumOfProducts", "Balance", "CreditScore"]

Path("images").mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(11, 8))
PartialDependenceDisplay.from_estimator(
    model,
    X_sample,
    features,
    kind="average",
    grid_resolution=30,
    n_jobs=1,
    ax=ax,
)
fig.suptitle("Partial Dependence of Predicted Churn Probability (Random Forest)")
fig.tight_layout()
fig.savefig("images/partial_dependence.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved: images/partial_dependence.png")