
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split

# 1. Load the dataset
df = pd.read_csv("data/churn.csv")

df = df.drop(
    columns=["CustomerId", "Surname", "Year"],
    errors="ignore"
)

# 2. Re-create the engineered features
df["BalanceSalaryRatio"] = (
    df["Balance"] / df["EstimatedSalary"].replace(0, 1)
)

df["ProductDensity"] = (
    df["NumOfProducts"] / (df["Tenure"] + 1)
)

df["EngagementProduct"] = (
    df["IsActiveMember"] * df["NumOfProducts"]
)

df["AgeTenureInteraction"] = df["Age"] * df["Tenure"]

X = df.drop(columns=["Exited"])
y = df["Exited"]

# 3. Use the same stratified split as model training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Load the existing trained Random Forest pipeline
model_path = Path("models/churn_model.pkl")

if not model_path.exists():
    raise FileNotFoundError(
        "Model not found. Run feature_importance.py first."
    )

model = joblib.load(model_path)

# 5. Get the fitted preprocessing pipeline
preprocessor = model.named_steps["preprocessing"]
classifier = model.named_steps["classifier"]

# Transform test data using the fitted preprocessing
X_test_transformed = preprocessor.transform(X_test)

# Convert sparse output to a dense array if necessary
if hasattr(X_test_transformed, "toarray"):
    X_test_transformed = X_test_transformed.toarray()

# 6. Get transformed feature names
feature_names = preprocessor.get_feature_names_out()

X_test_transformed = pd.DataFrame(
    X_test_transformed,
    columns=feature_names,
    index=X_test.index
)

# 7. Sample up to 200 customers to keep SHAP manageable
sample = X_test_transformed.sample(
    n=min(200, len(X_test_transformed)),
    random_state=42
)

# 8. Explain the Random Forest predictions
explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(sample)

# Support common SHAP output formats for binary classification
if isinstance(shap_values, list):
    values = shap_values[1]
elif getattr(shap_values, "ndim", 0) == 3:
    values = shap_values[:, :, 1]
else:
    values = shap_values

# 9. Save the global feature importance plot
Path("images").mkdir(exist_ok=True)

plt.figure()
shap.summary_plot(
    values,
    sample,
    show=False,
    max_display=15
)
plt.tight_layout()
plt.savefig(
    "images/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("SHAP explanation completed successfully.")
print("Saved chart: images/shap_summary.png")
