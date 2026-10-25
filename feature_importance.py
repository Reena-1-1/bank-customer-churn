
import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, roc_auc_score

# 1. Load the dataset
df = pd.read_csv("data/churn.csv")

# 2. Remove identifiers and the unverified Year column
df = df.drop(columns=["CustomerId", "Surname", "Year"])

# 3. Create engineered features
df["BalanceSalaryRatio"] = (
    df["Balance"] / df["EstimatedSalary"].replace(0, 1)
)
df["ProductDensity"] = df["NumOfProducts"] / (df["Tenure"] + 1)
df["EngagementProduct"] = df["IsActiveMember"] * df["NumOfProducts"]
df["AgeTenureInteraction"] = df["Age"] * df["Tenure"]

# 4. Separate features and target
X = df.drop(columns=["Exited"])
y = df["Exited"]

categorical_columns = ["Geography", "Gender"]
numerical_columns = [
    col for col in X.columns
    if col not in categorical_columns
]

# 5. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 6. Create the preprocessing pipeline
preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), numerical_columns),
    ("categorical",
     OneHotEncoder(handle_unknown="ignore"),
     categorical_columns)
])

# 7. Create and train Random Forest
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)

# 8. Evaluate on the held-out test set
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("Recall:", round(recall_score(y_test, predictions), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))

# 9. Extract feature importance
feature_names = (
    model.named_steps["preprocessing"].get_feature_names_out()
)
importance_values = model.named_steps["classifier"].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
}).sort_values("Importance", ascending=False)

print("\nTop 10 features associated with model predictions:")
print(importance_df.head(10).to_string(index=False))

# 10. Save the feature importance chart
os.makedirs("images", exist_ok=True)

top_features = importance_df.head(10).sort_values("Importance")

plt.figure(figsize=(10, 6))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("images/feature_importance.png", dpi=150)
plt.show()

# 11. Save the trained model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/churn_model.pkl")

print("\nModel saved to models/churn_model.pkl")
print("Chart saved to images/feature_importance.png")
