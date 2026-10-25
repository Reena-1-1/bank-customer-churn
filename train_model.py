
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# 1. Load the dataset
df = pd.read_csv("data/churn.csv")

# 2. Remove identifiers and the unverified Year column
df = df.drop(columns=["CustomerId", "Surname", "Year"])

# 3. Create additional features
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

# 4. Separate features and target
X = df.drop(columns=["Exited"])
y = df["Exited"]

# 5. Identify categorical and numerical columns
categorical_columns = ["Geography", "Gender"]

numerical_columns = [
    col for col in X.columns
    if col not in categorical_columns
]

# 6. Prepare data transformations
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numerical_columns
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ]
)

# 7. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training customers:", len(X_train))
print("Testing customers:", len(X_test))

# 8. Build the Logistic Regression pipeline
model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

# 9. Train the model
model.fit(X_train, y_train)

# 10. Make predictions on unseen test data
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 11. Evaluate model performance
print("\n--- Logistic Regression Results ---")

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 4))
print("Recall:", round(recall_score(y_test, y_pred, zero_division=0), 4))
print("F1-score:", round(f1_score(y_test, y_pred, zero_division=0), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
