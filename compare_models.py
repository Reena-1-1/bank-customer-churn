import pandas as pd
import matplotlib

matplotlib.use("Agg")  # save charts without opening windows
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# 1. Load and prepare the data
df = pd.read_csv("data/churn.csv")

df = df.drop(columns=["CustomerId", "Surname", "Year"], errors="ignore")

# 2. Feature engineering
df["BalanceSalaryRatio"] = df["Balance"] / df["EstimatedSalary"].replace(0, 1)
df["ProductDensity"] = df["NumOfProducts"] / (df["Tenure"] + 1)
df["EngagementProduct"] = df["IsActiveMember"] * df["NumOfProducts"]
df["AgeTenureInteraction"] = df["Age"] * df["Tenure"]

# 3. Separate features and target
X = df.drop(columns=["Exited"])
y = df["Exited"]

categorical_columns = ["Geography", "Gender"]
numerical_columns = [c for c in X.columns if c not in categorical_columns]

# 4. Stratified split (same split as before)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)


# 5. Preprocessing
def make_preprocessor():
    return ColumnTransformer([
        ("numeric", StandardScaler(), numerical_columns),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
    ])


# 6. Models (Logistic Regression and Random Forest settings unchanged)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

# Optional: XGBoost (only used if installed: pip install xgboost)
try:
    from xgboost import XGBClassifier

    models["XGBoost"] = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
except ImportError:
    print("XGBoost not installed - skipping the optional XGBoost model.")


# 7. Train and evaluate
results = []
confusion_matrices = {}
Path("images").mkdir(exist_ok=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, classifier in models.items():
    print(f"\nTraining {name}...")

    model = Pipeline([
        ("preprocessing", make_preprocessor()),
        ("classifier", classifier),
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # 5-fold cross-validated ROC-AUC on the training portion only
    cv_auc = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="roc_auc"
    )

    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, predictions), 4),
        "Precision": round(precision_score(y_test, predictions, zero_division=0), 4),
        "Recall": round(recall_score(y_test, predictions, zero_division=0), 4),
        "F1-score": round(f1_score(y_test, predictions, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, probabilities), 4),
        "CV ROC-AUC (5-fold mean)": round(cv_auc.mean(), 4),
        "CV ROC-AUC std": round(cv_auc.std(), 4),
    })

    confusion_matrices[name] = confusion_matrix(y_test, predictions, labels=[0, 1])


# 8. Confusion matrix for Random Forest (image used by the dashboard)
cm = confusion_matrices["Random Forest"]
print("\n--- Random Forest Confusion Matrix ---")
print("Rows = Actual; Columns = Predicted")
print("Class 0 = Stayed; Class 1 = Exited")
print(cm)

fig, ax = plt.subplots(figsize=(7, 5))
ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["Stayed", "Exited"]
).plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("Random Forest - Confusion Matrix")
fig.tight_layout()
fig.savefig("images/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 9. Model comparison table + bar chart
comparison = pd.DataFrame(results)

print("\n--- Model Comparison ---")
print(comparison.to_string(index=False))

comparison.to_csv("images/model_comparison.csv", index=False)

metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
ax = comparison.set_index("Model")[metrics].plot(
    kind="bar", figsize=(10, 5.5), rot=20
)
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Model Comparison on the Test Set")
ax.legend(loc="lower right", ncol=2)
plt.tight_layout()
plt.savefig("images/model_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved: images/confusion_matrix.png")
print("Saved: images/model_comparison.png")
print("Saved: images/model_comparison.csv")