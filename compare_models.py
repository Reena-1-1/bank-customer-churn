
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# 1. Load and prepare the data
df = pd.read_csv("data/churn.csv")

df = df.drop(
    columns=["CustomerId", "Surname", "Year"],
    errors="ignore"
)

# 2. Feature engineering
df["BalanceSalaryRatio"] = (
    df["Balance"] / df["EstimatedSalary"].replace(0, 1)
)

df["ProductDensity"] = (
    df["NumOfProducts"] / (df["Tenure"] + 1)
)

df["EngagementProduct"] = (
    df["IsActiveMember"] * df["NumOfProducts"]
)

df["AgeTenureInteraction"] = (
    df["Age"] * df["Tenure"]
)

# 3. Separate features and target
X = df.drop(columns=["Exited"])
y = df["Exited"]

categorical_columns = ["Geography", "Gender"]

numerical_columns = [
    col for col in X.columns
    if col not in categorical_columns
]

# 4. Split the data once for a fair model comparison
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 5. Create preprocessing function
def make_preprocessor():
    return ColumnTransformer([
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
    ])


# 6. Define models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}


# 7. Train and evaluate models
results = []

# Create images folder if it does not exist
Path("images").mkdir(exist_ok=True)

for name, classifier in models.items():

    print(f"\nTraining {name}...")

    model = Pipeline([
        ("preprocessing", make_preprocessor()),
        ("classifier", classifier)
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Predictions on unseen test data
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    results.append({
        "Model": name,
        "Accuracy": round(
            accuracy_score(y_test, predictions), 4
        ),
        "Precision": round(
            precision_score(
                y_test, predictions, zero_division=0
            ), 4
        ),
        "Recall": round(
            recall_score(
                y_test, predictions, zero_division=0
            ), 4
        ),
        "F1-score": round(
            f1_score(
                y_test, predictions, zero_division=0
            ), 4
        ),
        "ROC-AUC": round(
            roc_auc_score(y_test, probabilities), 4
        )
    })

    # 8. Create confusion matrix for Random Forest only
    if name == "Random Forest":

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        )

        print("\n--- Random Forest Confusion Matrix ---")
        print("Rows = Actual; Columns = Predicted")
        print("Class 0 = Stayed; Class 1 = Exited")
        print(cm)

        cm_display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Stayed", "Exited"]
        )

        fig, ax = plt.subplots(figsize=(7, 5))

        cm_display.plot(
            ax=ax,
            cmap="Blues",
            values_format="d"
        )

        ax.set_title(
            "Random Forest - Confusion Matrix"
        )

        fig.tight_layout()

        fig.savefig(
            "images/confusion_matrix.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()
        plt.close(fig)


# 9. Display model comparison
comparison = pd.DataFrame(results)

print("\n--- Model Comparison ---")
print(comparison.to_string(index=False))

print(
    "\nConfusion matrix saved to "
    "images/confusion_matrix.png"
)
