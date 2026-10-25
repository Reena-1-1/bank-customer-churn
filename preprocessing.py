
import pandas as pd

# 1. Load the dataset
df = pd.read_csv("data/churn.csv")

print("Original dataset shape:", df.shape)

# 2. Remove columns that identify customers
df = df.drop(columns=["CustomerId", "Surname"])

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

# 4. Encode categorical variables
df = pd.get_dummies(
    df,
    columns=["Geography", "Gender"],
    drop_first=True,
    dtype=int
)

# 5. Separate input features and target
X = df.drop(columns=["Exited"])
y = df["Exited"]

print("\nProcessed dataset shape:", df.shape)
print("Number of input features:", X.shape[1])
print("Target shape:", y.shape)

print("\nInput feature columns:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())
