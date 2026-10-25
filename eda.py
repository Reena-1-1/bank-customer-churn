
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("data/churn.csv")

# 1. Check the dataset size
print("Dataset shape:", df.shape)

# 2. Count customers who stayed and left
print("\nChurn counts:")
print(df["Exited"].value_counts().sort_index())

# 3. Calculate churn percentages
print("\nChurn percentages:")
print((df["Exited"].value_counts(normalize=True).sort_index() * 100).round(2))

# 4. Display basic statistics
print("\nNumerical summary:")
print(df.describe())

# 5. Plot customer churn distribution
churn_counts = df["Exited"].value_counts().sort_index()

churn_counts.plot(kind="bar")
plt.title("Bank Customer Churn Distribution")
plt.xlabel("Exited (0 = Stayed, 1 = Left)")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
