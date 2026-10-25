
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/churn.csv")

# Calculate churn rate by customer characteristic
print("Churn rate by Geography (%):")
print((df.groupby("Geography")["Exited"].mean() * 100).round(2))

print("\nChurn rate by Gender (%):")
print((df.groupby("Gender")["Exited"].mean() * 100).round(2))

print("\nChurn rate by Active Membership (%):")
print((df.groupby("IsActiveMember")["Exited"].mean() * 100).round(2))

print("\nChurn rate by Number of Products (%):")
print((df.groupby("NumOfProducts")["Exited"].mean() * 100).round(2))

# Compare average age of retained and churned customers
print("\nAverage age by churn status:")
print(df.groupby("Exited")["Age"].mean().round(2))

# Visualize churn rate by geography
(df.groupby("Geography")["Exited"].mean() * 100).plot(
    kind="bar"
)

plt.title("Churn Rate by Geography")
plt.xlabel("Geography")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
