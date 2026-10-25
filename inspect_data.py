
print("Starting Bank Customer Churn project...")

import pandas as pd

df = pd.read_csv("data/churn.csv")

print("\nFirst five customers:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nScript completed successfully!")
