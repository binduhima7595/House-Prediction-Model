import pandas as pd
import os

df = pd.read_csv("data/raw/housing.csv")

df = df.dropna()

# ⭐ drop text column
if "Address" in df.columns:
    df = df.drop("Address", axis=1)

os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/train.csv", index=False)

print("preprocess done")