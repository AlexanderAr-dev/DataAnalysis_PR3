import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

params = yaml.safe_load(open("params.yaml"))["data"]

df = pd.read_csv(params["csv_path"])
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.to_csv("data/train_split.csv", index=False)
val_df.to_csv("data/val_split.csv", index=False)
print("✅ Train/Val split completed.")
