# DATA PREPROCESSING
import pandas as pd
import re
import string
fake = pd.read_csv("Fake.csv", engine="python", encoding="latin1")
real = pd.read_csv("True.csv", engine="python", encoding="latin1")
fake['label'] = 0
real['label'] = 1
df = pd.concat([fake, real], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df["content"] = df["title"] + " " + df["text"]
print("Initial dataset shape:", df.shape)
print("Null values per column:")
print(df.isnull().sum())
duplicate_count = df.duplicated().sum()
print("Number of duplicate rows:", duplicate_count)
print("Dataset shape after cleaning:", df.shape)
df = df[["content", "label"]]
print("Final dataset shape:", df.shape)
print("Class distribution:")
print(df["label"].value_counts())


