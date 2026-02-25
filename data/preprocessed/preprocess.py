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

# TEXT PREPROCESSING FUNCTION
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
negation_words = {"not", "nor", "never", "no"}
stop_words = stop_words - negation_words

lemmatizer = WordNetLemmatizer()

def custom_preprocessor(text):
  text = text.lower()
  text = re.sub(r"http\S+|www\S+|https\S+", "", text)
  text = re.sub(r"\d+", "", text)
  text = text.translate(str.maketrans("", "", string.punctuation))
  words = word_tokenize(text)
  words = [
      lemmatizer.lemmatize(word)
      for word in words
      if word not in stop_words and word.isalpha()
  ]
  return " ".join(words)

print(custom_preprocessor(df["content"].iloc[0]))


