import os
import random
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
MODEL_NAME = "distilbert-base-uncased"
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Load dataset
dataset = load_dataset("GonzaloA/fake_news")

# Label mapping
label_groups = {0: "FAKE", 1: "REAL"}

print(f"Using device: {DEVICE}")

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

# Convert HuggingFace dataset split to pandas dataframe
def hf_to_df(dsplit):
    texts = dsplit["text"]
    labels = dsplit["label"]
    return pd.DataFrame({"text": texts, "label": labels})

train_df = pd.concat([hf_to_df(dataset["train"]), hf_to_df(dataset["validation"])], ignore_index=True)
test_df = hf_to_df(dataset["test"])

train_df["label_bin"] = train_df["label"].apply(lambda s: 0 if s == 0 else 1)
test_df["label_bin"] = test_df["label"].apply(lambda s: 0 if s == 0 else 1)

MAX_TRAIN = 1500
MAX_TEST = 500

if len(train_df) > MAX_TRAIN:
    train_df = train_df.sample(MAX_TRAIN, random_state=RANDOM_SEED).reset_index(drop=True)
if len(test_df) > MAX_TEST:
    test_df = test_df.sample(MAX_TEST, random_state=RANDOM_SEED).reset_index(drop=True)

X_train_texts = train_df["text"].tolist()
y_train = train_df["label_bin"].values
X_test_texts = test_df["text"].tolist()
y_test = test_df["label_bin"].values

# Mean pooling function
def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts

# Text embedding function
def embed_texts(texts, batch_size=32):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model(**enc)
            embs = mean_pool(outputs.last_hidden_state, enc["attention_mask"])
        all_embeddings.append(embs.cpu().numpy())
    return np.vstack(all_embeddings)

X_train = embed_texts(X_train_texts, batch_size=32)
X_test = embed_texts(X_test_texts, batch_size=32)
print("Embeddings:", X_train.shape, X_test.shape)

# Train Logistic Regression classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# Prediction function
def predict_headline(text):
    emb = embed_texts([text])
    pred = clf.predict(emb)[0]
    proba = clf.predict_proba(emb)[0]
    return ("FAKE" if pred == 0 else "REAL", float(np.max(proba)))

# Original sample headline
sample = "Breaking: Scientists confirm water on Mars for the first time!"
print("Sample:", sample, "->", predict_headline(sample))

# sample 5 random headlines from the test dataset
num_samples = 5
random_headlines = test_df["text"].sample(num_samples, random_state=RANDOM_SEED).tolist()

print("\nRandom headlines from the dataset:")
for headline in random_headlines:
    prediction, probability = predict_headline(headline)
    print(f"{headline} => {prediction} (Confidence: {probability:.2f})")

# Optional: save artifacts
import joblib
joblib.dump({"model": clf, "hf_model_name": MODEL_NAME}, "fake_news_clf.joblib")
