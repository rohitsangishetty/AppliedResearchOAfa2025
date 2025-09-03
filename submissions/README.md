# Fake News Detection with DistilBERT
# Rohit Sangishetty, Georgia Tech AI@GT

## What it Does
This project classifies news headlines as **FAKE** or **REAL**.  
- Uses Hugging Face DistilBERT embeddings + Scikit-learn Logistic Regression.  
- Trains on the [GonzaloA/fake_news](https://huggingface.co/datasets/GonzaloA/fake_news) dataset (sampled for speed).  
- Outputs evaluation metrics, a confusion matrix, and predictions on sample/test headlines.  
- Saves the trained model with **joblib** so it can be reloaded later without retraining.

**Dependencies:** `torch`, `transformers`, `datasets`, `pandas`, `numpy`, `scikit-learn`, `joblib`

---

## Example Output
Embeddings: (1500, 768) (500, 768)

Classification Report:
              precision    recall  f1-score   support
FAKE           0.84       0.81      0.82      245
REAL           0.87       0.89      0.88      255

Confusion matrix:
[[198  47]
 [ 28 227]]

Sample Prediction:
"Breaking: Scientists confirm water on Mars for the first time!" -> REAL (Confidence: 0.92)

Random Headlines:
"President proposes new education reforms nationwide" => REAL (0.89)
"Aliens discovered living among us, claims scientist" => FAKE (0.78)

## The file fake_news_clf.joblib contains the trained Logistic Regression classifier and the Hugging Face model reference.
To reload: 
import joblib
model_artifacts = joblib.load("fake_news_clf.joblib")
clf = model_artifacts["model"]

## How to Run
```bash
python fake_news.py


