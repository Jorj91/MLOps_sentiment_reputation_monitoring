'''
Docstring per src.sentiment
# src/sentiment.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import os

# Load model path from environment variable (default to full model)
MODEL_PATH = os.getenv("MODEL_PATH", "src/models/model_v1")

print(f"Loading model from {MODEL_PATH}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

LABELS = ["negative", "neutral", "positive"]

def predict_sentiment(text: str) -> dict:
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        scores = outputs.logits[0].numpy()
        probs = np.exp(scores) / np.exp(scores).sum()

    label = LABELS[np.argmax(probs)]

    return {
        "text": text,
        "sentiment": label,
        "probabilities": {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    }

'''


import os

USE_FAKE_MODEL = os.getenv("USE_FAKE_MODEL", "false").lower() == "true"


# FAKE MODEL (CI / docker test)

if USE_FAKE_MODEL:

    LABELS = ["negative", "neutral", "positive"]

    def predict_sentiment(text: str) -> dict:
        return {
            "text": text,
            "sentiment": "positive",
            "probabilities": {
                "negative": 0.05,
                "neutral": 0.05,
                "positive": 0.90
            }
        }
    

# REAL MODEL (training & deploy)
  
else:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    import numpy as np

    # Load model path from environment variable (default to full model)
    MODEL_PATH = os.getenv("MODEL_PATH", "src/models/model_v1")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    LABELS = ["negative", "neutral", "positive"]

    def predict_sentiment(text: str) -> dict:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            scores = outputs.logits[0].numpy()
            probs = np.exp(scores) / np.exp(scores).sum()

        label = LABELS[np.argmax(probs)]

        return {
            "text": text,
            "sentiment": label,
            "probabilities": {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
        }