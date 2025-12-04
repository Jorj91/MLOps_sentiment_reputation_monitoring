'''
# function to predict sentiment
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

#point to the latest fine-tuned model created with train.py
MODEL_PATH = "src/models/model_v1"

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

# function to predict sentiment
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import os

# Detect if we are running inside GitHub Actions
RUNNING_IN_CI = os.getenv("GITHUB_ACTIONS") == "true"

if RUNNING_IN_CI:
    MODEL_PATH = "local_model"           # tiny model created during CI
else:
    MODEL_PATH = "src/models/model_v1"   # your real fine-tuned model

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
