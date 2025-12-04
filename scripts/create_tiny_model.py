# script to create model for CI
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Load tokenizer and model without interactive prompts
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, ignore_mismatched_sizes=True)

# Save it locally
model.save_pretrained("local_model")
tokenizer.save_pretrained("local_model")
