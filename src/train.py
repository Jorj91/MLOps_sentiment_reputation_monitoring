
'''
import os
import json

def train_model():
    """
    Placeholder training function.
    In a real project, this would:
      - Load dataset
      - Fine-tune the model
      - Save updated weights
    """

    # Step 1: Simulate dataset loading
    print("Step 1: Loading dataset...")
    # TODO: replace with actual dataset loading logic

    # Step 2: Simulate model training
    print("Step 2: Fine-tuning model...")
    # TODO: replace with actual training logic

    # Step 3: Save model (mock)
    model_dir = "src/models/model_v1"
    os.makedirs(model_dir, exist_ok=True)
    # Save a metadata file to simulate training output
    metadata = {"status": "trained", "version": "v1"}
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    print(f"Step 3: Model saved to {model_dir}")
    print("Training completed successfully.")

    return {"status": "ok", "message": "Training pipeline executed (mock)."}


# Allow running as script
if __name__ == "__main__":
    train_model()
'''


import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from datasets import load_dataset


MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MODEL_DIR = "src/models/model_v1"
LABELS = ["negative", "neutral", "positive"]

def train_model():
    print("Step 1: Loading dataset...")
    dataset = load_dataset("tweet_eval", "sentiment")

    # Take small subset for demo
    train_data = dataset["train"].select(range(100))
    val_data = dataset["validation"].select(range(50))

    print("Step 2: Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, ignore_mismatched_sizes=True)

    print("Step 3: Tokenizing dataset...")
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)
        
    train_data = train_data.map(tokenize, batched=True)
    val_data = val_data.map(tokenize, batched=True)

    train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Step 4: Fine-tuning model (1 epoch for demo)...")
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print("Step 4: Fine-tuning completed.")


    # Step 5: Save model and metadata
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    metadata = {"status": "trained", "version": "v1"}
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    
    print(f"Step 5: Model saved to {MODEL_DIR}")
    print("Training pipeline executed successfully.")
    return {"status": "ok"}

if __name__ == "__main__":
    train_model()