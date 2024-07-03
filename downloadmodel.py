from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Define the directory
pretrain_dir = "./pretrain/bert-base-uncased"

# Ensure the directory exists
os.makedirs(pretrain_dir, exist_ok=True)

# Load and save the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer.save_pretrained(pretrain_dir)

# Load and save the model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
model.save_pretrained(pretrain_dir)

print(f"Model and tokenizer saved to {pretrain_dir}")
