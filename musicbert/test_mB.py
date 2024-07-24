import torch
from data import MidiDataset
from model import MusicBertFrontOri
from torch.utils.data import DataLoader

# Initialize dataset and dataloader
dataset = MidiDataset(data_dir="/home/brad/models/splitfinetuning/data/clean_midi", resolution=100, fixed_length=10000)  # Smaller fixed length
dataloader = DataLoader(dataset, batch_size=8, collate_fn=dataset.collate)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MusicBertFrontOri(pretrain_dir="/home/brad/models/splitfinetuning/pretrain/bert-base-uncased", split_num=6, rank=1, device=device)  # Example arguments

# Iterate through dataloader
for batch in dataloader:
    input_ids = batch["batch_input_ids"]
    token_type_ids = batch["batch_token_type_ids"]
    attention_mask = batch["batch_attention_mask"]
    labels = batch["batch_labels"]
    
    # Print batch details
    print(f"Batch input IDs shape: {input_ids.shape}")
    print(f"Batch token type IDs shape: {token_type_ids.shape}")
    print(f"Batch attention mask shape: {attention_mask.shape}")
    print(f"Batch labels shape: {labels.shape}")
    
    # Forward pass
    try:
        outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        print(f"Model outputs shape: {outputs.shape}")
    except Exception as e:
        print(f"Error during forward pass: {e}")
