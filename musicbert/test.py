import torch
from model import MusicBertFrontDecomposition

# Example dummy MIDI data
dummy_midi_data = torch.randn(1, 512, 88)  # Assume batch size of 1, sequence length of 512
dummy_attention_mask = torch.ones(1, 512)

# Initialize model
model = MusicBertFrontDecomposition(pretrain_dir='/home/brad/models/splitfinetuning/pretrain/bert-base-uncased/', split_num=12, rank=0, device='cpu')

# Forward pass
output = model(input_ids=dummy_midi_data, attention_mask=dummy_attention_mask)
print(output.shape)  # Check output shape