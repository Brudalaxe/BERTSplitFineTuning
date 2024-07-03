import torch
from model import MusicBertFrontDecomposition, MusicBertBackDecomposition, MusicBertFrontOri, MusicBertBackOri

def test_decomposition_model():
    print("Testing Decomposition Model")
    # Example dummy MIDI data
    dummy_midi_data = torch.randn(1, 512, 88)  # Assume batch size of 1, sequence length of 512
    dummy_attention_mask = torch.ones(1, 512)

    # Initialize front and back models
    front_model = MusicBertFrontDecomposition(pretrain_dir='/home/brad/models/splitfinetuning/pretrain/bert-base-uncased/', split_num=12, rank=64, device='cpu')
    back_model = MusicBertBackDecomposition(pretrain_dir='/home/brad/models/splitfinetuning/pretrain/bert-base-uncased/', split_num=12, rank=64, label_nums=10, device='cpu')

    # Forward pass through front model
    front_output = front_model(input_ids=dummy_midi_data, attention_mask=dummy_attention_mask)
    print(f"Front model output shape: {front_output.shape}")

    # Forward pass through back model
    back_output = back_model(hidden_states=front_output, attention_mask=dummy_attention_mask)
    print(f"Back model output shape: {back_output.shape}")

def test_ori_model():
    print("Testing Ori Model")
    # Example dummy MIDI data
    dummy_midi_data = torch.randn(1, 512, 88)  # Assume batch size of 1, sequence length of 512
    dummy_attention_mask = torch.ones(1, 512)

    # Initialize front and back models
    front_model = MusicBertFrontOri(pretrain_dir='/home/brad/models/splitfinetuning/pretrain/bert-base-uncased/', split_num=12, rank=64, device='cpu')
    back_model = MusicBertBackOri(pretrain_dir='/home/brad/models/splitfinetuning/pretrain/bert-base-uncased/', split_num=12, rank=64, label_nums=10, device='cpu')

    # Forward pass through front model
    front_output = front_model(input_ids=dummy_midi_data, attention_mask=dummy_attention_mask)
    print(f"Front model output shape: {front_output.shape}")

    # Forward pass through back model
    back_output = back_model(hidden_states=front_output, attention_mask=dummy_attention_mask)
    print(f"Back model output shape: {back_output.shape}")

if __name__ == "__main__":
    test_decomposition_model()
    test_ori_model()
