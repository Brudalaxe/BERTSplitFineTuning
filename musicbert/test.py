import torch
from torch.utils.data import DataLoader
from data import MidiDataset  # Make sure to import your MidiDataset class
from model import MusicBertFrontDecomposition, MusicBertBackDecomposition, MusicBertFrontOri, MusicBertBackOri

def get_real_data_point():
    # Create the dataset
    dataset = MidiDataset(data_dir='/home/brad/models/splitfinetuning/data/clean_midi/')  # Adjust this to point to your MIDI files directory
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Get a single data point
    data_point = next(iter(data_loader))
    
    # Extracting the data
    real_midi_data = data_point['input_ids']
    real_attention_mask = data_point['attention_mask']
    dummy_label = torch.tensor([0])  # Assign a dummy label
    
    return real_midi_data, real_attention_mask, dummy_label

def test_decomposition_model():
    print("Testing Decomposition Model")
    # Load a real data point
    real_midi_data, real_attention_mask, _ = get_real_data_point()

    # Reshape input_ids to match the expected shape (batch_size, sequence_length, 88)
    sequence_length = real_midi_data.shape[-1] // 88
    real_midi_data = real_midi_data.view(1, sequence_length, 88)
    
    # Ensure the sequence length is 512 tokens or less
    if sequence_length > 512:
        real_midi_data = real_midi_data[:, :512, :]
        real_attention_mask = real_attention_mask[:, :512]
    
    # Initialize front and back models
    front_model = MusicBertFrontDecomposition(pretrain_dir='/home/brad/models/splitfinetuning/pretrain/bert-base-uncased/', split_num=12, rank=64, device='cpu')
    back_model = MusicBertBackDecomposition(pretrain_dir='/home/brad/models/splitfinetuning/pretrain/bert-base-uncased/', split_num=12, rank=64, label_nums=10, device='cpu')

    # Forward pass through front model
    front_output = front_model(input_ids=real_midi_data, attention_mask=real_attention_mask)
    print(f"Front model output shape: {front_output.shape}")

    # Forward pass through back model
    back_output = back_model(hidden_states=front_output, attention_mask=real_attention_mask)
    print(f"Back model output shape: {back_output.shape}")

def test_ori_model():
    print("Testing Ori Model")
    # Load a real data point
    real_midi_data, real_attention_mask, _ = get_real_data_point()

    # Reshape input_ids to match the expected shape (batch_size, sequence_length, 88)
    sequence_length = real_midi_data.shape[-1] // 88
    real_midi_data = real_midi_data.view(1, sequence_length, 88)

    # Ensure the sequence length is 512 tokens or less
    if sequence_length > 512:
        real_midi_data = real_midi_data[:, :512, :]
        real_attention_mask = real_attention_mask[:, :512]

    # Initialize front and back models
    front_model = MusicBertFrontOri(pretrain_dir='/home/brad/models/splitfinetuning/pretrain/bert-base-uncased/', split_num=12, rank=0, device='cpu')
    back_model = MusicBertBackOri(pretrain_dir='/home/brad/models/splitfinetuning/pretrain/bert-base-uncased/', split_num=12, rank=0, label_nums=10, device='cpu')

    # Forward pass through front model
    front_output = front_model(input_ids=real_midi_data, attention_mask=real_attention_mask)
    print(f"Front model output shape: {front_output.shape}")

    # Forward pass through back model
    back_output = back_model(hidden_states=front_output, attention_mask=real_attention_mask)
    print(f"Back model output shape: {back_output.shape}")

if __name__ == "__main__":
    test_decomposition_model()
    test_ori_model()
