# -*- coding: UTF-8 -*-
"""
-----------------------------------
@Author : Your Name
@Date : 2024/6/25
-----------------------------------
"""
from tqdm import tqdm
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pretty_midi
import numpy as np

from config import args, logger

def adjust_piano_roll(piano_roll):
    total_length = piano_roll.shape[1]  # Time dimension
    remainder = total_length % 88
    if remainder != 0:
        padding_length = 88 - remainder
        # Ensure the padding matches the pitch dimension
        padding = torch.zeros((piano_roll.shape[0], padding_length), dtype=piano_roll.dtype)
        piano_roll = torch.cat((piano_roll, padding), dim=1)
    return piano_roll.view(-1, 88)  # Reshape to [time_steps/88, 88]

# Adjust the sequence length to be divisible by 88
def adjust_sequence_length(tensor, factor=88):
    if tensor.dim() == 2:
        remainder = tensor.size(1) % factor
        if remainder != 0:
            padding_length = factor - remainder
            padding = torch.zeros((tensor.size(0), padding_length), dtype=tensor.dtype)
            tensor = torch.cat((tensor, padding), dim=1)
    elif tensor.dim() == 1:
        remainder = tensor.size(0) % factor
        if remainder != 0:
            padding_length = factor - remainder
            padding = torch.zeros((padding_length,), dtype=tensor.dtype)
            tensor = torch.cat((tensor, padding), dim=0)
    return tensor

# Crop the piano roll to a fixed length
def crop_piano_roll(piano_roll, fixed_length=1000):
    if piano_roll.shape[1] > fixed_length:
        piano_roll = piano_roll[:, :fixed_length]
    else:
        padding_length = fixed_length - piano_roll.shape[1]
        padding = torch.zeros((piano_roll.shape[0], padding_length), dtype=piano_roll.dtype)
        piano_roll = torch.cat((piano_roll, padding), dim=1)
    return piano_roll

class MidiDataset(Dataset):
    def __init__(self, data_dir, resolution=100, fixed_length=10000):  # Use a smaller fixed length
        super().__init__()
        self.data_dir = data_dir
        self.midi_files = []
        self.labels = []
        self.resolution = resolution
        self.fixed_length = fixed_length

        self._load_data()

    def _load_data(self):
        # Iterate over artist directories to load MIDI files
        for artist_name in os.listdir(self.data_dir):
            artist_path = os.path.join(self.data_dir, artist_name)
            if os.path.isdir(artist_path):
                for midi_file in os.listdir(artist_path):
                    if midi_file.endswith('.mid'):
                        self.midi_files.append(os.path.join(artist_path, midi_file))
                        self.labels.append(artist_name)

        self.label2id = {label: idx for idx, label in enumerate(set(self.labels))}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

    def _process_midi(self, midi_file):
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_file)
        except Exception as e:
            print(f"Skipping file {midi_file} due to error: {e}")
            return None

        # Get the piano roll
        piano_roll = midi_data.get_piano_roll(fs=self.resolution)
        piano_roll = np.clip(piano_roll, 0, 1)

        # Crop the piano roll to the fixed length
        piano_roll_tensor = torch.tensor(piano_roll, dtype=torch.float32)
        piano_roll_tensor = crop_piano_roll(piano_roll_tensor, self.fixed_length)

        return piano_roll_tensor

    def __getitem__(self, idx):
        midi_file = self.midi_files[idx]
        label = self.labels[idx]

        # Process the MIDI file to get the piano roll tensor
        piano_roll_tensor = self._process_midi(midi_file)
        if piano_roll_tensor is None:
            return None

        # Adjust the piano roll tensor to ensure it can be reshaped correctly
        if piano_roll_tensor.shape[1] % 88 != 0:
            piano_roll_tensor = adjust_piano_roll(piano_roll_tensor)
            
        # Ensure sequence length is divisible by 88
        piano_roll_tensor = adjust_sequence_length(piano_roll_tensor)

        input_ids = piano_roll_tensor.view(-1)

        # Create an attention mask for the valid inputs (all ones since all inputs are valid)
        attention_mask = torch.ones_like(input_ids)

        # Token type IDs are not necessary for most music tasks but included here for completeness
        token_type_ids = torch.zeros_like(input_ids)

        # Convert label to a tensor
        label_id = torch.tensor([self.label2id[label]], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": label_id
        }

    @property
    def label_nums(self):
        return len(self.label2id)

    def __len__(self):
        return len(self.midi_files)

    @staticmethod
    def collate(batch_data):
        batch_data = [data for data in batch_data if data is not None]  # Remove None entries
        batch_input_ids = torch.nn.utils.rnn.pad_sequence([data["input_ids"] for data in batch_data], batch_first=True, padding_value=0)
        batch_token_type_ids = torch.nn.utils.rnn.pad_sequence([data["token_type_ids"] for data in batch_data], batch_first=True, padding_value=0)
        batch_attention_mask = torch.nn.utils.rnn.pad_sequence([data["attention_mask"] for data in batch_data], batch_first=True, padding_value=0)
        batch_labels = torch.cat([data["labels"] for data in batch_data], dim=0)

        print(f"Collate - Batch input IDs shape: {batch_input_ids.shape}")
        print(f"Collate - Batch token type IDs shape: {batch_token_type_ids.shape}")
        print(f"Collate - Batch attention mask shape: {batch_attention_mask.shape}")
        print(f"Collate - Batch labels shape: {batch_labels.shape}")

        return {
            "batch_input_ids": batch_input_ids,
            "batch_token_type_ids": batch_token_type_ids,
            "batch_attention_mask": batch_attention_mask,
            "batch_labels": batch_labels
        }

    @property
    def label_nums(self):
        return len(self.label2id)

    def __len__(self):
        return len(self.midi_files)

    @staticmethod
    def collate(batch_data):
        batch_data = [data for data in batch_data if data is not None]  # Remove None entries
        batch_input_ids = torch.nn.utils.rnn.pad_sequence([data["input_ids"] for data in batch_data], batch_first=True, padding_value=0)
        batch_token_type_ids = torch.nn.utils.rnn.pad_sequence([data["token_type_ids"] for data in batch_data], batch_first=True, padding_value=0)
        batch_attention_mask = torch.nn.utils.rnn.pad_sequence([data["attention_mask"] for data in batch_data], batch_first=True, padding_value=0)
        batch_labels = torch.cat([data["labels"] for data in batch_data], dim=0)

        print(f"Collate - Batch input IDs shape: {batch_input_ids.shape}")
        print(f"Collate - Batch token type IDs shape: {batch_token_type_ids.shape}")
        print(f"Collate - Batch attention mask shape: {batch_attention_mask.shape}")
        print(f"Collate - Batch labels shape: {batch_labels.shape}")

        # Check if all input_ids have the same length
        input_lengths = [data["input_ids"].size(0) for data in batch_data]
        print(f"Input lengths: {input_lengths}")
        assert all(length == input_lengths[0] for length in input_lengths), "Not all input sequences have the same length"

        return {
            "batch_input_ids": batch_input_ids,
            "batch_token_type_ids": batch_token_type_ids,
            "batch_attention_mask": batch_attention_mask,
            "batch_labels": batch_labels
        }

if __name__ == '__main__':
    train_data = MidiDataset(args.data_dir)
    train_loader = DataLoader(train_data, batch_size=args.train_batch_size, collate_fn=train_data.collate)

    for batch in train_loader:
        print(batch["batch_input_ids"])
        break
