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

class MidiDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.midi_files = []
        self.labels = []

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
        midi_data = pretty_midi.PrettyMIDI(midi_file)
        for instrument in midi_data.instruments:
            instrument.program = 0
        # Get the piano roll, which is a binary matrix where rows correspond to pitches
        # and columns correspond to fixed time steps.
        piano_roll = midi_data.get_piano_roll(fs=10)  # 'fs' is the sampling frequency in Hz
    
        # Convert the piano roll to a PyTorch tensor.
        # Note: Depending on the model's expected input, you might need to process or reshape this tensor further.
        # Here, we collapse the pitch dimension to a single channel.
        # You can also consider normalizing or scaling the piano roll data as needed.
    
        # Clip the values to 0 or 1 because piano roll can have overlapping notes leading to values >1
        piano_roll = np.clip(piano_roll, 0, 1)
        
        # Convert the piano roll to tensor
        piano_roll_tensor = torch.tensor(piano_roll, dtype=torch.float32)
        
        # Some models might expect a 3D batch x channels x sequence format, where channels = 1 for greyscale-like data
        # Uncomment the following line if your model expects a pseudo-3D input:
        # piano_roll_tensor = piano_roll_tensor.unsqueeze(0)  # Adds a batch dimension
    
        return piano_roll_tensor

    def __getitem__(self, idx):
        midi_file = self.midi_files[idx]
        label = self.labels[idx]
    
        # Process the MIDI file to get the piano roll tensor
        piano_roll_tensor = self._process_midi(midi_file)
        
        # Adjust the piano roll tensor to ensure it can be reshaped correctly
        if piano_roll_tensor.shape[1] % 88 != 0:
            piano_roll_tensor = adjust_piano_roll(piano_roll_tensor)
    
        # If the piano roll tensor is still 2D (pitch x time), you may need to add a channel dimension
        # Uncomment the following line if your model expects a pseudo-3D input (batch x channel x pitch x time)
        # piano_roll_tensor = piano_roll_tensor.unsqueeze(0)  # Adds a channel dimension
    
        # Since BERT and similar models usually expect a sequence, we might need to adjust the piano roll data.
        # Flatten the piano roll to fit into a 1D input sequence if necessary:
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
        batch_input_ids = torch.nn.utils.rnn.pad_sequence([data["input_ids"] for data in batch_data], batch_first=True, padding_value=0)
        batch_token_type_ids = torch.nn.utils.rnn.pad_sequence([data["token_type_ids"] for data in batch_data], batch_first=True, padding_value=0)
        batch_attention_mask = torch.nn.utils.rnn.pad_sequence([data["attention_mask"] for data in batch_data], batch_first=True, padding_value=0)
        batch_labels = torch.cat([data["labels"] for data in batch_data], dim=0)

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
#if __name__ == '__main__':
 #   dataset = MidiDataset(args.data_dir)
  #  data_loader = DataLoader(dataset, batch_size=1, shuffle=True)  # Set batch_size as needed

   # for i, data in enumerate(data_loader):
    #    print(f"Sample {i+1}")
     #   print("Input IDs Shape:", data['input_ids'].shape)
      #  print("Token Type IDs Shape:", data['token_type_ids'].shape)
       # print("Attention Mask Shape:", data['attention_mask'].shape)
        #print("Labels:", data['labels'])

#        if i == 0:  # Only visualize the first sample
 #           print("Attempting to reshape Input IDs to [-1, 88]")
  #          try:
   #             piano_roll = data['input_ids'].view(-1, 88)
    #            print("Reshaped Piano Roll Shape:", piano_roll.shape)
     #           # Visualization code here
      #      except Exception as e:
       #         print("Error reshaping piano roll:", str(e))
        
        #if i >= 4:  # Check the first 5 samples
         #   break
