import os
import pretty_midi

def validate_midi_files(data_dir):
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.mid') or file.endswith('.midi'):
                file_path = os.path.join(root, file)
                try:
                    midi_data = pretty_midi.PrettyMIDI(file_path)
                    print(f"Validated: {file_path}")
                except Exception as e:
                    print(f"Error in file {file_path}: {e}")

if __name__ == "__main__":
    data_dir = "/home/brad/models/splitfinetuning/data/clean_midi"
    validate_midi_files(data_dir)