import os
import requests
import time
import csv

def get_artist_names(data_dir):
    """
    Iterates through the folders in the given directory and returns a list of folder names.
    
    Args:
    data_dir (str): Path to the directory containing artist folders.
    
    Returns:
    list: A list of folder names.
    """
    artist_names = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]
    return artist_names

def get_artist_genre(artist_name):
    """
    Fetches the genre of an artist from the MusicBrainz API.
    
    Args:
    artist_name (str): The name of the artist.
    
    Returns:
    str: The genre of the artist or 'Unknown' if not found.
    """
    url = f"https://musicbrainz.org/ws/2/artist/?query=artist:{artist_name}&fmt=json"
    headers = {
        'User-Agent': 'GenreFetcher/1.0 (your_email@example.com)'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if 'artists' in data and len(data['artists']) > 0:
            # We take the first result assuming it's the best match
            artist = data['artists'][0]
            if 'tags' in artist and len(artist['tags']) > 0:
                # We take the first genre tag
                genre = artist['tags'][0]['name']
                return genre
    return 'Unknown'

def write_labels_csv(artist_names, output_file):
    """
    Writes the artist names and their genres to a CSV file.
    
    Args:
    artist_names (list): A list of artist names.
    output_file (str): Path to the output CSV file.
    """
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Artist", "Genre"])  # Write header
        for artist in artist_names:
            genre = get_artist_genre(artist)
            writer.writerow([artist, genre])
            time.sleep(1)  # Respectful delay between API calls

if __name__ == "__main__":
    data_dir = '/home/brad/models/splitfinetuning/data/clean_midi'
    output_file = 'labels.csv'  # Path to the output CSV file

    artist_names = get_artist_names(data_dir)
    
    write_labels_csv(artist_names, output_file)