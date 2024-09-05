import json
import os
from pyannote.audio import Pipeline
import torch
import requests
from pytubefix import YouTube
from pydub import AudioSegment

# Define the token and URL
token = "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"
url = "https://www.youtube.com/watch?v=0P_4WIZa8aU"

# Download the audio from YouTube
def download_audio_from_youtube(url, filename):
    print(f"Downloading {url} into {filename}")
    yt = YouTube(url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_file = audio_stream.download(output_path='./data', filename='audio.mp4')

    # Convert mp4 to wav
    audio = AudioSegment.from_file(audio_file, format='mp4')
    audio.export(filename, format='wav')

    # Clean up the mp4 file
    os.remove(audio_file)

filename = "./data/audio.wav"
if not os.path.exists('./data'):
    os.makedirs('./data')
download_audio_from_youtube(url, filename)

# Initialize the pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=token)

# Load and process the audio file
def diarize_audio(filename):
    #audio = {"uri": "filename", "audio": filename}
    diarization = pipeline(filename)

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
    return 
    # Save diarization result to a file
    with open('./data/diarization.json', 'w') as f:
        json.dump(diarization, f, indent=4)

    print("Diarization results saved to ./data/diarization.json")

diarize_audio(filename)
