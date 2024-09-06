import json
import os
from pyannote.audio import Pipeline
import torch
import requests
from pytubefix import YouTube
from pydub import AudioSegment
import random, audiofile, audresample
import soundfile as sf

# Define the token and URL
token = "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"
url = "https://www.youtube.com/watch?v=0P_4WIZa8aU"
SR = 44100

# Download the audio from YouTube
def download_audio_from_youtube(url, filename):
    print(f"Downloading {url} into {filename}")
    yt = YouTube(url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    temp_name = "audio.mp4"
    output_path = "./data"
    audio_file = audio_stream.download(output_path=output_path, filename=temp_name)
    # return audio_file
    # print(audio_file)
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
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
# print(pipeline)
pipeline.to(torch.device("cuda"))

# Load and process the audio file
def diarize_audio(filename):
    #audio = {"uri": "filename", "audio": filename}
   

    signal, source_rate = audiofile.read(filename)
    # print(audiofile.duration(filename))
    signal = audresample.resample(signal, source_rate, SR)
    signal = audresample.remix(
        signal,
        mixdown=True,)[0]
    # print(signal.shape)

    signal = torch.Tensor(signal).to(torch.device("cuda"))
    # print(signal.shape)
    # print(signal)

    diarization = pipeline({"waveform": signal.unsqueeze(0), "sample_rate": SR})
    
    prev_speaker = None
    # prev_start, prev_end = None, None
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start = turn.start
        end = turn.end
        if prev_speaker:
            if prev_start <= start and prev_end >= end:
                # we can merge
                # print("merging")
                continue

        prev_start, prev_end = start, end
        prev_speaker = speaker

        print(f"start={start:.1f}s stop={end:.1f}s speaker_{speaker}")
        #print(len(signal), int(turn.start*source_rate),int(turn.end*source_rate))
        segment = signal[int(start*SR):int(end*SR)].cpu()
        # audiofile.write("./data/temp.wav", segment, source_rate)
        segment = segment.numpy()
        # print(segment.shape)

        sf.write("./data/temp.wav", segment, SR)
        input()

    return 
    # Save diarization result to a file
    with open('./data/diarization.json', 'w') as f:
        json.dump(diarization, f, indent=4)

    print("Diarization results saved to ./data/diarization.json")

diarize_audio(filename)
