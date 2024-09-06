import json
import os
# from pyannote.audio import Pipelin?e
import torch
import requests
from pytubefix import YouTube
from pydub import AudioSegment
import random, audiofile, audresample
import soundfile as sf
import whisper

# Define the token and URL
token = "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"
url = "https://www.youtube.com/watch?v=0P_4WIZa8aU"
SR = 16000

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

model = whisper.load_model("large") # "base" is also good and faster
# result = model.transcribe(filename)
# print(result)
def transcribe_audio(filename):
    

    signal, source_rate = audiofile.read(filename)
    # print(audiofile.duration(filename))
    signal = audresample.resample(signal, source_rate, SR)
    signal = audresample.remix(
        signal,
        mixdown=True,
    )[0]
    # print(signal.shape)
    result = model.transcribe(signal)
    for turn in result["segments"]:
        print(f"start={turn["start"]:.1f}s stop={turn["end"]:.1f}s / text: {turn["text"]}")
        #print(len(signal), int(turn.start*source_rate),int(turn.end*source_rate))
        segment = signal[int(turn["start"]*SR):int(turn["end"]*SR)]  
        # audiofile.write("./data/temp.wav", segment, source_rate)
        #print(segment)
        sf.write("./data/temp.wav", segment, SR)
        input()

    return 
    # Save diarization result to a file
    with open('./data/diarization.json', 'w') as f:
        json.dump(diarization, f, indent=4)

    print("Diarization results saved to ./data/diarization.json")


def transcribe_audio2(filename):
    import torch
    from transformers import pipeline
    from transformers.utils import is_flash_attn_2_available

    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3", # select checkpoint from https://huggingface.co/openai/whisper-large-v3#model-details
        torch_dtype=torch.float16,
        device="cuda:0", 
        model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
    )

    outputs = pipe(
        filename,
        chunk_length_s=30,
        batch_size=24,
        return_timestamps=True,
    )

    print(outputs)

transcribe_audio(filename)