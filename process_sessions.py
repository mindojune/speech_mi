import json
import os
from pyannote.audio import Pipeline
import torch
import requests
from pytubefix import YouTube
from pydub import AudioSegment
import random, audiofile, audresample
import soundfile as sf
# import whisper
import fire
import tqdm
import csv

TR_SR = 16000 # transcribe
DR_SR = 44100 # diarize

def get_audio_from_youtube(url, path):
    yt = YouTube(url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    output_path = "/".join(path.split("/")[:-1])
    path = path.split("/")[-1]
    audio_file = audio_stream.download(output_path=output_path, filename=path)

    return 


def process_sessions(device=0):
    device = torch.device(f"cuda:{device}")
    # (1) Read the urls and get audios
    audios = {}
    with open("data/AnnoMI-full.csv", "r") as fh:
        total_len = len(fh.readlines())-1
    # total_len = 2
    with open("data/AnnoMI-full.csv", "r") as fh:
        reader = list(csv.DictReader(fh))[:total_len]
        
        for line in tqdm.tqdm(reader, total=total_len):
            seg_id = f"{line["mi_quality"]}_{line["transcript_id"]}_{line["utterance_id"]}"
            session_id = f"{line["mi_quality"]}_{line["transcript_id"]}"
            session_path = f"./data/session_audios/{session_id}.wav"
            if not os.path.exists("./data/session_audios"):
                os.makedirs("./data/session_audios")
            if os.path.exists(session_path):\
                audios[seg_id] = {"audio_path": session_path, "meta": line}
            else:
                url = line["video_url"] 
                try:
                    get_audio_from_youtube(url, session_path)
                    audios[seg_id] = {"audio_path": session_path,  "meta": line}
                except Exception as e:
                    print(e)
                    continue


    for seg_id, dic in audios.items():
        meta = dic["meta"]
        del dic["meta"]
        dic.update(meta)

    # adding timestamps
    for seg_id, dic in audios.items():
        begin = dic["timestamp"]
        next_seg_id = f"{dic["mi_quality"]}_{dic["transcript_id"]}_{int(dic["utterance_id"])+1}"
        if next_seg_id in audios:
            next_seg = audios[next_seg_id]
            end = next_seg["timestamp"]
        else:
            end = None
        dic["begin"] = begin
        dic["end"] = end

    with open(f"./data/segmental_information.json", "w") as fh:
        json.dump(audios, fh, indent=4)

    return 

def main():

    fire.Fire(process_sessions)

main()