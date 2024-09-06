import json
import os
from pyannote.audio import Pipeline
import torch
import requests
from pytubefix import YouTube
from pydub import AudioSegment
import random, audiofile, audresample
import soundfile as sf
import whisper
import fire
import tqdm

# Define the token and URL
token = "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"
TR_SR = 16000 # transcribe
DR_SR = 44100 # diarize

def get_audio_from_youtube(url):
    # print(url)
    yt = YouTube(url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    temp_name = "audio.mp4"
    output_path = "./data"
    audio_file = audio_stream.download(output_path=output_path, filename=temp_name)
    # return audio_file
    # print(audio_file)
    # Convert mp4 to wav
    #audio = AudioSegment.from_file(audio_file, format='mp4')
    audio, source_rate = audiofile.read(audio_file)
    #audio.export(filename, format='wav')

    # Clean up the mp4 file
    os.remove(audio_file)
    

    return audio, source_rate


def diarize_audio(signal, pipeline, source_rate, device=torch.device("cuda:0")):
 
    signal = audresample.resample(signal, source_rate, DR_SR)
    signal = audresample.remix(
        signal,
        mixdown=True,)[0]
    signal = torch.Tensor(signal).to(device)


    diarization = pipeline({"waveform": signal.unsqueeze(0), "sample_rate": DR_SR}, min_speakers=2, max_speakers=2)
    
    prev_speaker = None

    result = []
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

        #print(f"start={start:.1f}s stop={end:.1f}s speaker_{speaker}")

        #print(len(signal), int(turn.start*source_rate),int(turn.end*source_rate))
        segment = signal[int(start*DR_SR):int(end*DR_SR)]#.cpu()
        # audiofile.write("./data/temp.wav", segment, source_rate)
        # segment = segment.numpy()

        result.append({"segment": segment, "speaker": speaker})
        # print(segment.shape)

        #sf.write("./data/temp.wav", segment, SR)
    return result


def transcribe_audio(signal, model, source_rate, device=torch.device("cuda:0")):
    
    # print(audiofile.duration(filename))
    signal = audresample.resample(signal.cpu(), source_rate, TR_SR)
    signal = audresample.remix(
        signal,
        mixdown=True,
    )[0]
    
    signal = torch.Tensor(signal).to(device)
    #print(signal.shape)
    transcribed = model.transcribe(signal)

    result = []
    for turn in transcribed["segments"]:
        text = turn["text"]
        #print(f"start={turn["start"]:.1f}s stop={turn["end"]:.1f}s / text: {turn["text"]}")
        #print(len(signal), int(turn.start*source_rate),int(turn.end*source_rate))
        segment = signal[int(turn["start"]*TR_SR):int(turn["end"]*TR_SR)]  
        # audiofile.write("./data/temp.wav", segment, source_rate)
        #print(segment)
        #sf.write("./data/temp.wav", segment, sample_rate)
        #input()
        result.append({"segment": segment, "text": text})
    return result

def process_sessions(device=0, whisper_model="large"):
    device = torch.device(f"cuda:{device}")
    # (1) Read the urls and get audios
    audios = {}
    with open("./data/HighLowQualityCounseling/urls.csv", "r") as fh:
        lines = fh.readlines()[1:]
        # lines = lines[:2]
        for line in tqdm.tqdm(lines):
            session_id = line.split(",")[0]
            if not os.path.exists("./data/session_audios"):
                os.makedirs("./data/session_audios")
            if os.path.exists(f"./data/session_audios/{session_id}.wav"):
                audio, source_rate = audiofile.read(f"./data/session_audios/{session_id}.wav")
                audios[session_id] = {"audio": audio, "source_rate": source_rate}
            else:
                url = line.split(",")[1]
                try:
                    get = get_audio_from_youtube(url)
                    audio, source_rate = get[0], get[1]
                    audios[session_id] = {"audio": audio, "source_rate": source_rate}
                    audiofile.write(f"./data/session_audios/{session_id}.wav", audio, source_rate)
                except Exception as e:
                    print(e)
                    continue

    # (2) diarize 
    # Initialize the pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    pipeline.to(device)

    diarized = {}
    for session_id, get in tqdm.tqdm(audios.items()):
        audio, source_rate = get["audio"], get["source_rate"]
        result = diarize_audio(audio, pipeline, source_rate, device )
        diarized[session_id] = result

    # (3) then transcribe
    model = whisper.load_model(whisper_model) # "base" is also good and faster
    model.to(device)

    dt = {}
    for session_id, dlist in tqdm.tqdm(diarized.items()):
        count = 0 
        for v in dlist:
            dseg = v["segment"]
            speaker = v["speaker"]
            tlist = transcribe_audio(dseg, model, DR_SR, device)
            for vv in tlist:
                tseg = vv["segment"]
                text = vv["text"]
                
                
                seg_id = f"{session_id}_{count}"
                print(seg_id, speaker, text)
                count += 1
                dt[seg_id] = {"segment": tseg, "text": text, "speaker": speaker}

    # save


    if not os.path.exists(f'data/segment_files_{whisper_model}'):
        os.makedirs(f'data/segment_files_{whisper_model}')

    for seg_id, dic in dt.items():
        segment = dic["segment"]
        text = dic["text"]
        speaker = dic["speaker"]
        seg_path = f"./data/segment_files_{whisper_model}/{seg_id}.wav"
        sf.write(seg_path, segment.cpu(), TR_SR)
        dic["segment"] = seg_path
    
    with open(f"./data/seg_meta_{whisper_model}.json", "w") as fh:
        json.dump(dt, fh, indent=4)

    return 

def main():

    fire.Fire(process_sessions)

main()