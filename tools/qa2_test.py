import os
os.environ['HF_HOME'] = '/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/.cache'

from io import BytesIO
import json
import torch
from tqdm import tqdm
from urllib.request import urlopen
import librosa, audiofile
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
model = model.to(device)

def annotate_audio(meta_info):
    audio_path = meta_info["audio_path"]
    begin, end = meta_info["begin"], meta_info["end"]
    speaker = meta_info["interlocutor"]
    source_rate = 16000
    conversation = [
        {'role': 'system', 'content': 'You are a helpful assistant.'}, 
        {"role": "user", "content": [
            {"type": "audio", "audio_url": audio_path},
            {"type": "text", "text": f"Analyze the paralinguistic features in this counseling speech, \
focusing on tone, pitch, and intonation \
that is relevant to the speaker's role ({speaker}) and \
their conversational intent (e.g., reflection, question, change talk). \
Make the analysis brief, containing at maximum 2-3 sentences."},
        ]},
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios = []
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    # Resample and append audio data
                    # audio = librosa.resample(ele["audio_data"], 
                    #                         orig_sr=source_rate, 
                    #                         target_sr=processor.feature_extractor.sampling_rate)
                    # device
                    fname = ele["audio_url"]
                    signal, source_rate = audiofile.read(fname)
                    audio = signal.squeeze()

                    begin_time = sum(x * int(t) for x, t in zip([3600, 60, 1], begin.split(":")))
                    end_time = sum(x * int(t) for x, t in zip([3600, 60, 1], end.split(":")))

                    audio = audio[begin_time * 16000:end_time * 16000]
                    # audio = audio.unsqueeze(0)
                    # send audio to device
                    # audio = audio.to(device)
                    audios.append(audio)
    # print(text, audios)
    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
    # print(inputs)
    # print(inputs.keys())
    # print("Device: ", device)
    # inputs.input_ids = inputs.input_ids.to(device)
    # inputs.input_features = inputs.input_features.to(device)
    # inputs.attention_mask = inputs.attention_mask.to(device)
    # inputs.feature_attention_mask = inputs.feature_attention_mask.to(device)
    inputs = inputs.to(device)

    generate_ids = model.generate(**inputs, max_length=256)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]

    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response



def main():
    fname = "data/converted_segmental_information.json"
    with open(fname, "r") as fh:
        seg = json.load(fh)


    count = 0
    for seg_id, seg_info in tqdm(seg.items()):
        text = seg_info["utterance_text"]
        speaker = seg_info["interlocutor"]

        begin = seg_info["begin"]
        end = seg_info["end"]
        if begin == None or end == None:
            begin_time = 0
            end_time = 0
        else:
            begin_time = sum(x * int(t) for x, t in zip([3600, 60, 1], begin.split(":")))
            end_time = sum(x * int(t) for x, t in zip([3600, 60, 1], end.split(":")))

        if end_time - begin_time < 5:
            analysis = "EMPTY"
        else:
            analysis = annotate_audio(seg_info)
            print("="*30)
            print("Text: ", text)
            print("Speaker: ", speaker)
            print("Analysis: ", analysis)

        seg_info["audio_analysis"] = analysis
        if count > float("inf"):
            break
        count += 1



    with open("data/converted_segmental_information_with_audio_analysis.json", "w") as fh:
        json.dump(seg, fh, indent=4)

if __name__ == "__main__":
    main()