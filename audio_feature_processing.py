import numpy as np
import audiofile, audresample
import opensmile
import json
from tqdm import tqdm

def get_audio_features(dic, max_audio_s=100):
    fname = dic["audio_path"]
    begin = dic["begin"]
    end = dic["end"]

    signal, source_rate = audiofile.read(fname)
    # skipping resampling bc it's already 16k
    # signal = audresample.resample(signal, source_rate, 16000)
    audio = signal.squeeze()


    # turn 'begin': '00:00:39', 'end': '00:00:41' into seconds
    begin_time = sum(x * int(t) for x, t in zip([3600, 60, 1], begin.split(":")))
    end_time = sum(x * int(t) for x, t in zip([3600, 60, 1], end.split(":")))

    audio = audio[begin_time * 16000:end_time * 16000]
    if max_audio_s:
        audio = audio[:max_audio_s*16000]  

    smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
    )
    features = smile.process_signal(
        signal,
        16000
    )

    low_level_smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )
    low_level_features = low_level_smile.process_signal(
        signal,
        16000
    )
    combined_features = {**features, **low_level_features}
    return combined_features


dicname = "./data/converted_segmental_information.json"
with open(dicname, "r") as f:
    dic = json.load(f)

for key, entry in tqdm(dic.items()):
    features = get_audio_features(entry)
    print("*"*50)
    print(entry)
    print(features)
    input()