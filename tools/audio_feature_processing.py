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
    
    # functionals
    smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
    )
    features = smile.process_signal(
        signal,
        16000
    )

    # low level descriptors
    # low_level_smile = opensmile.Smile(
    #     feature_set=opensmile.FeatureSet.eGeMAPSv02,
    #         feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    # )
    # low_level_features = low_level_smile.process_signal(
    #     signal,
    #     16000
    # )
    combined_features = {**features}
    # combined_features = {**low_level_features}
    return combined_features


dicname = "./data/converted_segmental_information.json"
with open(dicname, "r") as f:
    dic = json.load(f)

dic = {k: v for k, v in dic.items() if v["mi_quality"] == "high"}

# first 4 of dic
dic = {k: v for k, v in list(dic.items())[:4]}

features = {}
for key, entry in tqdm(dic.items()):
    mi_quality = entry["mi_quality"]

    feat = get_audio_features(entry)
    # print(type(feat))
    # print(feat["equivalentSoundLevel_dBp"])
    # print(type(feat["equivalentSoundLevel_dBp"]))
    # input()
    feats = {}
    for k, v in feat.items():
        # print(k)
        # print(v.values[0])
        # input()
        feats[k] = v.values[0]
    features[key] = feats


# save to pkl
import pickle
with open("./data/audio_features_converted.pkl", "wb") as f:
    pickle.dump(features, f)
    # print("*"*50)
    # print(entry)
    # print(features)
    # input()