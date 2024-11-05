import torch
import torch.nn.functional as F
import os
import random
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import json
import torchaudio
import audiofile, audresample
import soundfile as sf
import numpy as np

def set_all_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def create_attention_mask(feats_lens: torch.Tensor) -> torch.Tensor:
    attention_mask = [torch.ones(i) for i in feats_lens]
    return pad_sequence(attention_mask, batch_first=True)


def filter_data(data, filter_key="text_query"):
    new_data = []
    keys = []
    for dat in data:
        if dat[filter_key] not in keys:
            keys += [dat[filter_key]]
            new_data += [dat]
    
    return new_data

def compute_num_audio_embeds(audio_samples, sr=16000):
    """
    Calculates the number of audio embeddings that will be produced by the audio
    encoder. Couuld be off by 1 (less than the actual
    number).
    """
    # HuBERT produces embeddings every 20ms.
    num_embeds = (audio_samples - (sr * 0.01)) // (sr * 0.02)

    # Downsampling factor of 4.
    num_pooled_embeds = int(num_embeds // 4 - 1)
    return num_pooled_embeds



def add_noise(wavform, sr, snr, device=None):
    with torch.no_grad():
        if type(wavform) != torch.Tensor:
            wavform = torch.Tensor(wavform)        
        noise = torch.normal(torch.zeros(wavform.shape[0],wavform.shape[-1]), torch.ones(wavform.shape[0],wavform.shape[-1]))
        snr_dbs = torch.tensor([snr]) 
        if device:
            wavform = wavform.to(device)
            noise = noise.to(device)
            snr_dbs = snr_dbs.to(device)
        # print(wavform.shape, noise.shape, snr_dbs)
        noised = torchaudio.functional.add_noise(torch.Tensor(wavform), torch.Tensor(noise), snr=snr_dbs)[0]
    return noised