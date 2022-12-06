
#%%
from main import (
    get_pretrained_model,
    # root_directory,
    base_params,
    AttrDict,
    )

base_params = AttrDict(base_params)

import numpy as np
# import random
import os

import torch
# import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
# from torch.utils.data.distributed import DistributedSampler

import torchaudio
from torchaudio.transforms import MelSpectrogram

# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.loggers import TensorBoardLogger
 
# from math import sqrt
from glob import glob
from tqdm import tqdm

class LaugesData(Dataset):
    def __init__(self, data_path, params):
        super().__init__()
        self.data_path = data_path
        self.audio_filenames = [x.replace("\\","/") for x in glob(f"{data_path}/**/*.wav", recursive=True)]
        self.spec_filenames = [f'{audio_filename}.spec.npy' for audio_filename in self.audio_filenames]
        self.params = params
            
    def __len__(self):
        return len(self.audio_filenames)

    def __getitem__(self, idx):
        audio, sample_rate = torchaudio.load(self.audio_filenames[idx])
        spectrogram = np.load(self.spec_filenames[idx]).T
        return {
                'audio': audio[0],
                'spectrogram': spectrogram
                }

    def prepare_data(self):

        mel_args = {
                'sample_rate': self.params.sample_rate,
                'win_length': self.params.hop_samples * 4,
                'hop_length': self.params.hop_samples,
                'n_fft': self.params.n_fft,
                'f_min': 20.0,
                'f_max': self.params.sample_rate / 2.0,
                'n_mels': self.params.n_mels,
                'power': 1.0,
                'normalized': True,
        }
        
        mel_spec_transform = MelSpectrogram(**mel_args)
        
        for i in range(len(self)):
            audio_file = self.audio_filenames[i]
            spec_file = self.spec_filenames[i]

            if os.path.exists(spec_file):
                continue
            
            audio, sr = torchaudio.load(audio_file)
            audio = torch.clamp(audio[0], -1.0, 1.0)

            if self.params.sample_rate != sr:
                raise ValueError(f'Invalid sample rate {sr} != {self.params.sample_rate} (True).')

            with torch.no_grad():
                spectrogram = mel_spec_transform(audio)
                
                if spectrogram.shape[1] < self.params.crop_mel_frames:
                    continue
                
                spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
                spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
                np.save(spec_file, spectrogram.cpu().numpy())


#%%


data_path = "data/NST - Kopi/individuals"

# model = get_pretrained_model(f'{root_directory}/data/diffwave-ljspeech-22kHz-1000578.pt', base_params)

data = LaugesData(data_path, base_params)

data.prepare_data()

#%%



a,sr = torchaudio.load("u0091021stripped.wav")
torchaudio.save("u0091021stripped_single.wav", a[0].unsqueeze(0), sr)


#%%

filenames_stereo = glob("data/Test/**/*.wav")

for f in filenames_stereo:
    a,sr = torchaudio.load(f)
    torchaudio.save(f[:-4] + "_mono.wav", a[0].unsqueeze(0), sr)


#%%


filenames_mono = glob("data/Test/**/*_mono.wav")
filenames_stereo = [x[:-9] + ".wav" for x in filenames_mono]

#%%

import torchaudio
from timer import Timer

timer = Timer()

n = 1000


for mono, stereo in tqdm(zip(filenames_mono[:n], filenames_stereo[:n]), total = len(filenames_mono[:n])):
    
    timer("stereo")
    a,sr = torchaudio.load(stereo)
    timer()

    timer("mono")
    a,sr = torchaudio.load(mono)
    timer()

timer.evaluate()
