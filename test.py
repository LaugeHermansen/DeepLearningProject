from main import (get_pretrained_model, root_directory, base_params, AttrDict, Dataset, 
                 torchaudio, np, glob,  )

base_params = AttrDict(base_params)

# import numpy as np
# import random
# import os

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Dataset, random_split
# from torch.utils.data.distributed import DistributedSampler

# import torchaudio
# from torchaudio.transforms import MelSpectrogram

# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.loggers import TensorBoardLogger

# from math import sqrt
# from glob import glob
# from tqdm import tqdm

class LaugesData(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.audio_filenames = glob(f'{data_path}/**/*.wav', recursive=True)
        self.spec_filenames = [f'{audio_filename}.spec.npy' for audio_filename in self.audio_filenames]
            
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        audio, sample_rate = torchaudio.load(self.audio_filenames[idx])
        spectrogram = np.load(self.spec_filenames[idx]).T
        return {
                'audio': audio.squeeze(0),
                'spectrogram': spectrogram
                }





model = get_pretrained_model(f'{root_directory}/data/diffwave-ljspeech-22kHz-1000578.pt', base_params)
print(model)