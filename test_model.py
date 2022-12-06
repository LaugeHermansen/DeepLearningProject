#%%

import argparse
import numpy as np
import random
import os
import shutil
from scipy.ndimage import zoom

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler

import torchaudio
from torchaudio.transforms import MelSpectrogram

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from math import sqrt, floor
from tqdm import tqdm
from datetime import timedelta

from main2 import DiffWave, get_pretrained_model, AttrDict, base_params, SpeechDataModule
from tools import mkdir, glob
import os


def remove_info(image, factor):
    factor = np.array(factor)
    return zoom(zoom(image,factor),(1/factor))

#%%

params = AttrDict(base_params)

use_cuda = True

model = get_pretrained_model("D:\DTU\DeepLearningProject\diffwave-ljspeech-22kHz-1000578.pt", params, use_cuda=use_cuda)

PATH = "data/NST - Kopi/dataset/"


params.data_dir = PATH

dm = SpeechDataModule(params, PATH + "/train/Male", PATH + "/test/Male")
dm.prepare_data()


test_audio_filames = glob(PATH + "/**/*.wav", recursive=True)
test_spectrogram_filames = [x + ".spec.npy" for x in test_audio_filames]


idx = np.random.randint(len(test_audio_filames))
idx = 10

to_root = mkdir("tests")
original_path = test_audio_filames[idx]
filename = original_path[-20:-4:]
spectrogram = np.load(test_spectrogram_filames[idx])


#%%

for r in [(1,1)]:

    audio = model.predict_step({"spectrogram": torch.Tensor(remove_info(spectrogram, r)).unsqueeze(0)}, 0, use_cuda=use_cuda)
    new_filename = f"{filename}_r={r}_idx={idx}"
    to_path_gen = f"{to_root}/{new_filename}_gen.wav"
    torchaudio.save(to_path_gen, audio.cpu(), params.sample_rate)

to_path_orig = f"{to_root}/{filename}_orig.wav"

shutil.copy2(original_path, to_path_orig)



# optimizer = model.configure_optimizers()

# PATH = "checkpoints"

# checkpoint = torch.load(PATH)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']



# %%

np.prod(params.noise_schedule)