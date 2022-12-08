#%%
#%%

import argparse
import numpy as np
import random
import os
import shutil
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import pandas as pd

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

from eat_my_balls import DiffWave, get_pretrained_model, AttrDict, base_params, SpeechDataModule
from tools import mkdir, glob, Timer, get_cmap
import os
import time


params = AttrDict(base_params)



use_cuda = True
use_cuda = torch.cuda.is_available() and use_cuda
device = torch.device('cuda:0') if use_cuda else torch.device('cpu')


model_pretrained = DiffWave(params).to(device)
checkpoint = torch.load('diffwave-ljspeech-22kHz-1000578.pt', map_location=device)
model_pretrained.load_state_dict(checkpoint['model'])

model_finetuned = DiffWave(params).to(device)
checkpoint = torch.load(r"data\spectrogram\danish\test_42\checkpoints\k-epoch=989-val_loss=0.029107.ckpt", map_location=device)
model_finetuned.load_state_dict(checkpoint['state_dict'])


filenames = glob("data/NST/dataset_male/test/**/*.wav", recursive=True)
spectrograms = [f + ".spec.npy" for f in filenames]


results = mkdir("results")

np.random.seed(42)
indices = np.random.choice(len(spectrograms), 200, replace=False)

for i in tqdm(indices):
    if not os.path.exists(f"{i}_finetuned.wav"):
        spec = np.expand_dims(np.load(spectrograms[i]),0)
        audio_pretrained = model_pretrained.predict_step({"spectrogram": spec}, 0, progress_bar=False)
        audio_finetuned = model_finetuned.predict_step({"spectrogram": spec}, 0, progress_bar=False)
        torchaudio.save(results + f"{i}_pretrained.wav", audio_pretrained.to("cpu"), 22050)
        torchaudio.save(results + f"{i}_finetuned.wav", audio_finetuned.to("cpu"), 22050)
        np.save(results + f"{i}_spec_original.npy", spec)

