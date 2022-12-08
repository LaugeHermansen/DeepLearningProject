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


base_params["data_dir"] = ""
params = AttrDict(base_params)


# load spectrograms from results/ - prepreocess using speech data module. read original spectrograms

dm = SpeechDataModule(params, "")

filenames = glob("results*.wav", recursive=True)

dm.prepare_data(filenames)

original_spectrograms = glob("results*_spec_original.npy")
pretrained_spectrograms = glob("results*_pretrained.wav.spec.npy")
finetuned_spectrograms = glob("results*_finetuned.wav.spec.npy")


loss_fine = []
loss_pre = []

for orig, pre, fine in tqdm(zip(original_spectrograms, pretrained_spectrograms, finetuned_spectrograms)):
    orig_spec = np.load(orig).squeeze(0)
    pre_spec = np.load(pre)
    fine_spec = np.load(fine)
    l = min(orig_spec.shape[1], pre_spec.shape[1], fine_spec.shape[1])
    orig_spec = orig_spec[:, :l]
    pre_spec = pre_spec[:, :l]
    fine_spec = fine_spec[:, :l]

    loss_fine.append(np.mean((orig_spec - fine_spec)**2))
    loss_pre.append(np.mean((orig_spec - pre_spec)**2))


print("Pretrained loss: ", np.mean(loss_pre), "+-", np.std(loss_pre)/np.sqrt(len(loss_pre)))
print("Finetuned loss: ", np.mean(loss_fine), "+-", np.std(loss_fine)/np.sqrt(len(loss_fine)))


