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



params = AttrDict(base_params)

PATH = "data/NST/dataset"

params.data_dir = PATH

dm = SpeechDataModule(params, PATH + "/train/Male", PATH + "/test/Male")
dm.prepare_data()

