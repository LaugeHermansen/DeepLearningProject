import pytorch_lightning as pl
import torch
from speech_datamodule import SpeechDataModule, SpeechDatasetDisk
from params import params
import os
from glob import glob
from tqdm import tqdm

pl.seed_everything(42, workers=True)

data = SpeechDataModule(params)
data.setup('fit')

for f in data.val_set.audio_file_paths:
    # move file to validation folder
    target = f.replace('train', 'val')
    os.renames(f, target)
