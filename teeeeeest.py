# import pytorch_lightning as pl
# import torch
# from speech_datamodule import SpeechDataModule, SpeechDatasetDisk
# from params import params
# import os
from glob import glob
from tqdm import tqdm

# pl.seed_everything(42, workers=True)

# audio_dir = os.path.join(params.data_dir_root, params.train_dir)
# spec_dir = os.path.join('spectrograms', params.train_dir)

# # data = SpeechDatasetDisk(params)
outs = []
for i in tqdm(range(1000)):
    outs.append(glob("/dtu/blackhole/11/155505/audio/**/*.wav", recursive=True))

assert len(outs[0]) != 0, "no files found"

for k in tqdm(zip(*outs), total = len(outs[0])):
    assert len(set(k)) == 1, "not same order"
    assert len(k) == len(outs)
    
print(f"Everything OK: The list of files were all in the same order: n_files={len(outs[0])}, n_trials={len(outs)}")
