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

from archive_2.eat_my_balls import DiffWave, get_pretrained_model, AttrDict, base_params, SpeechDataModule
from tools import mkdir, glob, Timer, get_cmap
import os
import time



########### plug parameters in here <3

# model path
model_path = "D:\DTU\DeepLearningProject\diffwave-ljspeech-22kHz-1000578.pt"

# data paths
data_path = "data/NST/dataset/"
test_path = "data/NST/dataset/test/Female"

train_path = "data/NST/dataset/train/Female"

# output path
output_path = "tests"


#dont touch this
manual_test_id = "NST_test"



####################################

#%%

def reduce(image, factor):
    factor = np.array(factor)
    return image if np.all(factor==1.) else zoom(zoom(image,factor),(1/factor))

params = AttrDict(base_params)

use_cuda = True
use_cuda = torch.cuda.is_available() and use_cuda
device = torch.device('cuda:0') if use_cuda else torch.device('cpu')

checkpoint = torch.load(model_path, map_location=device)
model = DiffWave(params).to(device)

try:
    model.load_state_dict(checkpoint['state_dict'])
except:
    model.load_state_dict(checkpoint['model'])

model.eval()


params.data_dir = data_path

dm = SpeechDataModule(params, train_path, test_path)
dm.prepare_data()


#%%


np.random.seed(42)

test_audio_filenames = glob(dm.val_path + "/**/*.wav", recursive=True)
test_spectrogram_filames = [x + ".spec.npy" for x in test_audio_filenames]


reductions = np.round(np.linspace(0.2, 1, 9), 2)
# reductions = [0.2,1]
n_test_points = 50
batch_size = 24
fast = True

to_root = mkdir(output_path)

file_paths = []
spectrogram_slice = slice(0,100)

indices = np.random.choice(len(test_audio_filenames), n_test_points)


#%%

if manual_test_id is None:
    test_id = 0
    for f in os.listdir(output_path):
        try: test_id = max(int(f)+1, test_id)
        except: pass
else:
    test_id = manual_test_id


#%%
spectrograms = [np.load(test_spectrogram_filames[idx])[:, spectrogram_slice] for idx in indices]
test_paths = [mkdir(f"{to_root}/{test_id}/{i}_{idx}") for i, idx in enumerate(indices)]
original_paths = [(idx, test_audio_filenames[idx]) for idx in indices]
original_filenames = [original_path[original_path.rfind("/")+1:-4:] for idx, original_path in original_paths]


iteration_values = [(i, idx, r1, r2) for i, idx in enumerate(indices) for r1 in reductions for r2 in reductions]

batch_info = []
batch_spectrogram = []
batch_file_paths = []

#%%

for iteration_number, (i, idx, r_freq, r_time) in enumerate(tqdm(iteration_values)):

    new_filename=f"{r_freq},{r_time},{idx}"
    file_path = f"{test_paths[i]}/{new_filename}.wav"
    file_paths.append(file_path)


    if not os.path.exists(file_path):
        spectrogram_reduced = np.expand_dims(reduce(spectrograms[i], (r_freq, r_time)),0)
        batch_info.append((new_filename, file_path))
        batch_spectrogram.append(spectrogram_reduced)

    # if (iteration_number%batch_size == batch_size-1 or iteration_number == len(iteration_values) - 1):
    if len(batch_info) == batch_size or (iteration_number == len(iteration_values) - 1 and len(batch_info) > 0):
        batch_spectrogram = np.stack(batch_spectrogram).squeeze(1)
        audio = model.predict_step({"spectrogram": batch_spectrogram}, 0, fast=fast, progress_bar=False)
        for (new_filename, file_path), a in zip(batch_info, audio):
            torchaudio.save(file_path, a.unsqueeze(0).cpu(), params.sample_rate)
        
        batch_info = []
        batch_spectrogram = []

np.save(f"{to_root}/{test_id}/reductions.npy", reductions)
np.save(f"{to_root}/{test_id}/indices.npy", indices)
np.save(f"{to_root}/{test_id}/file_paths.npy", file_paths)
np.save(f"{to_root}/{test_id}/original_paths.npy", original_paths)

print(f"processed {len(np.unique(indices))} sound files")




#%%




dm = SpeechDataModule(params, f"{to_root}/{test_id}", f"{to_root}/{test_id}")

while not os.path.exists(f"{to_root}/{test_id}/file_paths.npy"):
    for i in tqdm(range(60)):
        time.sleep(1)
    dm.prepare_data(file_paths)




#%%


reductions = np.load(f"{to_root}/{test_id}/reductions.npy")
indices = np.load(f"{to_root}/{test_id}/indices.npy")
file_paths = np.load(f"{to_root}/{test_id}/file_paths.npy")
original_paths = np.load(f"{to_root}/{test_id}/original_paths.npy")



samples_per_frame = params.hop_samples
start = spectrogram_slice.start*samples_per_frame
stop = spectrogram_slice.stop*samples_per_frame
original_spectrograms = {int(idx): np.load(f"{p}.spec.npy") for idx, p in original_paths}

losses = {(r1,r2):[] for r1 in reductions for r2 in reductions}

for file_path in tqdm(file_paths, desc="computing losses"):
    r_freq, r_time, idx = map(float, file_path[:-4].split("/")[-1].split(","))
    idx = int(idx)
    spectrogram_gen = np.load(f"{file_path}.spec.npy")[:,spectrogram_slice]
    spectrogram_og = original_spectrograms[idx][:,spectrogram_slice]
    loss = np.mean((spectrogram_gen - spectrogram_og)**2)
    if (r_freq,r_time) in losses: losses[r_freq,r_time].append(loss.item())


pd.to_pickle(losses, f"{to_root}/{test_id}/losses.pkl")

#%%


# %%




temp_reductions =  np.round(np.linspace(0.3, 1, 8), 2)

losses_img_mean = np.array([[np.mean(losses[r_freq,r_time]) for r_time in temp_reductions]  for r_freq in temp_reductions])
losses_img_sd = np.array([[np.std(losses[r_freq,r_time], ddof=1)*1.96/len(losses[r_freq,r_time])**0.5 for r_time in temp_reductions]  for r_freq in temp_reductions])



r_time = 0.7
idx_time = np.where(temp_reductions == r_time)[0][0]


r_freq = 0.7
idx_freq = np.where(temp_reductions == r_freq)[0][0]

# plot loss heatmap

plt.rcParams['figure.dpi'] = 400

fig, axs = plt.subplots(1, 2, figsize=(13,3.7))

im = axs[0].imshow(losses_img_mean[::-1],cmap=get_cmap())

axs[0].plot([0, len(temp_reductions)-1], [len(temp_reductions)-1-idx_time,len(temp_reductions)-1-idx_time], label = f"$r_{{freq}}={r_freq}$")
axs[0].plot([idx_time, idx_time], [0, len(temp_reductions)-1], label = f"$r_{{time}}={r_time}$")
axs[0].plot([0, len(temp_reductions)-1], [len(temp_reductions)-1, 0], label="$r_{freq}=r_{time}$")

axs[0].set_xticks(np.arange(len(temp_reductions)))#, temp_reductions)
axs[0].set_yticks(np.arange(len(temp_reductions)))#, reversed(temp_reductions))
axs[0].set_xticklabels(temp_reductions)
axs[0].set_yticklabels(reversed(temp_reductions))

axs[0].set_ylabel("Scale on frequency axis")
axs[0].set_xlabel("Scale on time axis")
for pos in ['right', 'top', 'bottom', 'left']:
    axs[0].spines[pos].set_visible(False)
fig.suptitle("Mean error")
fig.colorbar(im, ax =axs[0])
# axs[0].legend()


# plot loss for r_freq = constant
l_mean = losses_img_mean[idx_freq]
l_sd = losses_img_sd[idx_freq]
axs[1].plot(temp_reductions, l_mean, label=f"$r_{{freq}}$ = {r_freq}")
axs[1].fill_between(temp_reductions, l_mean-l_sd, l_mean+l_sd, alpha=0.3, label="95% confidence interval")


# plot loss for r_time = constant
l_mean = losses_img_mean[:,idx_time]
l_sd = losses_img_sd[:,idx_time]
axs[1].plot(temp_reductions, l_mean, label=f"$r_{{time}}$ = {r_time}")
axs[1].fill_between(temp_reductions, l_mean-l_sd, l_mean+l_sd, alpha=0.3, label="95% confidence interval")

# plot loss for r_time = r_freq
l_mean = losses_img_mean.diagonal()
l_sd = losses_img_sd.diagonal()
axs[1].plot(temp_reductions, l_mean, label="$r_{time} = r_{freq}$")
axs[1].fill_between(temp_reductions, l_mean-l_sd, l_mean+l_sd, alpha=0.3, label="95% confidence interval")
axs[1].legend()
axs[1].set_xlabel("Scale on time/frequency axis")
axs[1].set_ylabel("Mean error")

plt.savefig("poster_report/reduced_generation_quality_lines", bbox_inches = 'tight')
plt.show()

#%%

idx = 23

r_freq = 0.5
r_time = 0.5

filename_audio_reduced = f"{to_root}/{test_id}/{idx}_{indices[idx]}/{r_freq},{r_time},{indices[idx]}.wav"
filename_audio_full = f"{to_root}/{test_id}/{idx}_{indices[idx]}//1.0,1.0,{indices[idx]}.wav"
filename_audio_original = f"{original_paths[idx][1]}" 


filename_spec_reduced = f"{to_root}/{test_id}/{idx}_{indices[idx]}/{r_freq},{r_time},{indices[idx]}.wav.spec.npy"
filename_spec_full = f"{to_root}/{test_id}/{idx}_{indices[idx]}/1.0,1.0,{indices[idx]}.wav.spec.npy"
filename_spec_original = filename_audio_original + ".spec.npy"

spectrograms = []
spectrograms.append(np.load(filename_spec_original)[:,spectrogram_slice])
spectrograms.append(zoom(np.load(filename_spec_original)[:,spectrogram_slice], (r_freq, r_time)))
spectrograms.append(np.load(filename_spec_full)[:,spectrogram_slice])
spectrograms.append(np.load(filename_spec_reduced)[:,spectrogram_slice])


def plot_spectrogram(spec, ax, title, r=1):
    ax.imshow(spec, origin="upper")
    ax.set_xticklabels(np.round(ax.get_xticks()/params.sample_rate*params.hop_samples/r, 3))
    ax.set_yticks([])
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")


figsize = (9, 3.7)

fig, axs = plt.subplots(1, 2, figsize=figsize)
plot_spectrogram(spectrograms[0], axs[0], "Full")
plot_spectrogram(spectrograms[1], axs[1], f"Reduced ($r_{{freq}}={r_freq}, r_{{time}}={r_time}$)")
fig.suptitle("Spectrogram from original audio")
fig.savefig("poster_report/reduced_original_quality_spectrograms", bbox_inches = 'tight')

fig, axs = plt.subplots(1, 2, figsize=figsize)
plot_spectrogram(spectrograms[2], axs[0], "Full")
plot_spectrogram(spectrograms[3], axs[1], f"Reduced ($r_{{freq}}={r_freq}, r_{{time}}={r_time}$)")
fig.suptitle("Spectrogram from generated audio")

fig.savefig("poster_report/reduced_generation_quality_spectrograms", bbox_inches = 'tight')

#%%

spectrogram_path = filename_audio_original + ".spec.npy"
audio_path = filename_audio_original

spectrogram = np.load(spectrogram_path)
spectrogram_reduced = np.expand_dims(reduce(spectrogram, (r_freq, r_time)),0)

audio, sr = torchaudio.load(audio_path)
audio_gen = model.predict_step({"spectrogram": spectrogram}, 0)[0]
audio_gen_reduced = model.predict_step({"spectrogram": spectrogram_reduced}, 0)[0]

#%%

torchaudio.save(f"poster_report/{idx}_audio_original.wav", audio, sr)
torchaudio.save(f"poster_report/{idx}_audio_gen.wav", audio_gen.cpu().unsqueeze(0), sr)
torchaudio.save(f"poster_report/{idx}_audio_gen_reduced.wav", audio_gen_reduced.cpu().unsqueeze(0), sr)

