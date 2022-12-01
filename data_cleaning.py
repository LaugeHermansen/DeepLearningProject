#%%


import numpy as np
import os
import torchaudio
from torchaudio.transforms import Resample
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from timer import Timer
import math
from collections import defaultdict
import shutil
import pandas as pd

#%%
####### helper functions and classes

class Data:
    def __init__(self, path, load=True):
        self.path = path
        success = True
        if load:
            try: self.load()
            except:
                success = False
                print("loading failed")
        if not load or not success:
            self.filenames = [x.replace("\\","/") for x in glob(f"{path}/**/*.wav", recursive=True)]
            self.audio_lengths = {f:None for f in self.filenames}

    def get_audio_len(self,idx):
        out = self[idx]
        assert isinstance(out, tuple), ValueError("only one index at a time when getting lenghts")
        if self.audio_lengths[out[0]] is None: self.load_audio(out[0])
        return self.audio_lengths[out[0]]

    def is_stripped(self, f):
        return f, f[-12:-4] == "stripped"
    
    def stripped_version_exists(self, f):
        return os.path.exists(f[:-4] + "stripped.wav")

    def load_audio(self, f):
        is_stripped = self.is_stripped(f)
        audio, sample_rate = torchaudio.load(f)
        self.audio_lengths[f] = audio.shape[1]
        return audio, sample_rate, f, is_stripped

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor): idx = idx.detach().item()
        if isinstance(idx, int):   return self.is_stripped(self.filenames[idx])
        elif isinstance(idx, str): return self.is_stripped(idx)
        else:                      return list(map(self.is_stripped, self.filenames[idx]))

    def __len__(self):
        return len(self.filenames)

    def __iter__(self):
        return map(self.is_stripped, self.filenames)
    
    def save(self):
        path = mkdir(self.path + "/data_class")
        pd.to_pickle(self.filenames, path + "/filenames")
        pd.to_pickle(self.audio_lengths, path + "/audio_lengths")
        pd.to_pickle(self.path, path + "/path")

    def load(self):
        path = self.path + "/data_class"
        self.filenames = pd.read_pickle(path + "/filenames")
        self.audio_lengths = pd.read_pickle(path + "/audio_lengths")
        self.path = pd.read_pickle(path + "/path")


def plot(sr, array, *args, **kwargs):
    plt.plot(np.arange(len(array))/sr, array, *args, **kwargs)

def conv(tensor: torch.Tensor, kernel_size: int, mean=True, zero_pad=True):
    """
    audio: waveform tensor of size CxL
    kernel_size: int, size of window
    """
    kernel_size = kernel_size + (kernel_size%2==1)
    z = torch.zeros((tensor.shape[0], int(kernel_size/2)))
    tensor_pad = torch.cat([z, tensor, z], dim=1)
    diff = tensor_pad[:,kernel_size:] - tensor_pad[:,:-kernel_size]
    cum_diff = diff.cumsum(dim=1)
    assert cum_diff.shape == tensor.shape, "conv: output and input tensor should have same shape"
    return (tensor_pad[:,:kernel_size].sum(dim=1,keepdim=True) + cum_diff)*(1/kernel_size if mean else 1), kernel_size


def remove_silence(audio: torch.Tensor, sample_rate: int, kernel_size: float, threshold: float, min_count_in: float):
    kernel_size = int(kernel_size*sample_rate)
    min_count_in = int(min_count_in*sample_rate)
    conv_audio, kernel_size = conv(torch.abs(audio), kernel_size)
    below_thr = (conv_audio < threshold).int()
    conv_below_thr, min_count_in = conv(below_thr, min_count_in, mean=False)
    all_below_thr = torch.all(conv_below_thr == min_count_in, dim=0)
    out = torch.stack([a[~all_below_thr] for a in audio], dim=1).T
    return out


def mkdir(path: str, strip=True):
    path_ = path.strip("/")
    idx = [i for i, c in enumerate(path_) if c == "/"] + [len(path_)]
    add = 0
    for i, pos in reversed(list(enumerate(idx))):
        if os.path.exists(path_[:pos]):
            add = int(os.path.exists(path_[:pos]))
            break
    for j in idx[i+add:]:
        os.mkdir(path_[:j])
    return path
        


###### init data



path1 = "D:/DTU/DeepLearningProject/data/NST - Kopi"
path2 = "D:/DTU/DeepLearningProject/data/NST"

path = path2

data = Data(path)


#%%
#### delete too short files

errors = []

remove_silent_args = {
    "kernel_size": 0.1,
    "threshold": 0.01,
    "min_count_in": 0.5

}

min_length = 5. #seconds

del_file_idx = set()
_, sample_rate, _, _ = data.load_audio(data[0][0])
min_length = math.ceil(sample_rate*min_length)

# find stripped files - and strip non-stripped
for i, (filename, is_stripped) in enumerate(tqdm(data)):
    if not is_stripped and not data.stripped_version_exists(filename):
        try:    
            audio, sample_rate, _, _ = data.load_audio(filename)
        except:
            errors.append(filename)
            continue
        audio = remove_silence(audio, sample_rate, *remove_silent_args.values())
        if audio.shape[1] > min_length:
            new_filename = filename[:-4] + "stripped.wav"
            torchaudio.save(new_filename, audio, sample_rate)
            data.filenames.append(new_filename)
            data.audio_lengths[new_filename] = audio.shape[1]

data.save()

#%%

# delete the non-stripped files
for f, is_stripped in data:
    if not is_stripped:
        os.remove(f)
        del data.audio_lengths[f]

data.filenames = [f for f, i_s in data if i_s]


data.save()

#%%
######### reorganize the files 

new_structure_root = "individuals"
dist_root = path.rstrip("/") + "/" + new_structure_root.strip("/")
dist_root = dist_root.replace("\\", "/")
dist_root = dist_root.strip("/")

individuals = defaultdict(list)

#identify structure
for idx, f in enumerate(tqdm(data.filenames)):
    f_ = f.strip("/")
    i = f_.rfind("/")
    ind = f_[:i]
    audio_name = f_[i+1:]
    individuals[ind].append((idx, f, audio_name))

# move files
for i, (ind, audio_info) in enumerate(tqdm(individuals.items())):
    i = str(i)
    i = "0"*(3-len(i)) + i
    dist = mkdir(f"{dist_root}/{i}")
    for idx, original_path, audio_name in audio_info:
        new_path = f"{dist}/{audio_name}"
        data.filenames[idx] = new_path
        data.audio_lengths[new_path] = data.audio_lengths[original_path]
        del data.audio_lengths[original_path]
        os.rename(original_path, new_path)


#%% Colect samples from each individual in "samples" folder

total_audio_lengths = {}

for ind in os.listdir(dist_root):
    total_audio_lengths[ind] = 0
    for f in glob(f"{dist_root}/{ind}/*.wav"):
        total_audio_lengths[ind] += data.get_audio_len(f.replace("\\","/"))/sample_rate/60


#%%

sorted_by_len = sorted(total_audio_lengths.items(), key=lambda x: x[1], reverse=True)

#%%
sample_path = mkdir(path + "/samples")
for ind, total_len in tqdm(total_audio_lengths.items()):
    try:
        from_path = glob(f"{dist_root}/{ind}/*.wav")[0].replace("\\", "/")
        to_path = f"{sample_path}/{ind}_{int(total_len)}.wav"
        shutil.copy2(from_path, to_path)
    except IndexError:
        print("Index error", ind, total_len)
    except PermissionError:
        print("Permission denied", ind, total_len)
