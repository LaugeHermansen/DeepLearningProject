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


timer = Timer()



class Data:
    def __init__(self, path):
        self.filenames = [x.replace("\\","/") for x in glob(f"{path}/**/*.wav", recursive=True) if x[-9:-4] != "short"]
    
    def unpack_filename(self, f):
        return f, os.path.exists(f + ".short.wav")

    def load(self, f, short=True):
        f, short_exists = self.unpack_filename(f)
        if short and short_exists: audio, sample_rate = torchaudio.load(f + ".short.wav")
        else:                      audio, sample_rate = torchaudio.load(f)
        return audio, sample_rate, f, short_exists

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor): idx = idx.detach().item()
        if isinstance(idx, int):   return self.unpack_filename(self.filenames[idx])
        elif isinstance(idx, str): return self.unpack_filename(idx)
        else:                      return list(map(self.unpack_filename, self.filenames[idx]))
    def __len__(self):
        return len(self.filenames)
    def __iter__(self):
        return map(self.unpack_filename, self.filenames)
        

path1 = "D:/DTU/DeepLearningProject/data\\NST - Kopi"
path2 = "D:/DTU/DeepLearningProject/data\\NST"

path = path1

data = Data(path1)


#%%

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


def remove_silent(audio: torch.Tensor, kernel_size: float, sample_rate: int, threshold: float, min_count_int: float):
    kernel_size = int(kernel_size*sample_rate)
    min_count_int = int(min_count_int*sample_rate)
    conv_audio, kernel_size = conv(torch.abs(audio), kernel_size)
    below_thr = (conv_audio < threshold).int()
    conv_below_thr, min_count_int = conv(below_thr, min_count_int, mean=False)
    all_below_thr = torch.all(conv_below_thr == min_count_int, dim=0)
    out = torch.stack([a[~all_below_thr] for a in audio], dim=1).T
    return out


for f,short_exists in tqdm(data):
    short_path = f"{f}.short.wav"
    if not short_exists:
        audio, sample_rate, _, _ = data.load(f)
        audio_shortened = remove_silent(audio, 0.1, sample_rate, 0.01, 0.5)
        torchaudio.save(short_path, audio_shortened, sample_rate)



#%%

mean = []
length = []
l = []

dy = 0.008

timer = Timer()

for f,se in tqdm(data):
    a,sr,_,_ = data.load(f)
    # a = a
    mean.append(torch.abs(a).mean(axis=-1, keepdim=True))
    l.append(a.shape[-1])

mean = torch.cat(mean)
l = torch.Tensor(l)/sr
sort_idx = l.argsort()

plt.plot(mean[sort_idx])
plt.show()
plt.plot(l[sort_idx])
plt.show()

#%%

a,sr,f,s = data.load(*data[sort_idx[3000]])

plt.plot(torch.arange(len(a[0]))/sr, a[0])
plt.show()

# timer.evaluate("timer")
