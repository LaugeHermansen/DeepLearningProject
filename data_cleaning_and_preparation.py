#%%

import numpy as np
import os
import torchaudio
from torchaudio.transforms import Resample
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from tools import Timer, mkdir, glob, str_replaces
import math
from collections import defaultdict
import shutil
import pandas as pd

from experiment_main import base_params
from tools import AttrDict

base_params = AttrDict(base_params)

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
            self.filenames = glob(f"{path}/**/*.wav", recursive=True)
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



path1 = "D:/DTU/DeepLearningProject/data/NST - Kopi"
path2 = "D:/DTU/DeepLearningProject/data/NST"

PATH = path2# + "/speakers"

data = Data(PATH)


#%%
#### delete too short files - and resample the useful files

errors = []

remove_silent_args = {
    "kernel_size": 0.1,
    "threshold": 0.01,
    "min_count_in": 0.5

}

min_length = 5. #seconds

del_file_idx = set()
_, original_sample_rate, _, _ = data.load_audio(data[0][0])
min_length = math.ceil(original_sample_rate*min_length)
resample = Resample(original_sample_rate, base_params.sample_rate, )

# find stripped files - and strip non-stripped if they're long enough
for i, (filename, is_stripped) in enumerate(tqdm(data, desc = "remove, strip and resample")):
    if not is_stripped and not data.stripped_version_exists(filename):
        try:    
            audio, sr, _, _ = data.load_audio(filename)
        except:
            errors.append(filename)
            continue
        audio = remove_silence(audio, sr, *remove_silent_args.values())
        if audio.shape[1] > min_length:
            if sr == original_sample_rate: audio = resample(audio)
            new_filename = filename[:-4] + "stripped.wav"
            torchaudio.save(new_filename, audio[0].unsqueeze(0), base_params.sample_rate)
            data.filenames.append(new_filename)
            data.audio_lengths[new_filename] = audio.shape[1]

data.save()

#%%

# delete the non-stripped files
for f, is_stripped in tqdm(data, desc="removing unused files"):
    if not is_stripped:
        os.remove(f)
        del data.audio_lengths[f]

data.filenames = [f for f, i_s in data if i_s]

data.save()


#%%
########## reorganize the files 

data = Data(PATH)

speakers_info = {}
speakers_path_to_id = {}


re_str = [
          ("Scr", "scr"),
          ("Speech", "data"),
          ("speech", "data"),
         ]

n_errors = 0
n_flags = 0
#identify structure
for audio_file_idx, filename in enumerate(tqdm(data.filenames, desc="identify structure of data set")):
    try:
        i = filename.rfind("/")
        original_speaker_path = filename[:i]
        audio_name = filename[i+1:]
        spl_file = f"{original_speaker_path.replace('speech', 'data').replace('Speech', 'data')}.spl"
        with open(spl_file) as spl_file_wrapper:
            info_list = spl_file_wrapper.read().split("\n")[22:29]
    except:
        n_errors += 1
        continue
    spl_info = {x.split(">-<")[0][2:]: x.split(">-<")[1] for x in info_list}
    spl_info["Sex"] = "Unknown" if spl_info["Sex"] == "" else spl_info["Sex"]
    if original_speaker_path in speakers_path_to_id:
        speaker_id = speakers_path_to_id[original_speaker_path]
        assert speakers_info[speaker_id]["spl_file"] == spl_file
    else:
        speaker_id = str(len(speakers_path_to_id))
        speaker_id = "0"*(3-len(speaker_id)) + speaker_id
        speakers_path_to_id[original_speaker_path] = speaker_id
        speakers_info[speaker_id] = {
                                        "original_speaker_path": original_speaker_path,
                                        "spl_info": spl_info,
                                        "audio_names": [],
                                        "spl_file": spl_file
                                    }
    speakers_info[speaker_id]["audio_names"].append((audio_file_idx, audio_name))

print(n_errors, "files were omitted due to an error getting the spl file")
pd.to_pickle(speakers_info, PATH + "/speakers_info.pkl")
pd.to_pickle(speakers_path_to_id, PATH + "/speakers_path_to_id.pkl")


#%%
data = Data(PATH)

np.random.seed(42)
new_structure_root = "dataset"
destination_root = PATH.rstrip("/") + "/" + new_structure_root.strip("/")
destination_root = mkdir(destination_root)


#------- train/test split

get_split_property = lambda speaker_info: speaker_info["spl_info"]["Sex"]
train_path = mkdir(f"{destination_root}/train")
test_path = mkdir(f"{destination_root}/test")


test_size_ratio = 0.2
unseen_test_speakers_ratio = 0.1

#--------------------------


removed_files = set()
added_files = set()


n_train = 0
n_test = 0

file_exists_errors = []
speakers_info_items = list(speakers_info.items())
speaker_mask = np.random.permutation(len(speakers_info_items))


n_speakers = defaultdict(int)
for speaker_info in speakers_info.values():
    n_speakers[get_split_property(speaker_info)] += 1
n_speakers = dict(n_speakers)
n_unseen = {key:max(1,int(unseen_test_speakers_ratio*value)) for key,value in n_speakers.items()}


test_speakers  = set()
train_speakers = set()

count_speakers = {key:0 for key in n_speakers}
for speaker_idx in tqdm(speaker_mask, desc = "Train/test split"):
    speaker_id, speaker_info = speakers_info_items[speaker_idx]
    split_property = get_split_property(speaker_info)
    n_audio_names = len(speaker_info["audio_names"])
    split_audio = int(n_audio_names*test_size_ratio)
    audio_mask = np.random.permutation(n_audio_names)
    for count_audio, audio_idx in enumerate(audio_mask):
        audio_idx, audio_name = speaker_info['audio_names'][audio_idx]
        is_test_point = (count_speakers[split_property] < n_unseen[split_property] or count_audio < split_audio)
        if is_test_point:
            temp = mkdir(f"{test_path}/{split_property}/{speaker_id}")
            n_test += 1
            test_speakers.add(speaker_idx)
        else:
            temp = mkdir(f"{train_path}/{split_property}/{speaker_id}")
            n_train += 1
            train_speakers.add(speaker_idx)
        destination_file = f"{temp}/{audio_name}"
        original_file_path = data[audio_idx][0]
        data.filenames[audio_idx] = destination_file
        data.audio_lengths[destination_file] = data.audio_lengths.pop(original_file_path)
        os.rename(original_file_path, destination_file)
        # assert os.path.exists(original_file_path)
        # assert original_file_path not in removed_files
        # assert destination_file not in added_files
        # removed_files.add(original_file_path)
        # added_files.add(destination_file)
    count_speakers[split_property] += 1

    

