import numpy as np
import random
import os

import torch
from torch.utils.data import DataLoader, Dataset, random_split

import torchaudio
from torchaudio.transforms import MelSpectrogram

import pytorch_lightning as pl

from math import floor
from glob import glob
from tqdm import tqdm
from tools import mkdir, Timer, AttrDict


def prepare_data(params, audio_file_paths, audio_dir, spec_file_paths, spec_dir):
    """
    Preprocess audio files and save them as spectrograms.

    Parameters
    ----------
    * params: AttrDict
    * audio_file_paths: list of str - paths to the audio files
    * audio_dir: str - directory containing the audio files (only used for printing the progress bar)
    * spec_file_paths: list of str - paths to the spectrogram files - where the spectrograms will be saved
    * spec_dir: str - directory containing the spectrogram files (only used for printing the progress bar)

    """

    ingnored_idx = []

    mel_args = {
            'sample_rate': params.sample_rate,
            'win_length': params.hop_samples * 4,
            'hop_length': params.hop_samples,
            'n_fft': params.n_fft,
            'f_min': 20.0,
            'f_max': params.sample_rate / 2.0,
            'n_mels': params.n_mels,
            'power': 1.0,
            'normalized': True,
    }

    downscale = params.get("downscale", None)

    if downscale is not None:
        mel_args['win_length'] = int(mel_args['win_length'] / downscale)
        mel_args['hop_length'] = int(mel_args['hop_length'] / downscale)
        mel_args['n_fft'] = int(mel_args['n_fft'] / downscale)
        mel_args['n_mels'] = int(mel_args['n_mels'] * downscale)

    
    mel_spec_transform = MelSpectrogram(**mel_args)
    
    for i in tqdm(range(len(audio_file_paths)), desc=f'Preprocessing {audio_dir}'):
        audio_file_path = audio_file_paths[i]
        spec_file_path = spec_file_paths[i]
        assert os.path.exists(audio_file_path), f"Audio file not found: {audio_file_path}, {audio_dir}, {spec_dir}"
        if os.path.exists(spec_file_path):
            continue
        
        audio, sr = torchaudio.load(audio_file_path)
        audio = torch.clamp(audio[0], -1.0, 1.0)

        if params.sample_rate != sr:
            raise ValueError(f'Invalid sample rate: {sr} != {params.sample_rate} (True).')

        with torch.no_grad():
            spectrogram = mel_spec_transform(audio)
            
            if spectrogram.shape[1] < params.crop_mel_frames:
                ingnored_idx.append(i)
            
            spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
            spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
            np.save(spec_file_path, spectrogram.detach().cpu().numpy())

    return ingnored_idx 


class SpeechDataset(Dataset):
    def __init__(self, audio_dir, spec_dir, use_timing=False):
        super().__init__()
        self.audio_dir = audio_dir
        self.spec_dir = spec_dir
        self.audio_file_paths = np.array([os.path.realpath(f) for f in glob(f"{audio_dir}/**/*.wav", recursive=True)])
        self.audio_filenames = np.array([os.path.relpath(f, self.audio_dir) for f in self.audio_file_paths])
        self.spec_filenames = np.array([f'{f}.spec.npy' for f in self.audio_filenames])
        self.spec_file_paths = np.array([os.path.join(spec_dir, f) for f in self.spec_filenames])
        self.ignored_files = []
        self.timer = Timer(use_timing)

        for f in self.spec_file_paths:
            mkdir(os.path.dirname(f))

    def __repr__(self):
        return f'{self.__class__.__name__}(dirs = {self.audio_dir}, {self.spec_dir}, {len(self)} items, {len(self.ignored_files)} ignored)'

    def split(self, split_ratio=0.1):
        def mask_set(set, mask):
            set.audio_file_paths = self.audio_file_paths[mask]
            set.audio_filenames = self.audio_filenames[mask]
            set.spec_filenames = self.spec_filenames[mask]
            set.spec_file_paths = self.spec_file_paths[mask]

        n_split = floor(len(self) * split_ratio)
        mask = np.random.permutation(len(self)) < n_split
        set1 = self.__class__(self.audio_dir, self.spec_dir)
        set2 = self.__class__(self.audio_dir, self.spec_dir)
        mask_set(set1, mask)
        mask_set(set2, ~mask)
        return set1, set2


    def ignore_item(self, idx):
        self.ignored_files.append(self.audio_file_paths[idx])
        del self.audio_file_paths[idx]
        del self.audio_filenames[idx]
        del self.spec_filenames[idx]
        del self.spec_file_paths[idx]


    def __len__(self):
        return len(self.audio_filenames)
    
    def _load_one_item(self, idx):
        audio_file_path = self.audio_file_paths[idx]
        spec_file_path = self.spec_file_paths[idx]
        signal, _ = torchaudio.load(audio_file_path)
        spectrogram = np.load(spec_file_path).T
        assert signal.shape[0] == 1, f"Only mono audio is supported, found {signal.shape}"
        return signal.squeeze(0), spectrogram

    def __getitem__(self, idx):
        signal, spectrogram  = self._load_one_item(idx)
        return {
            'audio': signal,
            'spectrogram': spectrogram
        }   

    def prepare_data(self, params):
        ignored_idx = prepare_data(params, self.audio_file_paths, self.audio_dir, self.spec_file_paths, self.spec_dir)
        for i in ignored_idx:
            self.ignore_item(i)   

class SpeechDataModule(pl.LightningDataModule):
    def __init__(self, params, use_timing=False):
        
        super().__init__()
        self.use_timing = use_timing
        self.params = params
        
        # this is for improving speed
        self.num_workers = self.params.get('num_workers', os.cpu_count())
        
        # this is for the dataloader
        self.loader_kwargs = {
            'batch_size': params.batch_size,
            'collate_fn': self.collate,
            'num_workers': self.num_workers,
            'pin_memory': True,
            'drop_last': True
        }
    
    def setup(self, stage):
        # Assign Train/val split(s) for use in Dataloaders
        if stage == 'fit':
            audio_path_train = os.path.join(self.params.data_dir_root, self.params.train_dir)
            spec_path_train = os.path.join(self.params.spectrogram_dir_root, self.params.spectrogram_dir, self.params.train_dir)
            # if (self.params.val_dir is None) and False:
            #     # load train set - split train set into train and val
            #     self.val_set, self.train_set = SpeechDataset(audio_path_train, spec_path_train).split(self.params.val_size)
            # else:
            #     # load train and val set
            audio_path_val = os.path.join(self.params.data_dir_root, self.params.val_dir)
            spec_path_val = os.path.join(self.params.spectrogram_dir_root, self.params.spectrogram_dir, self.params.val_dir)
            
            self.val_set = SpeechDataset(audio_path_val, spec_path_val)
            self.train_set = SpeechDataset(audio_path_train, spec_path_train)

            self.val_set.timer = Timer(self.use_timing)
            self.train_set.timer = Timer(self.use_timing)
            self.val_set.prepare_data(self.params)
            self.train_set.prepare_data(self.params)
            
        if stage == 'test':
            audio_path_test = os.path.join(self.params.data_dir_root, self.params.test_dir)
            spec_path_test = os.path.join(self.params.spectrogram_dir_root, self.params.spectrogram_dir, self.params.test_dir)
            self.test_set = SpeechDataset(audio_path_test, spec_path_test)
            self.test_set.timer = Timer(self.use_timing)
            self.test_set.prepare_data(self.params)

    
    def collate(self, minibatch):
        
        samples_per_frame = self.params.hop_samples
        for record in minibatch:
            
            # Filter out records that aren't long enough.
            assert self.params.crop_mel_frames <= len(record['spectrogram'])

            start = random.randint(0, record['spectrogram'].shape[0] - self.params.crop_mel_frames)
            end = start + self.params.crop_mel_frames
            record['spectrogram'] = record['spectrogram'][start:end].T

            start *= samples_per_frame
            end *= samples_per_frame
            record['audio'] = record['audio'][start:end]
            record['audio'] = np.pad(record['audio'], (0, (end-start) - len(record['audio'])), mode='constant')

        audio = np.stack([record['audio'] for record in minibatch])
        spectrogram = np.stack([record['spectrogram'] for record in minibatch])
        
        return {
            'audio': torch.from_numpy(audio),
            'spectrogram': torch.from_numpy(spectrogram),
        }
    
    def train_dataloader(self):
        return DataLoader(self.train_set, shuffle=True, **self.loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_set, shuffle=False, **self.loader_kwargs)

    def test_dataloader(self):
        return DataLoader((self.test_set), shuffle=False, **self.loader_kwargs(self.test_set))


if __name__ == "__main__":
    pass
#%%

