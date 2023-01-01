from params import params
from diffwave_model import DiffWave
from tools import AttrDict
import torch
import pytorch_lightning as pl
import os
import numpy as np
from tqdm import tqdm
from tools import mkdir
from torch.utils.data import DataLoader, Dataset, random_split
import random
from torch import nn

import pytorch_lightning as pl

from math import floor
from glob import glob
from tqdm import tqdm

class UpsamplerDataset(Dataset):
    def __init__(self, audio_dir, spec_dir, target_dir: str):
        """
        upsampler dataset

        Parameters
        ----------
         * audio_dir: directory containing audio files
         * spec_dir: directory containing spectrograms (the ones with the desired (reduced) resolution)
         * target_dir: directory containing the upsampler outputs (the true output as produced on the original spectrogrms)
        
        """
        self.audio_dir = audio_dir
        self.target_dir = target_dir
        self.spec_dir = spec_dir
        self.audio_file_paths = np.array([os.path.realpath(f) for f in glob(f"{audio_dir}/**/*.wav", recursive=True)])
        self.audio_filenames = np.array([os.path.relpath(f, self.audio_dir) for f in self.audio_file_paths])
        self.spec_filenames = np.array([f'{f}.spec.npy' for f in self.audio_filenames])
        self.spec_file_paths = np.array([os.path.join(spec_dir, f) for f in self.spec_filenames])
        self.target_filenames = np.array([f'{f}.target.npy' for f in self.audio_filenames])
        self.target_file_paths = np.array([os.path.join(target_dir, f) for f in self.target_filenames])

        for f in self.spec_file_paths:
            mkdir(os.path.dirname(f))
    
    def __len__(self):
        return len(self.audio_file_paths)
    
    def __getitem__(self, idx):
        spec = np.load(self.spec_file_paths[idx]).T
        target = np.load(self.target_file_paths[idx])
        return spec, target

    def generate_upsampler_targets(self, spec_full_dir, spectrogram_upsampler: nn.Module):
        """
        Generate the true upsampler outputs on the full spectrogrms

        Parameters
        ----------
        spec_full_dir: directory containing the spectrograms with the original resolution
        spectrogram_upsampler: the upsampler model
        """
        for i in tqdm(range(len(self.dataset)), desc = "generating upsampler data"):
            spec_full_path = os.path.join(spec_full_dir, self.spec_filenames[i])
            assert os.path.exists(spec_full_path), f"full spec {spec_full_path} does not exist"
            assert os.path.exists(self.spec_file_paths[i]), f"spec {self.spec_file_paths[i]} does not exist"
            if not os.path.exists(self.target_file_paths[i]):
                spec_full = np.load(spec_full_path).T
                target = spectrogram_upsampler(spec_full).detach().cpu().numpy()
                np.save(self.target_file_paths[i], target)

class UpsamplerDataModule(pl.LightningDataModule):
    """
    DataModule for the upsampler
    """
    def __init__(self, params: AttrDict):
        super().__init__()
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
    
    def prepare_data(self, spectrogram_upsampler: nn.Module, stage = 'fit'):
        """
        Generate the true upsampler outputs on the full spectrogrms

        Parameters
        ----------
        spectrogram_upsampler: the upsampler model
        stage: 'fit' or 'test'
        """
        self.setup(stage)
        if stage == 'fit':
            spec_full_path_train = os.path.join(self.params.spectrogram_dir_root, self.params.spectrogram_full_dir, self.params.train_dir)
            spec_full_path_val = os.path.join(self.params.spectrogram_dir_root, self.params.spectrogram_full_dir, self.params.val_dir)
            self.train_set.generate_upsampler_targets(spec_full_path_train, spectrogram_upsampler)
            self.val_set.generate_upsampler_targets(spec_full_path_val, spectrogram_upsampler)
        if stage == 'test':
            spec_full_path_test = os.path.join(self.params.spectrogram_dir_root, self.params.spectrogram_full_dir, self.params.test_dir)
            self.test_set.generate_upsampler_targets(spec_full_path_test, spectrogram_upsampler)

    def setup(self, stage):
        if stage == 'fit':
            audio_path_train = os.path.join(self.params.data_dir_root, self.params.train_dir)
            spec_path_train = os.path.join(self.params.spectrogram_dir_root, self.params.spectrogram_dir, self.params.train_dir)
            spec_full_path_train = os.path.join(self.params.spectrogram_dir_root, self.params.spectrogram_full_dir, self.params.train_dir)
            audio_path_val = os.path.join(self.params.data_dir_root, self.params.val_dir)
            spec_path_val = os.path.join(self.params.spectrogram_dir_root, self.params.spectrogram_dir, self.params.val_dir)
            spec_full_path_val = os.path.join(self.params.spectrogram_dir_root, self.params.spectrogram_full_dir, self.params.val_dir)
            
            self.val_set = UpsamplerDataset(audio_path_val, spec_path_val, spec_full_path_val)
            self.train_set = UpsamplerDataset(audio_path_train, spec_path_train, spec_full_path_train)
        
        if stage == 'test':
            audio_path_test = os.path.join(self.params.data_dir_root, self.params.test_dir)
            spec_path_test = os.path.join(self.params.spectrogram_dir_root, self.params.spectrogram_dir, self.params.test_dir)
            spec_full_path_test = os.path.join(self.params.spectrogram_dir_root, self.params.spectrogram_full_dir, self.params.test_dir)
            self.test_set = UpsamplerDataset(audio_path_test, spec_path_test, spec_full_path_test)

    def train_dataloader(self):
        return DataLoader(self.train_set, **self.loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_set, **self.loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_set, **self.loader_kwargs)

    def collate(self, minibatch):

        samples_per_frame = self.params.hop_samples

        specs = []
        targets = []

        for spec, target in minibatch:
            
            # Filter out records that aren't long enough.
            assert self.params.crop_mel_frames <= len(spec)
            assert len(spec) * samples_per_frame == len(target) # TODO check the shape of the target

            start = random.randint(0, spec.shape[0] - self.params.crop_mel_frames)
            end = start + self.params.crop_mel_frames
            spec = spec[start:end].T

            start *= samples_per_frame
            end *= samples_per_frame
            target = target[start:end]
            target = np.pad(target, (0, (end-start) - len(target)), mode='constant')

            specs.append(spec)
            targets.append(target)
        
        specs = torch.from_numpy(np.stack(specs))
        targets = torch.from_numpy(np.stack(targets))

        return specs, targets