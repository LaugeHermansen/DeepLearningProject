from params import params
from speech_datamodule import SpeechDatasetBase, SpeechDataModule
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

class UpsamplerDatasetBase(Dataset):
    def __init__(self, dataset, target_dir: str):
        self.dataset = dataset
        self.audio_dir = dataset.audio_dir
        self.spec_dir = dataset.spec_dir
        self.target_dir = target_dir
        self.audio_file_paths = dataset.audio_file_paths
        self.audio_filenames = dataset.audio_filenames
        self.spec_file_paths = dataset.spec_file_paths
        self.spec_filenames = dataset.spec_filenames
        self.target_filenames = np.array([f'{f}.target.npy' for f in self.audio_filenames])
        self.target_file_paths = np.array([os.path.join(target_dir, f) for f in self.target_filenames])

        self.ignored_files = dataset.ignored_files
        
        for f in self.spec_file_paths:
            mkdir(os.path.dirname(f))
    
    def __len__(self):
        return len(self.audio_file_paths)
    
    def _load_one_item(self, idx):
        spec = np.load(self.spec_file_paths[idx])
        target = np.load(self.target_file_paths[idx])
        return spec, target

    def generate_upsampler_targets(self, spectrogram_upsampler: nn.Module):

        for i in tqdm(range(len(self.dataset)), desc = "generating upsampler data"):
            if not os.path.exists(self.target_file_paths[i]):
                spec = torch.from_numpy(np.load(self.spec_file_paths[i]))
                target = spectrogram_upsampler(spec).detach().cpu().numpy()
                np.save(target, self.target_file_paths[i])


class UpsamplerDatasetDisk(UpsamplerDatasetBase):

    def __getitem__(self, idx):
        return self._load_one_item(idx)

class UpsamplerDatasetRAM(UpsamplerDatasetBase):

        def __init__(self, dataset: SpeechDatasetBase, model: DiffWave):
            super(UpsamplerDatasetRAM).__init__(dataset, model)
            self.specs = []
            self.targets = []
            for i in tqdm(range(len(self)), desc = "loading spectrograms"):
                spec, target = self._load_one_item(i)
                self.specs.append(spec)
                self.targets.append(target)
        
        def __getitem__(self, idx):
            return self.specs[idx], self.targets[idx]


class UpsamplerDataModule(pl.LightningDataModule):
    """
    DataModule for the upsampler
    """
    def __init__(self, params: AttrDict, speech_datamodule: SpeechDataModule):
        super().__init__()
        self.params = params
        self.speech_datamodule = speech_datamodule

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

        if params.load_data_to_ram: self.data_class = UpsamplerDatasetRAM
        else:                       self.data_class = UpsamplerDatasetDisk
    
    def prepare_data(self):
        self.


    def setup(self, stage):
        # Assign Train/val split(s) for use in Dataloaders
        if stage == 'fit':
            self.train_set = self.data_class(self.speech_datamodule.train_set)
            self.val_set   = self.data_class(self.speech_datamodule.val_set  )
        if stage == 'test':
            self.test_set  = self.data_class(self.speech_datamodule.test_set , self.model)

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