# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import numpy as np
import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler

import torchaudio
from torchaudio.transforms import MelSpectrogram

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from math import sqrt, floor
from glob import glob
from tqdm import tqdm
from datetime import timedelta



base_params = dict(
    # Training params
    batch_size=16,
    learning_rate=2e-4,
    max_grad_norm=None,

    # Data params
    sample_rate=22050,
    n_mels=80,
    n_fft=1024,
    hop_samples=256,
    crop_mel_frames=62,  # Probably an error in paper.

    # Model params
    residual_layers=30,
    residual_channels=64,
    dilation_cycle_length=10,
    unconditional = False,
    noise_schedule=np.linspace(1e-4, 0.05, 50).tolist(),
    inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],

    # unconditional sample len
    audio_len = 22050*5, # unconditional_synthesis_samples
    
    # own params
    data_dir = 'LJSpeech', # used only to preprocess, has to be parent to all other
    num_workers = 4,
    fp16 = True
)


class AttrDict(dict):
  def __init__(self, *args, **kwargs):
      super(AttrDict, self).__init__(*args, **kwargs)
      self.__dict__ = self

  def override(self, attrs):
    if isinstance(attrs, dict):
      self.__dict__.update(**attrs)
    elif isinstance(attrs, (list, tuple, set)):
      for attr in attrs:
        self.override(attr)
    elif attrs is not None:
      raise NotImplementedError
    return self


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
        self.projection1 = nn.Linear(128, 512)
        self.projection2 = nn.Linear(512, 512)

    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1) # [T,1]
        dims = torch.arange(64).unsqueeze(0)         # [1,64]
        table = steps * 10.0**(dims * 4.0 / 63.0)    # [T,64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class SpectrogramUpsampler(pl.LightningModule):
    def __init__(self, n_mels, downscale=None, params=None):
        super().__init__()
        
        if params is not None:
            self.params = params
        
        self.loss_fn = nn.MSELoss()
        self.use_downscale = downscale is not None # bool
        if self.use_downscale:
            scale = int(1 / downscale) # e.g.: 1, 2, 4, 8
            assert scale in [1, 2, 4, 8]
            self.conv_downscale = nn.ConvTranspose2d(1, 1, [2 * scale, 2 * scale], stride=[scale, scale], padding=[int(scale // 2), int(scale // 2)])
        self.conv1 = nn.ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])
        self.conv2 = nn.ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])

        self.save_hyperparameters()
        
    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        if self.use_downscale:
            x = self.conv_downscale(x)
            x = F.leaky_relu(x, 0.4)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.4)
        x = torch.squeeze(x, 1)
        return x
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.params.learning_rate)
    
    def process_batch(self, batch):
        
        downscaled, original = batch['downscaled'].to(self.device), batch['original'].to(self.device)
        
        x = self(downscaled)
        y = pretrained_model.spectrogram_upsampler(original).detach()
        loss = self.loss_fn(x, y)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


class ResidualBlock(nn.Module):
    def __init__(self, n_mels, residual_channels, dilation):
        '''
        :param n_mels: inplanes of conv1x1 for spectrogram conditional
        :param residual_channels: audio conv
        :param dilation: audio conv dilation
        :param uncond: disable spectrogram conditional
        '''
        super().__init__()
        
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = nn.Linear(512, residual_channels)
        
        self.conditioner_projection = Conv1d(n_mels, 2 * residual_channels, 1)

        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, diffusion_step, conditioner):

        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = x + diffusion_step
            
        conditioner = self.conditioner_projection(conditioner)
        y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class DiffWave(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        
        # saveconda install -c conda-forge tensorflow hyperparams for load
        self.save_hyperparameters()
        self.params = params
        
        self.diffusion_embedding = DiffusionEmbedding(len(params.noise_schedule))
        self.spectrogram_upsampler = SpectrogramUpsampler(params.n_mels)
        self.residual_layers = nn.ModuleList([
            ResidualBlock(params.n_mels, params.residual_channels, 2**(i % params.dilation_cycle_length))
            for i in range(params.residual_layers)
        ])
        self.input_projection = Conv1d(1, params.residual_channels, 1)
        self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)
        self.output_projection = Conv1d(params.residual_channels, 1, 1)
        
        # fill the input with zeros
        nn.init.zeros_(self.output_projection.weight)
        
        # train params
        self.loss_fn = nn.L1Loss()
        beta = np.array(self.params.noise_schedule)
        noise_level = np.cumprod(1 - beta)
        self.noise_level = torch.tensor(noise_level.astype(np.float32))
        self.autocast = torch.cuda.amp.autocast(enabled=params.get('fp16', False))

    def forward(self, audio, diffusion_step, spectrogram):
        x = audio.unsqueeze(1)
        x = self.input_projection(x)
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)
        
        spectrogram = self.spectrogram_upsampler(spectrogram)

        skip = torch.zeros_like(x)
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_step, spectrogram)
            skip += skip_connection

        x = skip / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x
    
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.params.learning_rate)
    
    def process_batch(self, batch):
        # for param in self.parameters():
        #     param.grad = None
            
        audio, spectrogram = batch['audio'], batch['spectrogram']

        N, T = audio.shape
        device = audio.device
        self.noise_level = self.noise_level.to(device)

        with self.autocast:
            t = torch.randint(0, len(self.params.noise_schedule), [N], device=audio.device)
            noise_scale = self.noise_level[t].unsqueeze(1)
            noise_scale_sqrt = torch.sqrt(noise_scale)
            noise = torch.randn_like(audio)
            noisy_audio = noise_scale_sqrt * audio + torch.sqrt(1.0 - noise_scale) * noise

            predicted = self(noisy_audio, t, spectrogram)
            loss = self.loss_fn(noise, predicted.squeeze(1))
        
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        with torch.no_grad():
            # Change in notation from the DiffWave paper for fast sampling.
            # DiffWave paper -> Implementation below
            # --------------------------------------
            # alpha -> talpha
            # beta -> training_noise_schedule
            # gamma -> alpha
            # eta -> beta
            
            spectrogram = batch['spectrogram']
            
            # get device
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
            
            training_noise_schedule = np.array(self.params.noise_schedule)
            inference_noise_schedule = np.array(self.params.inference_noise_schedule)

            talpha = 1 - training_noise_schedule
            talpha_cum = np.cumprod(talpha)

            beta = inference_noise_schedule
            alpha = 1 - beta
            alpha_cum = np.cumprod(alpha)

            T = []
            for s in range(len(inference_noise_schedule)):
                for t in range(len(training_noise_schedule) - 1):
                    if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
                        twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
                        T.append(t + twiddle)
                        break
            T = torch.tensor(T, dtype=torch.float).unsqueeze(1) # to tensor and create batch dim


            if len(spectrogram.shape) == 2: # Expand rank 2 tensors by adding a batch dimension.
                spectrogram = spectrogram.unsqueeze(0)
            spectrogram = spectrogram.to(device)
            audio = torch.randn(spectrogram.shape[0], self.params.hop_samples * spectrogram.shape[-1], device=device)
            
            # noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device)
            
            for n in range(len(alpha) - 1, -1, -1):
                c1 = 1 / alpha[n]**0.5
                c2 = beta[n] / (1 - alpha_cum[n])**0.5
                audio = c1 * (audio - c2 * self(audio, T[n], spectrogram).squeeze(1))
                if n > 0:
                    noise = torch.randn_like(audio)
                    sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
                    audio += sigma * noise
                audio = torch.clamp(audio, -1.0, 1.0)
                
        return audio
        

def get_pretrained_model(model_path, params):
    
    use_cuda = torch.cuda.is_available()
    # get device
    device = torch.device('cuda:0') if use_cuda else torch.device('cpu')

    # init model
    model = DiffWave(params).to(device)
    
    #checkpoint = torch.load(model_path, map_location=None if use_cuda else torch.device('cpu'))
    #model.load_state_dict(checkpoint['model'])
    model.load_from_checkpoint(model_path)
    
    model.eval()
    return model


class SpeechDataset(Dataset):
    def __init__(self, filenames, params):
        super().__init__()
        self.filenames = filenames
        self.params = params
            
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        audio_filename = self.filenames[idx]
        spec_filename = f'{audio_filename}.spec.npy'
        signal, _ = torchaudio.load(audio_filename)
        spectrogram = np.load(spec_filename).T
        return {
            'audio': signal[0],
            'spectrogram': spectrogram
        }


class SpeechDataModule(pl.LightningDataModule):
    def __init__(self, params, train_path, val_path=None, test_path=None):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
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
    
    def prepare_data(self):
        filenames = glob(f'{self.params.data_dir}/**/*.wav', recursive=True)
        
        mel_args = {
                'sample_rate': self.params.sample_rate,
                'win_length': self.params.hop_samples * 4,
                'hop_length': self.params.hop_samples,
                'n_fft': self.params.n_fft,
                'f_min': 20.0,
                'f_max': self.params.sample_rate / 2.0,
                'n_mels': self.params.n_mels,
                'power': 1.0,
                'normalized': True,
        }
        
        mel_spec_transform = MelSpectrogram(**mel_args)
        
        for filename in tqdm(filenames, desc='Preprocessing'):
            
            if os.path.exists(f'{filename}.spec.npy'):
                continue
            
            audio, sr = torchaudio.load(filename)
            
            #if audio.shape[0] != 1:
            #    raise ValueError(f'Audio signal does not seem to be mono, channels = {audio.shape[0]}.')
            
            audio = torch.clamp(audio[0], -1.0, 1.0)

            if self.params.sample_rate != sr:
                raise ValueError(f'Invalid sample rate {sr} != {self.params.sample_rat} (True).')

            with torch.no_grad():
                spectrogram = mel_spec_transform(audio)
                
                if spectrogram.shape[1] < self.params.crop_mel_frames:
                    continue
                
                spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
                spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
                np.save(f'{filename}.spec.npy', spectrogram.cpu().numpy())
    
    def setup(self, stage):
        # Assign Train/val split(s) for use in Dataloaders
        if stage == 'fit':
            if self.val_path is None:
                train_files = glob(f'{self.train_path}/**/*.wav.spec.npy', recursive=True)
                train_files = [f[:-9] for f in train_files] # removes .spec.npy
                all_speech = SpeechDataset(train_files, self.params)
                len_val_set = int(floor(len(train_files) * 0.1))
                self.train_set, self.val_set = random_split(all_speech, [len(train_files)-len_val_set, len_val_set], generator=torch.Generator().manual_seed(42)) # altid samme split
            else:
                train_files = glob(f'{self.train_path}/**/*.wav.spec.npy', recursive=True)
                val_files = glob(f'{self.val_path}/**/*.wav.spec.npy', recursive=True)
                train_files = [f[:-9] for f in train_files] # removes .spec.npy
                val_files = [f[:-9] for f in val_files] # removes .spec.npy
                self.train_set = SpeechDataset(train_files, self.params)
                self.val_set = SpeechDataset(val_files, self.params)
            
        if stage == 'test':
            test_files = glob(f'{self.test_path}/**/*.wav.spec.npy', recursive=True)
            test_files = [f[:-9] for f in test_files] # removes .spec.npy
            self.test_set = SpeechDataset(val_files, self.params)
    
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


class SpectrogramDataset(Dataset):
    def __init__(self, state, downscale):
        super().__init__()
        self.state = state
        self.downscale = downscale
        
        self.downscale_dir = f'data/spectrograms/downscale_{downscale}/{state}/'
        self.original_dir = f'data/spectrograms/original/{state}/'
        
        filenames = glob(f'{self.original_dir}/**/*.spec.npy', recursive=True)
        self.filenames = [f[len(self.original_dir):] for f in filenames]
            
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        downscaled_path = self.downscale_dir + self.filenames[idx]
        original_path = self.original_dir + self.filenames[idx]
        downscaled = torch.from_numpy(np.load(downscaled_path).T)
        original = torch.from_numpy(np.load(original_path).T)
        return {
            'downscaled': downscaled,
            'original': original
        }


class SpectrogramDataModule(pl.LightningDataModule):
    def __init__(self, params, data_dir, downscale):
        super().__init__()
        self.data_dir = data_dir
        self.downscale = downscale
        self.scale = int(1 // downscale)
        assert self.scale in [1, 2, 4, 8, 16, 32]
        self.params = params
        
        # this is for improving speed
        self.num_workers = self.params.get('num_workers', os.cpu_count())
        
        # this is for the dataloader
        self.loader_kwargs = {
            'batch_size': params.batch_size,
            'num_workers': self.num_workers,
            'collate_fn': self.collate,
            'pin_memory': True,
            'drop_last': True
        }
  
    def setup(self, stage):
        # Assign Train/val split(s) for use in Dataloaders
        if stage == 'fit':
            
            self.train_set = SpectrogramDataset('train', self.downscale)
            self.val_set = SpectrogramDataset('val', self.downscale)
            
        if stage == 'test':
            
            self.test_set = SpectrogramDataset('test', self.downscale)
    
    
    def collate(self, minibatch):
        
        downscale_lenth = int(self.params.crop_mel_frames * self.downscale)
        buffer = self.scale if self.scale > 1 else 0
        for record in minibatch:
            
            # Filter out records that aren't long enough.
            assert downscale_lenth <= len(record['downscaled']) + buffer

            start = random.randint(0, record['downscaled'].shape[0] - (downscale_lenth + buffer))
            end = start + downscale_lenth
            record['downscaled'] = record['downscaled'][start:end].T

            start *= self.scale
            end = start + self.scale * downscale_lenth
            record['original'] = record['original'][start:end].T
        

        downscaled = np.stack([record['downscaled'] for record in minibatch])
        original = np.stack([record['original'] for record in minibatch])
        
        return {
            'downscaled': torch.from_numpy(downscaled),
            'original': torch.from_numpy(original),
        }
    
    def train_dataloader(self):
        return DataLoader(self.train_set, shuffle=True, **self.loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_set, shuffle=False, **self.loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_set, shuffle=False, **self.loader_kwargs)


def prepare_data(params, data_dir, savedir, downscale=None):
    filenames = glob(f'{data_dir}/**/*.wav', recursive=True)
        
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

    if downscale is not None:
        mel_args['win_length'] = int(mel_args['win_length'] / downscale)
        mel_args['hop_length'] = int(mel_args['hop_length'] / downscale)
        mel_args['n_fft'] = int(mel_args['n_fft'] / downscale)
        mel_args['n_mels'] = int(mel_args['n_mels'] * downscale)
    
    mel_spec_transform = MelSpectrogram(**mel_args)
    
    for filename in tqdm(filenames, desc='Preprocessing'):
        
        save_path = f'{savedir}/{filename[len(data_dir):]}.spec.npy'
        
        if os.path.exists(save_path):
            continue
        
        audio, sr = torchaudio.load(filename)
        
        #if audio.shape[0] != 1:
        #    raise ValueError(f'Audio signal does not seem to be mono, channels = {audio.shape[0]}.')
        
        audio = torch.clamp(audio[0], -1.0, 1.0)

        if params.sample_rate != sr:
            raise ValueError(f'Invalid sample rate {sr} != {params.sample_rat} (True).')

        with torch.no_grad():
            spectrogram = mel_spec_transform(audio)
            
            if spectrogram.shape[1] < params.crop_mel_frames:
                continue
            
            spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
            spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
            save_folder = os.path.dirname(save_path)
            # create folder if not exists
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            np.save(save_path, spectrogram.cpu().numpy())


def create_conditions(model, data_dir, save_dir):
    
    with torch.no_grad():
    
        for file in tqdm(glob(f'{data_dir}/**/*.spec.npy', recursive=True), desc='Creating conditions'):
            # load spectrogram
            spectrogram = np.load(file)
            
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
            
            spectrogram = torch.from_numpy(spectrogram).unsqueeze(0).to(device)
            
            # get conditions
            conditions = model.spectrogram_upsampler(spectrogram)
            
            # save conditions
            save_path = f'{save_dir}/{file[len(data_dir):]}'
            save_folder = os.path.dirname(save_path)
            # create folder if not exists
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            np.save(save_path, conditions.cpu().numpy())
    



def main(model_path, params, data_search_string, downscale):
    
    # model = get_pretrained_model(model_path=model_path, params=params)
    
    upsampler_model = SpectrogramUpsampler(None, downscale=downscale, params=params)
    
    # data_paths = glob(data_search_string, recursive=True)
    # data_paths = [path[len(data_search_string):] for path in data_paths]
    for downscale in [0.5, 0.25]:
       prepare_data(params, data_search_string, f"data/spectrograms/downscale_{downscale}", downscale)
    # create_conditions(model, "data/spectrograms/original", "data/conditions")
    
    # upsampler training 
    data = SpectrogramDataModule(params=params,
                            data_dir='data/spectrograms/',
                            downscale=downscale
                            )
    
    save_dir = f'upsamplers/{downscale}'
    
    # save k best end of epoch models
    checkpoint_callback_top_k = ModelCheckpoint(
    	dirpath=save_dir,
    	filename='best-{epoch}-{val_loss:.6f}',
    	save_top_k=5,
    	monitor='val_loss',
    	mode='min'
    	)
    
    logger = CSVLogger(
        save_dir=save_dir,
        name='log',
        flush_logs_every_n_steps=10
        )
    
    use_cuda = torch.cuda.is_available()
    accelerator = 'gpu' if use_cuda else 'cpu'
    
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback_top_k], # runs at the end of every train loop
        log_every_n_steps=10,
        max_epochs=2,
        accelerator=accelerator,
        # devices=1,
        logger=logger,
    )
    
    trainer.fit(upsampler_model, data)
    

if __name__ == '__main__':
    params = AttrDict(base_params)
    
    model_path = './k-epoch=53-val_loss=0.037580.ckpt' # get pytorch model in directory
    data_search_string = 10 * '../' + 'dtu/blackhole/11/155505/audio/NST_dataset/'
    
    pretrained_model = get_pretrained_model(model_path=model_path, params=params)
    
    for downscale in [1]:
        main(model_path, params, data_search_string, downscale)

