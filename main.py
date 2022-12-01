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
from pytorch_lightning.loggers import TensorBoardLogger

from math import sqrt
from glob import glob
from tqdm import tqdm


# specify path to root

root_directory = r'D:\DTU\DeepLearningProject'


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
    crop_mel_frames=62,  # TODO: Probably an error in paper.

    # Model params
    residual_layers=30, # N
    residual_channels=64, # C
    dilation_cycle_length=10, # n
    unconditional = False, #TODO remove this?
    noise_schedule=np.linspace(1e-4, 0.05, 50).tolist(), # beta
    inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],

    # unconditional sample len / min audio length of conditional 
    audio_len = 22050*5, # unconditional_synthesis_samples
    
    # own params
    data_dir = f'{root_directory}/data/LJSpeech-1.1/wavs', # used only to preprocess, has to be parent to all other
    # num_workers = 0,
    fp16 = True
)


class AttrDict(dict):
    """
    helper class for convenience - no methematical 
    """
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
    """
    Costum conv 1d - with Kaiming normal initialized weights and biases
    nothing new here either
    """
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer



class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps):
        """
        embedding the diffusion step as input to the model
        as described in section 3.1 + figure 2 in diffwave paper

        parameters
        --------
        * max_steps: int, T from the paper - maximal numbe rof diffusion steps

        methods
        --------
        * __init__: initialize the embeddings (eq 8) for all diffusion steps
        * forward: get embedding at given diffusion step
        * _build_embedding: create the sin,cos vector
        * _lerp_embedding: linearly interpolate embeddings in case of non integer diff step.

        """
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
        self.projection1 = nn.Linear(128, 512)
        self.projection2 = nn.Linear(512, 512)

    def forward(self, diffusion_step):
        """
        return embedding at diffusion step: diffusion_step.
        """

        #if diffusion_step is integer, yield embedding
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        
        #otherwize linearly interpolate
        else:
            x = self._lerp_embedding(diffusion_step)

        # 2FC layers according to figure 2
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _lerp_embedding(self, t):
        """
        used in fast sampling
        
        linearly interpolate embedding in case of non-integer diffusion step: t.
        """
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_steps):
        """
        create the embedding as described in section 3.1, eq(8)
        """
        steps = torch.arange(max_steps).unsqueeze(1) # [T,1]
        dims = torch.arange(64).unsqueeze(0)         # [1,64]
        table = steps * 10.0**(dims * 4.0 / 63.0)    # [T,64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class SpectrogramUpsampler(nn.Module):
    def __init__(self, n_mels):
        """
        upsampler to make spectogram as long as waveform using transposed 2d conv - section 3.2
        """
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])
        self.conv2 = nn.ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.4)
        x = torch.squeeze(x, 1)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, n_mels, residual_channels, dilation):
        '''
        One residual block as described in figure 2, and also in the appendix C

        parameters
        ---------
        * n_mels: bins in the mel spectrogram (dimensionality of spectrogram on first axis)
        * residual_channels: C as described in he paper as well
        * dilation: dilation of the 1d convolution - see paper as well.
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
        """
        Colect it all, to build the diffwave model

        parameters
        ------------
        params: attribute dict as in the top of this file
        """
        super().__init__()
        
        # save hyperparams for load
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


        #TODO: what does this do?
        nn.init.zeros_(self.output_projection.weight)
        
        # train params
        #TODO: why use L1-loss. it says MSE loss in the paper (eq 7) doesn't it?
        self.loss_fn = nn.L1Loss()
        beta = np.array(self.params.noise_schedule)
        noise_level = np.cumprod(1 - beta)
        self.noise_level = torch.tensor(noise_level.astype(np.float32))
        self.autocast = torch.cuda.amp.autocast(enabled=params.get('fp16', False))

    def forward(self, audio, diffusion_step, spectrogram):
        """
        predict noise at diffusion step given noisy audio while conditioning on spectrogram
        """
        x = audio.unsqueeze(1)
        x = self.input_projection(x)
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)
        
        spectrogram = self.spectrogram_upsampler(spectrogram)

        skip = torch.zeros_like(x)
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_step, spectrogram)
            skip = skip + skip_connection

        x = skip / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.params.learning_rate)
    
    def process_batch(self, batch):
        """
        compute loss as described in eq 6 - for the given batch.

        """

        #TODO remove?
        # for param in self.parameters():
        #     param.grad = None
            
        audio, spectrogram = batch['audio'], batch['spectrogram']

        N, T = audio.shape

        #TODO remove?
        # device = audio.device
        # self.noise_level = self.noise_level.to(device)

        with self.autocast:
            t = torch.randint(0, len(self.params.noise_schedule), [N]) # , device=audio.device TODO remove?
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
        """
        generate sound conditional on spectrogram (given in batch)
        fast sampling (inference) as described in appendix b
        """
        with torch.no_grad():
            # Change in notation from the DiffWave paper for fast sampling.
            # DiffWave paper -> Implementation below
            # --------------------------------------
            # alpha -> talpha
            # beta -> training_noise_schedule
            # gamma -> alpha
            # eta -> beta
            
            spectrogram = batch['spectrogram']
            
            #TODO avoid sending to device in pytorch lightning modules

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
            
            # noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device) #TODO what is this?
            
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
    
    # get model params, has to specify it to cpu if cuda not available
    checkpoint = torch.load(model_path, map_location=None if use_cuda else torch.device('cpu'))
    
    # load model
    model = DiffWave(params).to(device)
    model.load_state_dict(checkpoint['model'])
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
        #TODO sample rate?
        return {
            'audio': signal.squeeze(0),
            'spectrogram': spectrogram
        }


# TODO: I don't understand this class.
# why both a validation, and train, and test path?
# why not create a sampler function or whatever it is, that samples from the data
# with this structure, we need separate folders for train/test/validation data?

class SpeechDataModule(pl.LightningDataModule):
    def __init__(self, params, train_path, val_path=None, test_path=None):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.params = params
        
        # this is for improving speed

        #TODO according to pyorch lightning docs using num_workers in dataloader is a baaad idea <3 use DDP instead
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
                self.train_set, self.val_set = random_split(all_speech, [0.9, 0.1], generator=torch.Generator().manual_seed(42)) # altid samme split
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
        return DataLoader((dataset:=self.test_set), shuffle=False, **self.loader_kwargs(dataset))


def main(model_path, params):
    pl.seed_everything(42, workers=True)
    
    data = SpeechDataModule(params=params,
                            train_path=params.data_dir,
                            val_path=None,
                            test_path=None
                            )
    
    model = get_pretrained_model(model_path=model_path, params=params)
    
    
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode='min')
    
    out_name = 'english'
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        log_every_n_steps=5,
        max_epochs=1,
        accelerator='cpu',
        # devices=1,
        logger=TensorBoardLogger('conditional', name=out_name),
    )
    
    trainer.fit(model, data)


if __name__ == '__main__':
    params = AttrDict(base_params)
    model_path = glob('./*.pt')[0] # get pytorch model in directory
    main(model_path, params)
    