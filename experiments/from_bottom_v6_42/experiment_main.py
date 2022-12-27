#%%
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

import torch
import os

from tools import AttrDict, mkdir, Timer

from diffwave_model import DiffWave
from experiment_helpers import fit_model

import shutil
import __main__

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

    data_dir_root = '/dtu/blackhole/11/155505', # the root dir of the data files
    project_dir_root = '', # the root dir of the project files

    train_dir = 'NST_dataset/dataset/train', #relative to data_dir
    test_dir = 'NST_dataset/dataset/test',  #relative to data_dir
    val_dir = None, #relative to data_dir
    val_size = 0.1, # if val_dir is None, val_size is used to split train_dir into train and val
    num_workers = 4,
    fp16 = True,
    load_data_to_ram = False,
    accelerator = 'gpu',
    gradient_clip_val = 0,
)


params = AttrDict(base_params)



#%%



def main():


    #########################################################################
    # -------------- specify the options for the eperiment ------------------

    experiment_name = 'from_bottom_v6'
    global_seed = 42
    max_epochs = 100000
    
    params.gradient_clip_val = 10.

    # load the model somehow
    model = DiffWave(params, measure_grad_norm=True)

    # ----------------- don't change anything below this line ---------------
    #########################################################################

    save_dir = os.path.join(params.project_dir_root, 'experiments', f'{experiment_name}_{global_seed}')
    mkdir(save_dir)

    shutil.copy(__main__.__file__, os.path.join(save_dir, os.path.basename(__main__.__file__)))
    
    fit_model(model, params, experiment_name, global_seed, max_epochs)
    

if __name__ == '__main__':
    main()
