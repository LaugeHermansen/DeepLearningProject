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
from glob import glob

from params import params

#%%

def main():

    #########################################################################
    # -------------- specify the options for the experiment -----------------

    zoom = 0.05
    
    experiment_name = f'from_bottom_zoom_{zoom}'
    global_seed = 42
    max_epochs = 100000
    # checkpoint_name = "k-epoch=53-val_loss=0.037580.ckpt"

    #params.learning_rate = 5e-5
    params.gradient_clip_val = 80.
    # params.load_data_to_ram = True
    params.spectrogram_dir = str(zoom)
    
    assert os.path.exists(os.path.join(params.spectrogram_dir_root, params.spectrogram_dir)), "spectrogram dir root does not exist"


    # ckpt_path = os.path.join(params.project_dir_root, params.checkpoint_dir_root, f'{experiment_name}_{global_seed}', checkpoint_name)


    # checkpoint = torch.load(ckpt_path)#, map_location=None if use_cuda else torch.device('cpu'))
    model = DiffWave(params, measure_grad_norm=True)
    # model.load_state_dict(checkpoint['state_dict'])
    
    # load the model somehow
    
    

    # ----------------- don't change anything below this line ---------------
    #########################################################################

    src = str(__main__.__file__)

    fit_model(model, params, experiment_name, global_seed, max_epochs, src)
    

#%%

if __name__ == '__main__':
    main()


