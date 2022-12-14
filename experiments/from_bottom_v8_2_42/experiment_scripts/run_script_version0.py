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

from params import params

#%%

def main():

    #########################################################################
    # -------------- specify the options for the eperiment ------------------

    experiment_name = 'from_bottom_v8'
    global_seed = 42
    max_epochs = 100000
    
    params.gradient_clip_val = 100.
    # params.checkpoint_name = "k-epoch=250-val_loss=0.041077.ckpt"
    params.load_data_to_ram = True

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

