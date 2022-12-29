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

    experiment_name = 'from_bottom_v8'
    global_seed = 42
    max_epochs = 100000
    
    params.gradient_clip_val = 100.
    params.checkpoint_name = "time-epoch=239-val_loss=0.042926.ckpt"
    # params.load_data_to_ram = True

    # load the model somehow
    model = DiffWave(params, measure_grad_norm=True)

    # ----------------- don't change anything below this line ---------------
    #########################################################################

    save_dir = os.path.join(params.project_dir_root, 'experiments', f'{experiment_name}_{global_seed}')
    mkdir(save_dir)
    exp_script_dir = os.path.join(save_dir, 'experiment_scripts')
    mkdir(exp_script_dir)
    
    # save the experiment script
    script_names = glob(os.path.join(exp_script_dir, 'run_script_version*.py'))
    versions = [int(name.split('run_script_version')[-1].split('.')[0]) for name in script_names]
    if len(versions) == 0:
        version = 0
    else:
        version = max(versions) + 1
    shutil.copy(__main__.__file__, os.path.join(exp_script_dir, f'run_script_version{version}.py'))

    fit_model(model, params, experiment_name, global_seed, max_epochs)
    

#%%

if __name__ == '__main__':
    main()


