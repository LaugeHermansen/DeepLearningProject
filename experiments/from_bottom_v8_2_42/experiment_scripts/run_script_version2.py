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

    experiment_name = 'from_bottom_v8_2'
    global_seed = 42
    max_epochs = 100000

    params.learning_rate = 5e-5
    params.gradient_clip_val = 80.
    params.checkpoint_name = "time-epoch=479-val_loss=0.040237.ckpt"
    # params.load_data_to_ram = True

    # load the model somehow
    model = DiffWave(params, measure_grad_norm=True)

    # ----------------- don't change anything below this line ---------------
    #########################################################################

    src = str(__main__.__file__)

    fit_model(model, params, experiment_name, global_seed, max_epochs, src)
    

#%%

if __name__ == '__main__':
    main()


