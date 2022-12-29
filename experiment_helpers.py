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


import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

from speech_datamodule import SpeechDataModule
from diffwave_model import DiffWave
from glob import glob

from datetime import timedelta
from tools import mkdir, Timer
import numpy as np
import shutil

#%%

class StoreGradNormCallback(pl.Callback):

    def on_before_optimizer_step(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer, opt_idx: int):
        
        def compute_grad_norm(model):
            if model.measure_grad_norm:
                grad_norm = 0.0
                for p in list(filter(lambda p: p.grad is not None, model.parameters())):
                    grad_norm += torch.linalg.norm(p.grad.data).detach()**2
            return grad_norm**0.5

        pl_module.log('grad_2_norm', compute_grad_norm(pl_module), on_step=True, on_epoch=True, prog_bar=True, logger=True)


def update_gitignore(params):
    
    with open(".gitignore", "r") as g:
        content = g.read().split("\n")
    with open(".gitignore", "a") as g:
        root = params.project_dir_root
        names = [params[name] for name in ["train_dir", "test_dir", "val_dir"] if not (params[name] is None)]
        names += ["spectrograms"]
        for name in names:
            path = os.path.join(root, name).replace("\\", "/") + "/**"
            if not path in content:
                g.write("\n" + path)
                print(f'added to .gitignore: \"{path}\"')


def get_trainer(params, max_epochs, checkpoint_dir, results_dir):

    # store grad norm
    store_grad_norm_callback = StoreGradNormCallback()

    #costum proqress bar
    progress_bar = TQDMProgressBar(refresh_rate=100)

    # save model every 1 hour
    checkpoint_callback_time = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='time-{epoch}-{val_loss:.6f}',
        every_n_epochs=15,
        save_top_k=-1,
        )
    
    # save k best end of epoch models
    checkpoint_callback_top_k = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='k-{epoch}-{val_loss:.6f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
        )
    
    # create logger
    logger = CSVLogger(
        save_dir=results_dir,
        name='log',
        flush_logs_every_n_steps=10
        )
    

    if params.accelerator == 'gpu':
        assert torch.cuda.is_available(), "CUDA is not available."


    trainer = pl.Trainer(
        callbacks=[checkpoint_callback_time, checkpoint_callback_top_k, store_grad_norm_callback, progress_bar], # runs at the end of every train loop
        log_every_n_steps=10,
        max_epochs=max_epochs,
        accelerator=params.accelerator,
        # devices=1,
        logger=logger,
        gradient_clip_val=params.gradient_clip_val,
        # track_grad_norm=2,
    )

    return trainer


def fit_model(model: DiffWave, params, experiment_name, global_seed, max_epochs, main_file_path, use_timing=False):

    checkpoint_dir = os.path.join(params.project_dir_root, params.checkpoint_dir_root, f'{experiment_name}_{global_seed}')
    mkdir(checkpoint_dir)
    results_dir = os.path.join(params.project_dir_root, params.results_dir_root, f'{experiment_name}_{global_seed}')
    mkdir(results_dir)
    exp_script_dir = os.path.join(results_dir, 'experiment_scripts')
    mkdir(exp_script_dir)
    
    # save the experiment script
    script_names = glob(os.path.join(exp_script_dir, 'run_script_version*.py'))
    versions = [int(name.split('run_script_version')[-1].split('.')[0]) for name in script_names]
    if len(versions) == 0: version = 0
    else: version = max(versions) + 1
    shutil.copy(main_file_path, os.path.join(exp_script_dir, f'run_script_version{version}.py'))


    timer_experiment_helpers = Timer(use_timing)
    model.use_timing(use_timing)
    pl.seed_everything(global_seed, workers=True)
    timer_experiment_helpers("preprocessing data")
    data = SpeechDataModule(params=params, use_timing=use_timing)
    timer_experiment_helpers()
    trainer = get_trainer(params, max_epochs, checkpoint_dir, results_dir)
    timer_experiment_helpers("fitting model")
    ckpt_path = os.path.join(checkpoint_dir, params.checkpoint_name) if params.checkpoint_name is not None else None
    if ckpt_path is not None:
        assert os.path.exists(ckpt_path), f"checkpoint path {ckpt_path} does not exist"
    trainer.fit(model, data, ckpt_path=ckpt_path,)
    timer_experiment_helpers()
    update_gitignore(params)

    return data.val_set.timer, data.train_set.timer, timer_experiment_helpers, model.timer
    



if __name__ == "__main__":
    from experiment_main import params
    update_gitignore(params)
