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
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from speech_datamodule import SpeechDataModule
from diffwave_model import DiffWave


from datetime import timedelta
from tools import mkdir, Timer

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


def get_trainer(params, exp_name, global_seed, max_epochs):

    save_dir = os.path.join(params.project_dir_root, 'experiments', f'{exp_name}_{global_seed}')
    mkdir(save_dir)

    # store grad norm
    store_grad_norm_callback = StoreGradNormCallback()

    # save model every 1 hour
    checkpoint_callback_time = ModelCheckpoint(
        dirpath=save_dir,
        filename='time-{epoch}-{val_loss:.6f}',
        # train_time_interval=timedelta(hours=1),
        # train_time_interval=timedelta(hours=1),
        every_n_epochs=15,
        save_top_k=-1,
        )
    
    # save k best end of epoch models
    checkpoint_callback_top_k = ModelCheckpoint(
        dirpath=save_dir,
        filename='k-{epoch}-{val_loss:.6f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
        )
    
    # create logger
    logger = CSVLogger(
        save_dir=save_dir,
        name='log',
        flush_logs_every_n_steps=10
        )
    
    if params.accelerator == 'gpu':
        assert torch.cuda.is_available(), "CUDA is not available."


    trainer = pl.Trainer(
        callbacks=[checkpoint_callback_time, checkpoint_callback_top_k, store_grad_norm_callback], # runs at the end of every train loop
        log_every_n_steps=10,
        max_epochs=max_epochs,
        accelerator=params.accelerator,
        # devices=1,
        logger=logger,
        gradient_clip_val=params.gradient_clip_val,
        # track_grad_norm=2,
        ckpt_path=os.path.join(save_dir, params.checkpoint_name),
    )

    return trainer


def fit_model(model: DiffWave, params, exp_name, global_seed, max_epochs, use_timing=False):
    
    timer_experiment_helpers = Timer(use_timing)
    model.use_timing(use_timing)
    pl.seed_everything(global_seed, workers=True)
    timer_experiment_helpers("preprocessing data")
    data = SpeechDataModule(params=params, use_timing=use_timing)
    timer_experiment_helpers()
    trainer = get_trainer(params, exp_name, global_seed, max_epochs)
    timer_experiment_helpers("fitting model")
    trainer.fit(model, data)
    timer_experiment_helpers()
    update_gitignore(params)

    #return timers
    return data.val_set.timer, data.train_set.timer, timer_experiment_helpers, model.timer
    



if __name__ == "__main__":
    from experiment_main import params
    update_gitignore(params)
