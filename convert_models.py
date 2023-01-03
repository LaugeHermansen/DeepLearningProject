import torch
import torch.nn as nn
from params import params as params_lauge


from diffwave_model import DiffWave as DiffWaveLauge  
from Christians_seje_hpc_scripts.experiment_from_bottom import DiffWave as DiffWaveChristian, base_params, AttrDict

model_names = ["models_for_comparison/k-epoch=53-val_loss=0.037580.ckpt",
             "models_for_comparison/time-epoch=209-val_loss=0.041662.ckpt",]

params_christian = AttrDict(base_params)

models_lauge = [DiffWaveLauge.load_from_checkpoint(path) for path in model_names]
models_christian = [DiffWaveChristian(params_christian) for path in model_names]

for model_lauge, model_christian, name in zip(models_lauge, models_christian, model_names):
    model_christian.load_state_dict(model_lauge.state_dict())
    torch.save(model_christian.state_dict(), name[:-5] + "_christian.pt")





