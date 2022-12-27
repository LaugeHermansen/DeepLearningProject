import numpy as np
from tools import AttrDict

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
    checkpoint_name = None,
)


params = AttrDict(base_params)