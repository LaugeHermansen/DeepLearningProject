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

    # data_dir_root = '/dtu/blackhole/11/155505/data', # the root dir of the data files
    # spectrogram_dir_root = 'spectrograms', # the root dir of the spectrogram files
    # # upsampler_target_dir_root = '/dtu/blackhole/11/155505/upsampler_targets', # the root dir of the upsampler target files
    # project_dir_root = '', # the root dir of the project files
    # checkpoint_dir_root = '/dtu/blackhole/11/155505/checkpoints', # the root dir of the checkpoint files
    # results_dir_root = 'results',

    data_dir_root = '/dtu/blackhole/11/155505/audio', # the root dir of the data files
    spectrogram_dir_root = '/dtu/blackhole/11/155505/spectrograms', # the root dir of the spectrogram files
    upsampler_target_dir_root = '/dtu/blackhole/11/155505/upsampler_targets', # the root dir of the upsampler target files
    generated_audio_dir_root = '/dtu/blackhole/11/155505/generated_audio', # the root dir of the generated audio files
    generated_spectrogram_dir_root = '/dtu/blackhole/11/155505/generated_spectrograms', # the root dir of the generated spectrogram files
    project_dir_root = '', # the root dir of the project files
    checkpoint_dir_root = 'experiments', # the root dir of the checkpoint files
    results_dir_root = 'experiments',
    model_evaluator_results_dir = 'model_evaluator_results',


    # audio dir - is relative to data_dir_root
    # train_dir = 'NST_dataset/dataset/train', #relative to data_dir
    # test_dir = 'NST_dataset/dataset/test',  #relative to data_dir

    train_dir = 'NST_dataset/train', #relative to data_dir
    test_dir = 'NST_dataset/test',  #relative to data_dir
    val_dir = 'NST_dataset/val', #relative to data_dir

    # val_dir = None, #relative to data_dir
    # val_size = 0.1, # if val_dir is None, val_size is used to split train_dir into train and val
    
    # spectrogram_full_dir = '',
    # spectrogram_dir = '', # the version of spectrograms to use relative to spectrogram_dir_root

    spectrogram_full_dir = 'full',
    spectrogram_dir = 'full', # the version of spectrograms to use relative to spectrogram_dir_root


    num_workers = 4,
    fp16 = True,
    load_data_to_ram = False,
    accelerator = 'gpu',
    gradient_clip_val = 0,
    checkpoint_name = None,
    downscale = None
)


params = AttrDict(base_params)