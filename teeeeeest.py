#%%

import numpy as np
import os
from tqdm import tqdm

from speech_datamodule import SpeechDataset, SpeechDataModule
from params import params

# audio_file_paths = np.load("val_files.npy")

# for f in tqdm(audio_file_paths):
#     # move file to validation folder
#     # source = f.replace('/data/', '/audio/').replace('/dataset/','/')
#     source = f.replace('/data/', '/spectrograms/').replace('/dataset/','/')
#     target = source.replace('/train/', '/val/')
#     # assert os.path.exists(source)
#     os.renames(source, target)

data = SpeechDataModule(params)

data.setup('fit')


