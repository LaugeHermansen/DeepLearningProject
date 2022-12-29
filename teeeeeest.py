#%%

import numpy as np
import os

audio_file_paths = np.load("val_files.npy")


for f in audio_file_paths:
    # move file to validation folder
    target = f.replace('train', 'val')
    os.renames(f, target)
