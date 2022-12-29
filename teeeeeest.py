#%%

import numpy as np
import os

audio_file_paths = np.load("val_files.npy")

for f in audio_file_paths:
    # move file to validation folder
    source = f.replace('/data/', '/audio/').replace('/dataset/','/')
    target = source.replace('/train/', '/val/')
    # assert os.path.exists(source)
    os.renames(source, target)




