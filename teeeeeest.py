#%%

import numpy as np
import os

audio_file_paths = np.load("val_files.npy")

try:
    for f in audio_file_paths:
        # move file to validation folder
        source = f.replace('/data/', '/audio/').replace('/dataset/','/')
        target = source.replace('/train/', '/val/')
        # os.renames(source, target)
        assert os.path.exists(source)
    
    print("we're good")
except:
    print("we're NOT good")



