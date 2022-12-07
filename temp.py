#%%
from eat_my_balls import SpeechDataModule, base_params, AttrDict
from tools import mkdir
import time
from tqdm import tqdm
import os


model_path = "D:\DTU\DeepLearningProject\diffwave-ljspeech-22kHz-1000578.pt"

# data paths
data_path = "data/NST/dataset/"
test_path = "data/NST/dataset/test/Female"

train_path = "data/NST/dataset/train/Female"

# output path
output_path = "tests"


#dont touch this
manual_test_id = "NST_test"
test_id=manual_test_id

to_root = mkdir(output_path)

base_params["data_dir"] = f"{to_root}/{test_id}"

params = AttrDict(base_params)


dm = SpeechDataModule(params, f"{to_root}/{test_id}", f"{to_root}/{test_id}")

while not os.path.exists(f"{to_root}/{test_id}/file_paths.npy"):
    dm.prepare_data()
    for i in tqdm(range(60)):
        time.sleep(1)