#%%

import numpy as np
from glob import glob
from params import params
import os


result_paths = glob(f"{params.model_evaluator_results_dir}/**/*loss.npy", recursive=True)
experiment_names = [os.path.basename(os.path.dirname(p)) for p in result_paths]
results = [np.load(p) for p in result_paths]

for result, experiment_name in zip(results, experiment_names):
    print(experiment_name)
    print(f"{result.mean():.4f} +- {1.96*result.std():.4f}")
    print()


