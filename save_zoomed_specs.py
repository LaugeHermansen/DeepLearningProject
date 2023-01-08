from scipy.ndimage import zoom
import numpy as np
from params import params
import os
from glob import glob
from tqdm import tqdm


def zoom_spec(spec, reduction_factor):
    ret = zoom(zoom(spec, reduction_factor), 1/reduction_factor)
    assert ret.shape[0] == params.n_mels
    return ret

if __name__ == "__main__":
    reduction_factor = 0.05
    
    spec_reduced_dir = os.path.join(params.spectrogram_dir_root, str(reduction_factor))
    spec_full_dir = os.path.join(params.spectrogram_dir_root, params.spectrogram_full_dir)

    spec_paths = glob(f"{spec_full_dir}/**/*.spec.npy", recursive=True)
    for spec_path in tqdm(spec_paths):
        spec_reduced_path = os.path.join(spec_reduced_dir, os.path.relpath(spec_path, spec_full_dir))
        if os.path.exists(spec_reduced_path): continue
        spec = np.load(spec_path)
        spec_reduced = zoom_spec(spec, reduction_factor)

        os.makedirs(os.path.dirname(spec_reduced_path), exist_ok=True)
        np.save(spec_reduced_path, spec_reduced)
