#%%


from params import params
from upsampler_data_module import UpsamplerDataModule, UpsamplerDataset
from speech_datamodule import SpeechDataModule, SpeechDataset, prepare_data
import os
from diffwave_model import DiffWave
from tqdm import tqdm
import numpy as np
import torch
import torchaudio
import multiprocessing as mp


EVAL_PATH = params.val_dir

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ORIGINAL_AUDIO_PATH = os.path.join(params.data_dir_root, EVAL_PATH)
ORIGINAL_SPEC_PATH = os.path.join(params.spectrogram_dir_root, params.spectrogram_full_dir, EVAL_PATH)

class ModelEvaluator:
    def __init__(self, model: DiffWave, experiment_dir, spectrogram_dir=None, n_samples=100, downscale=1.0):
        # init parameters
        self.model = model
        self.experiment_dir = experiment_dir
        self.downscale = downscale
        self.n_samples = n_samples

        self.start = 256
        self.end = 256 + 64

        self.spectrogram_dir = spectrogram_dir if spectrogram_dir is not None else experiment_dir
        self.path = os.path.join(params.project_dir_root, params.model_evaluator_results_dir, experiment_dir)
        os.makedirs(self.path, exist_ok=True)

        self.generated_audio_path = os.path.join(params.generated_audio_dir_root, experiment_dir, EVAL_PATH)
        self.generated_spec_path = os.path.join(params.generated_spectrogram_dir_root, experiment_dir, EVAL_PATH)

        self.original_dataset = SpeechDataset(ORIGINAL_AUDIO_PATH, ORIGINAL_SPEC_PATH)
        self.original_dataset.prepare_data(params) # to remove
        
        rng = np.random.default_rng(seed=42)
        self.idx = rng.choice(len(self.original_dataset), self.n_samples, replace=False)

        self.reduced_spec_file_paths = [os.path.join(params.spectrogram_dir_root, self.spectrogram_dir, EVAL_PATH, f) for f in self.original_dataset.spec_filenames]
        self.generated_audio_file_paths = [os.path.join(self.generated_audio_path, f) for f in self.original_dataset.audio_filenames]
        self.generated_spec_file_paths = [os.path.join(self.generated_spec_path, f) for f in self.original_dataset.spec_filenames]
        
        for f in self.generated_audio_file_paths:
            os.makedirs(os.path.dirname(f), exist_ok=True)
        for f in self.generated_spec_file_paths:
            os.makedirs(os.path.dirname(f), exist_ok=True)
        
        self.audio_generated = False
        self.spec_generated = False
        self.loss = None
        
        self.sanity_check()


    def sanity_check(self):
        assert os.path.exists(self.original_dataset.audio_file_paths[0]), f"{self.original_dataset.audio_file_paths[0]}"
        assert os.path.exists(self.original_dataset.spec_file_paths[0]), f"{self.original_dataset.spec_file_paths[0]}"
        assert os.path.exists(self.reduced_spec_file_paths[0]), f"{self.reduced_spec_file_paths[0]}"

    def __repr__(self):
        ret = f"ModelEvaluator for {self.path} with {len(self)} samples\n"
        ret += f"generated_audio_path: {self.generated_audio_path}\n"
        ret += f"generated_spec_path: {self.generated_spec_path}\n"
        ret += f"original_dataset_audio[0]: {self.original_dataset.audio_file_paths[0]}\n"
        ret += f"original_dataset_spec[0]: {self.original_dataset.spec_file_paths[0]}\n"
        ret += f"reduced_file_paths[0]: {self.reduced_spec_file_paths[0]}\n"
        ret += f"generated_audio_file_paths[0]: {self.generated_audio_file_paths[0]}\n"
        ret += f"generated_spec_file_paths[0]: {self.generated_spec_file_paths[0]}\n"
        return ret


    def __len__(self):
        return len(self.original_dataset)

    def evaluate(self, overwrite=False, parallel=False):
        self.generate_audio_from_spectrograms(parallel)
        self.generate_spectrograms_from_generated_audio()
        self.compute_loss(overwrite=overwrite)

    def compute_loss(self, overwrite=False):
        assert self.audio_generated and self.spec_generated, "Audio and spectrograms must be generated before evaluation can be performed"
        if os.path.exists(os.path.join(self.path, "loss.npy")) and not overwrite:
            self.loss = np.load(os.path.join(self.path, "loss.npy"))
        else:
            loss = []
            for i in tqdm(self.idx, desc=f"Computing loss {self.experiment_dir}"):
                generated_spec = np.load(self.generated_spec_file_paths[i])
                original_spec = np.load(self.original_dataset.spec_file_paths[i])
                original_spec = original_spec[:, self.start:self.end]
                assert generated_spec.shape[0] == original_spec.shape[0], "Spectrograms must have the same number of n_mels"
                length = min(generated_spec.shape[1], original_spec.shape[1])
                generated_spec = generated_spec[:, :length]
                original_spec = original_spec[:, :length]
                loss.append(np.mean((generated_spec - original_spec)**2))
            self.loss = np.array(loss)
            np.save(os.path.join(self.path, "loss.npy"), self.loss)

    def generate_audio_from_spectrograms(self, parallel=False):
        # generate audio from spectrograms
        for i in tqdm(self.idx, desc=f"Generating audio from spectrograms {self.experiment_dir}"):
            spec_path = self.reduced_spec_file_paths[i]
            audio_path = self.generated_audio_file_paths[i]
            if not os.path.exists(audio_path):

                spec = np.load(spec_path)[:, int(self.start*self.downscale):int(self.end*self.downscale)]
                spec = torch.from_numpy(spec).to(DEVICE)
                audio = self.model.predict_step({"spectrogram": spec}, None)
                torchaudio.save(audio_path, audio, params.sample_rate)
    
        self.audio_generated = True

    def generate_spectrograms_from_generated_audio(self):
        # generate spectrograms from audio
        assert self.audio_generated, "Audio must be generated before spectrograms can be generated from audio"
        
        prepare_data(
                    params,
                    self.generated_audio_file_paths[self.idx],
                    self.generated_audio_path,
                    self.generated_spec_file_paths[self.idx],
                    self.generated_spec_path,
                    )
        
        self.spec_generated = True
            


if __name__ == "__main__":
    

    paths = ["models_for_comparison/k-epoch=53-val_loss=0.037580.ckpt",
             "models_for_comparison/k-epoch=31-val_loss=0.043005_zoom_0_5.ckpt",]
    
    exp_names = ["full", "0.5"]

    spec_dirs = ["full", "0.5"]

    models = [DiffWave.load_from_checkpoint(path, map_location=DEVICE).to(DEVICE) for path in paths]

    model_evaluators = [ModelEvaluator(model, exp_name, spec_dir) for model, exp_name, spec_dir in zip(models, exp_names, spec_dirs)]

    for evaluator in model_evaluators:
        print(evaluator)
        evaluator.evaluate(overwrite=False, parallel=False)
