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

EVAL_PATH = params.val_dir


ORIGINAL_AUDIO_PATH = os.path.join(params.data_dir_root, EVAL_PATH)
ORIGINAL_SPEC_PATH = os.path.join(params.spectrogram_dir_root, params.spectrogram_full_dir, EVAL_PATH)

class ModelEvaluator:
    def __init__(self, model: DiffWave, experiment_dir, spectrogram_dir=None):
        # init parameters
        self.model = model
        self.experiment_dir = experiment_dir
        self.spectrogram_dir = spectrogram_dir if spectrogram_dir is not None else experiment_dir
        self.path = os.path.join(params.project_dir_root, params.model_evaluator_results_dir, experiment_dir)
        os.makedirs(self.path, exist_ok=True)

        self.generated_audio_path = os.path.join(params.generated_audio_dir_root, experiment_dir, EVAL_PATH)
        self.generated_spec_path = os.path.join(params.generated_spectrogram_dir_root, experiment_dir, EVAL_PATH)

        self.original_dataset = SpeechDataset(ORIGINAL_AUDIO_PATH, ORIGINAL_SPEC_PATH)
        self.original_dataset.prepare_data(params) # to remove

        self.reduced_spec_file_paths = [os.path.join(params.spectrogram_dir_root, experiment_dir, EVAL_PATH, f) for f in self.original_dataset.spec_filenames]
        self.generated_audio_file_paths = [os.path.join(self.generated_audio_path, f) for f in self.original_dataset.audio_filenames]
        self.generated_spec_file_paths = [os.path.join(self.generated_spec_path, f) for f in self.original_dataset.spec_filenames]
        
        for f in self.reduced_spec_file_paths:
            os.makedirs(os.path.dirname(f), exist_ok=True)
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

    def evaluate(self, overwrite=False):
        self.generate_audio_from_spectrograms()
        self.generate_spectrograms_from_generated_audio()
        self.compute_loss(overwrite=overwrite)

    def compute_loss(self, overwrite=False):
        assert self.audio_generated and self.spec_generated, "Audio and spectrograms must be generated before evaluation can be performed"
        if os.path.exists(os.path.join(self.path, "loss.npy")) and not overwrite:
            self.loss = np.load(os.path.join(self.path, "loss.npy"))
        else:
            loss = []
            for i in tqdm(range(len(self)), desc=f"Computing loss {self.experiment_dir}"):
                generated_spec = np.load(self.generated_spec_file_paths[i])
                original_spec = np.load(self.original_dataset.spec_file_paths[i])
                assert generated_spec.shape[0] == original_spec.shape[0], "Spectrograms must have the same number of n_mels"
                length = min(generated_spec.shape[1], original_spec.shape[1])
                generated_spec = generated_spec[:, :length]
                original_spec = original_spec[:, :length]
                loss.append(np.mean((generated_spec - original_spec)**2))
            self.loss = np.array(loss)
            np.save(os.path.join(self.path, "loss.npy"), self.loss)

    def generate_audio_from_spectrograms(self):
        # generate audio from spectrograms
        for i in tqdm(range(len(self)), desc=f"Generating audio from spectrograms to {self.generated_audio_path}"):
            
            spec_path = self.reduced_spec_file_paths[i]
            audio_path = self.generated_audio_file_paths[i]

            if not os.path.exists(audio_path):
                spec = np.load(spec_path)
                audio = self.model.predict_step({"spectrogram": spec})
                torchaudio.save(audio_path, audio, params.sample_rate)
        
        self.audio_generated = True

    def generate_spectrograms_from_generated_audio(self):
        # generate spectrograms from audio
        assert self.audio_generated, "Audio must be generated before spectrograms can be generated from audio"
        
        prepare_data(
                    params,
                    self.generated_audio_file_paths,
                    self.generated_audio_path,
                    self.generated_spec_file_paths,
                    self.generated_spec_path,
                    )
        
        self.spec_generated = True
            


if __name__ == "__main__":
    

    paths = ["models_for_comparison/k-epoch=53-val_loss=0.037580.ckpt",
             "models_for_comparison/time-epoch=209-val_loss=0.041662.ckpt",]
    
    exp_names = ["epoch_53", "epoch_209"]

    models = [DiffWave.load_from_checkpoint(path) for path in paths]

    model_evaluators = [ModelEvaluator(model, exp_name) for model, exp_name in zip(models, exp_names)]

    for evaluator in model_evaluators:
        print(evaluator)
        # evaluator.evaluate(overwrite=False)

