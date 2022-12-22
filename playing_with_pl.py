#%%

# implement simple torch nn regressor for synthetic data

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import Callback

from typing import List, Callable
from collections import defaultdict
from tqdm import tqdm
from tools import Timer

timer = Timer()
timer2 = Timer()

class Plotter(Callback):
    def __init__(self, update_freq=1):
        _ = timer("plot", False)
        super().__init__()
        self.update_freq = update_freq
        self.counter = 0
        self.y_train = []
        self.y_val = []
        self.x_train = []
        self.x_val = []
        self.ylim = [0, 1e-5]
    def on_fit_start(self, trainer: pl.Trainer, *args):
        _ = timer("plot", False)
        self.fig, self.ax = plt.subplots(1, 1)
        self.fig.show()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.total = trainer.max_epochs*(len(trainer.datamodule.train_dataloader()))
        self.ax.set_xlim(0, self.total)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel("Training step")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Training loss")
        self.ax.grid()
        self.line_train, = self.ax.plot([], [], label="train loss")
        self.line_val, = self.ax.plot([], [], label="val loss")
        self.ax.legend()
    
    def on_train_batch_end(self, trainer, *args):
        self.counter += 1
        if self.counter % self.update_freq == 0 or self.counter == self.total:
            _ = timer("plot", False)
            loss = trainer.callback_metrics["train_loss"].item()
            self.y_train.append(loss)
            self.x_train.append(self.counter)
            self.line_train.set_ydata(self.y_train)
            self.line_train.set_xdata(self.x_train)
            self.ylim[0] = min(self.ylim[0], loss)
            self.ylim[1] = max(self.ylim[1], loss)
            self._draw()
    
    def on_validation_end(self, trainer, *args):
        _ = timer("plot", False)
        loss = trainer.callback_metrics["val_loss"].item()
        self.y_val.append(loss)
        self.x_val.append(self.counter)
        self.line_val.set_ydata(self.y_val)
        self.line_val.set_xdata(self.x_val)
        self.ylim[0] = min(self.ylim[0], loss)
        self.ylim[1] = max(self.ylim[1], loss)
        self._draw()

    def _draw(self):
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.ax.set_ylim(self.ylim)

class MyProgressBar(Callback):
    def __init__(self, update_freq=1):
        super().__init__()
        self.bar = None
        self.update_freq = update_freq
        self.counter = 0
    def on_train_start(self, trainer: pl.Trainer, *args):
        self.total = trainer.max_epochs*(len(trainer.datamodule.train_dataloader()))
        self.bar = tqdm(total=self.total, desc="Epoch", position=0)
    def on_train_batch_end(self, trainer, *args):
        self.counter += 1
        if self.counter % self.update_freq == 0 or self.counter == self.total:
            self.desc = f"Epoch {trainer.current_epoch}, train loss {trainer.callback_metrics['train_loss']:.3f}"
            try: self.desc += f", val loss {trainer.callback_metrics['val_loss']:.3f}"
            except: pass
            self.bar.set_description(self.desc)
            if self.counter != self.total:
                self.bar.update(self.update_freq)
            else:
                self.bar.update(self.total - self.update_freq*(self.counter//self.update_freq))
                self.bar.close()


    def on_train_end(self, trainer, *args):
        self.bar.close()
        print(self.counter)

class Dataset(data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = torch.from_numpy(self.X[index]).float()
        y = torch.from_numpy(self.y[index]).float()
        return X, y

def generate_data(n_samples, functions: List[Callable]):
    _ = timer("generate_data")
    n_features = len(functions)
    X = np.random.randn(n_samples, n_features)
    for i, f in enumerate(functions):
        functions[i] = np.vectorize(f)
    X_transformed = np.array([f(X[:, i]) for i, f in enumerate(functions)]).T
    y = np.sum(X_transformed, axis=1, keepdims=True)
    # timer()
    return Dataset(X, y)

class Net(pl.LightningModule):
# class Net(nn.Module):
    def __init__(self, n_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, n_features*5)
        self.fc2 = nn.Linear(n_features*5, n_features*5)
        self.fc3 = nn.Linear(n_features*5, 1)
        self.dropout = nn.Dropout(0.2)
        # self._log = defaultdict(list)
    
    # def log(self, key, value):
    #     self._log[key].append(value)

    def forward(self, x):
        _ = timer("forward", False)
        x = self.fc1(x)
        x = F.relu(x)
        x_res = self.fc2(x)
        x_res = self.dropout(x_res)
        x = x + x_res
        x = F.relu(x)
        x = self.fc3(x)
        # timer()
        return x
    
    def _compute_loss(self, batch, log=None, **log_kwargs):
        X, y = batch
        y_hat = self(X)
        loss = F.mse_loss(y_hat, y)
        if log: self.log(f'{log}_loss', loss, **log_kwargs)
        # timer()
        return loss

    def training_step(self, batch, batch_idx):
        _ = timer("training_step", False)
        return self._compute_loss(batch, log='train')
    
    def validation_step(self, batch, batch_idx):
        _ = timer("validation_step", False)
        return self._compute_loss(batch, log='val')
    
    def test_step(self, batch, batch_idx):
        _ = timer("test_step", False)
        return self._compute_loss(batch, log='test')
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

class DataModule(pl.LightningDataModule):
    def __init__(self, n_test, n_train, n_val, functions):
        super().__init__()
        self.n_test = n_test
        self.n_train = n_train
        self.n_val = n_val
        self.functions = functions
    
    def setup(self, stage=None):
        self.train_dataset = generate_data(self.n_train, self.functions)
        self.val_dataset = generate_data(self.n_val, self.functions)
        self.test_dataset = generate_data(self.n_test, self.functions)
    
    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=32, shuffle=True)
    
    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, batch_size=32, shuffle=True)
    
    def test_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=32, shuffle=True)

#%%
