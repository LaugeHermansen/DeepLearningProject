#%%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from tools import glob

def plot_epoch(ax: plt.Axes, metric_path):
    markersize = 2
    window = 100
    
    df = pd.read_csv(metric_path)
    
    df_val_epoch = df[['epoch', 'val_loss_epoch']].dropna(subset='val_loss_epoch').set_index('epoch')
    df_train_epoch = df[['epoch', 'train_loss_epoch']].dropna(subset='train_loss_epoch').set_index('epoch')
    
    ax.plot(df_train_epoch, '.', markersize=markersize, label='train')
    ax.plot(df_val_epoch, '.', markersize=markersize, label='val')
    
    ax.plot(df_train_epoch.rolling(window=window).mean(), label='train (smoothed)')
    ax.plot(df_val_epoch.rolling(window=window).mean(), label='val (smoothed)')


def plot_step(ax, metric_path):
    markersize = 2
    window = 1000
    
    df = pd.read_csv(metric_path)
    
    df_train_step = df[['step', 'train_loss_step']].dropna(subset='train_loss_step').set_index('step')
    
    ax.plot(df_train_step, '.', markersize=markersize, label='train')
    
    ax.plot(df_train_step.rolling(window=window).mean(), label='train (smoothed)')

metric_paths = glob('model_checkpoints/**/metrics*.csv', recursive=True)
seeds = []
for p in metric_paths:
    try:
        seeds.append(int(p.split('.')[0].split('_')[-1]))
    except ValueError:
        seeds.append(None)


fig = plt.figure(figsize=(20, 10))

ax0 = fig.add_subplot(121)
plot_epoch(ax0, metric_paths[0])
ax0.set_title('Losses for seed {}'.format(seeds[0]), fontsize=20)
ax0.set_xlabel('Epoch', fontsize=16)
ax0.set_ylabel('Loss', fontsize=16)
ax0.legend()

plt.show()


#%%

ax1 = fig.add_subplot(122)
plot_epoch(ax1, metric_paths[1])
ax1.set_title('Losses for seed {}'.format(seeds[1]), fontsize=20)
ax1.set_xlabel('Epoch', fontsize=16)
ax1.legend()

_ = plt.plot()

metric_paths = glob('metrics*.csv')
seeds = [int(metric_path.split('.')[0].split('_')[-1]) for metric_path in metric_paths]

fig = plt.figure(figsize=(20, 10))

ax0 = fig.add_subplot(121)
plot_step(ax0, metric_paths[0])
ax0.set_title('Step losses for seed {}'.format(seeds[0]), fontsize=20)
ax0.set_xlabel('Step', fontsize=16)
ax0.set_ylabel('Step Loss', fontsize=16)
ax0.legend()

ax1 = fig.add_subplot(122)
plot_step(ax1, metric_paths[1])
ax1.set_title('Step losses for seed {}'.format(seeds[1]), fontsize=20)
ax1.set_xlabel('Step', fontsize=16)
ax1.legend()

plt.show()

_ = plt.plot()
