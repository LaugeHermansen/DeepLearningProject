#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os



data = pd.read_csv("experiments/from_bottom_v4_42/log/version_0/metrics.csv")


#%%

grad_norm_step = data["grad_norm_step"].dropna().values
train_loss_step = data["train_loss_step"].dropna().values
start = 100
fig, axs = plt.subplots(2,2)
axs[0,0].plot(grad_norm_step[start:], alpha=0.2, markersize=2, marker='o', linestyle='None', label='grad_norm_step')
axs[0,0].plot(np.convolve(grad_norm_step[start:], np.ones(100)/100, mode='valid'), label='grad_norm_step (smoothed)')
axs[0,0].set_title('grad_norm_step')
axs[0,0].set_ylim([0, 36])
axs[0,0].legend()
axs[0,1].plot(train_loss_step[start:], label='train_loss_step')
axs[0,1].plot(np.convolve(train_loss_step[start:], np.ones(100)/100, mode='valid'), label='train_loss_step (smoothed)')
axs[0,1].set_title('train_loss_step')
axs[0,1].legend()
axs[1,1].hist(grad_norm_step[start:], bins=100, label='histogram of grad_norm_step')
axs[1,1].set_title('histogram of grad_norm_step')


plt.show()



# %%
