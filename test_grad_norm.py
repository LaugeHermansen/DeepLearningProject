#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


data = pd.read_csv("experiments/from_bottom_v2_42/log/version_2/metrics.csv")

grad_norm_step = data["grad_norm_step"].dropna().values
train_loss_step = data["train_loss_step"].dropna().values

fig, axs = plt.subplots(2,2)
axs[0,0].plot(grad_norm_step, alpha=0.2, markersize=2, marker='o', linestyle='None')
axs[0,0].plot(np.convolve(grad_norm_step, np.ones(100)/100, mode='valid'))
axs[0,1].plot(train_loss_step)
axs[0,1].plot(np.convolve(train_loss_step, np.ones(100)/100, mode='valid'))
axs[1,1].hist(grad_norm_step, bins=100)
plt.show()



# %%
