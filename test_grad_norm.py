#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


data = pd.read_csv("experiments/from_bottom_v2_42/log/version_2/metrics.csv")

grad_norm_step = data["grad_norm_step"].dropna().values
train_loss_step = data["train_loss_step"].dropna().values

fig, axs = plt.subplots(1,2)
axs[0].plot(grad_norm_step)
axs[1].plot(train_loss_step)
axs[1].plot(np.convolve(train_loss_step, np.ones(100)/100, mode='valid'))
plt.show()



# %%
