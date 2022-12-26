#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


data = pd.read_csv("experiments/from_bottom_v2_42/log/version_1/metrics.csv")

grad_norms = data["grad_norm_step"].dropna()
val_losses = data["train_loss_step"].dropna()


plt.plot(grad_norms)
plt.show()
plt.plot(val_losses)
plt.show()




# %%
