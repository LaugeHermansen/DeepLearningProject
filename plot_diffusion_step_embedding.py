#%%
# 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import pytorch_lightning




#%%


t = np.arange(1,51)
t_emb = np.vstack([np.concatenate((np.sin(10**(np.arange(64)*4/63)*t_), np.cos(10**(np.arange(64)*4/63)*t_))) for t_ in t])

step_size = 5
labels = t[step_size-1::step_size]
ticks = labels-1

plt.figure(dpi=400, figsize=(10,3))
plt.imshow(t_emb)
plt.yticks(ticks, labels)
# plt.xticks([], [])
plt.ylabel("Diffusion step, $t$")
plt.suptitle("Diffusion step embedding, $t_{embedding}$")
plt.colorbar(shrink=1)
plt.savefig("poster_report/Diffusion_step_embedding", bbox_inches = 'tight')

#%%


step_size = 40
