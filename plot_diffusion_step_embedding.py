#%%

import numpy as np
import matplotlib.pyplot as plt
import torch
import pytorch_lightning
from archive_2.eat_my_balls import DiffusionEmbedding
from tools import get_cmap

de = DiffusionEmbedding(50)

fig = plt.figure(dpi=400, figsize=(10,3))
ax = plt.imshow(de.embedding, cmap=get_cmap())
plt.yticks([0,*np.arange(1,6)*10-1], [1, *np.arange(1,6)*10])
plt.ylabel("Diffusion step, $t$")
plt.suptitle("Diffusion step embedding, $t_{embedding}$")
plt.colorbar()

for pos in ['right', 'top', 'bottom', 'left']:
    plt.gca().spines[pos].set_visible(False)
plt.savefig("poster_report/Diffusion_step_embedding", bbox_inches = 'tight')
plt.show()
