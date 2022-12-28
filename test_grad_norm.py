#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


experiments = [
    # ("from_bottom_v3_42", "version_1"),
    # ("from_bottom_v3_42", "version_2"),
    ("from_bottom_v8_42", "version_0"),
    # ("from_bottom_RAM_42", "version_0"),
    # ("from_bottom_RAM_42", "version_1"),
    ]

versions = {}
for exp in experiments:
    if exp[0] not in versions:
        versions[exp[0]] = []
    versions[exp[0]].append(exp[1].split("_")[-1])

title = ", ".join([f"{exp}: {','.join(versions[exp])}" for exp in versions])


metrics_paths = [os.path.join("experiments", exp[0], "log", exp[1], "metrics.csv") for exp in experiments]

data = [pd.read_csv(path) for path in metrics_paths]

#%%



grad_norm_step = np.concatenate([d['grad_2_norm_step'].dropna().to_numpy() for d in data])
train_loss_step = np.concatenate([d['train_loss_step'].dropna().to_numpy() for d in data])
val_loss_epoch = np.concatenate([d['val_loss_epoch'].dropna().to_numpy() for d in data if "val_loss_epoch" in d.columns])


start = 0
conv_width = 50
assert start < len(grad_norm_step) and start < len(train_loss_step), "start index is too large"

fig, axs = plt.subplots(2,2)
# plot the gradient norm
axs[0,0].plot(grad_norm_step[start:], ',', label='grad_norm_step')

if len(grad_norm_step) > conv_width:
    axs[0,0].plot(np.convolve(grad_norm_step[start:], np.ones(conv_width)/conv_width, mode='valid'), label='grad_norm_step (smoothed)')
axs[0,0].set_title('grad_norm_step')
# axs[0,0].legend()

# plot the train loss step
axs[0,1].plot(train_loss_step[start:], ',',label='train_loss_step')
if len(train_loss_step) > conv_width:
    axs[0,1].plot(np.convolve(train_loss_step[start:], np.ones(conv_width)/conv_width, mode='valid'), label='train_loss_step (smoothed)')

axs[0,1].set_ylim(0, 0.15)
axs[0,1].set_title('train_loss_step')
# axs[0,1].legend()

# plot the histogram of the gradient norm
axs[1,0].hist(grad_norm_step[start:], bins=100, label='histogram of grad_norm_step')
axs[1,0].set_title('histogram of grad_norm_step')


# plot the validation loss epoch
axs[1,1].plot(val_loss_epoch, label='val_loss_epoch')
if len(val_loss_epoch) > conv_width:
    axs[1,1].plot(np.convolve(val_loss_epoch, np.ones(conv_width)/conv_width, mode='valid'), label='val_loss_epoch (smoothed)')
axs[1,1].set_title('val_loss_epoch')
# axs[1,1].legend()

fig.suptitle(title)
fig.tight_layout()
plt.show()
print(len(grad_norm_step))



# %%
