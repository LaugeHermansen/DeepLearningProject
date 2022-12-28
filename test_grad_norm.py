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
        versions[exp[0]] = [exp[1].split("_")[-1]]
    else:
        versions[exp[0]].append(exp[1].split("_")[-1])
        assert int(versions[exp[0]][-2]) < int(versions[exp[0]][-1]), "versions are not in order"

title = ", ".join([f"{exp}: {','.join(versions[exp])}" for exp in versions])


metrics_paths = [os.path.join("experiments", exp[0], "log", exp[1], "metrics.csv") for exp in experiments]

data = pd.concat([pd.read_csv(path) for path in metrics_paths])


start_step = 0
start_epoch = 0


fig = plt.figure(figsize=(10,5))


ax = fig.add_subplot(2,2,1)
# step_data = data[['step','grad_2_norm_step']].dropna()
epoch_data = data[['epoch','grad_2_norm_epoch']].dropna()
# ax.plot(step_data['step'], step_data['grad_2_norm_step'], ',', label='step')
ax.plot(epoch_data['epoch'], epoch_data['grad_2_norm_epoch'])
ax.set_title('Gradient norm epoch mean')
ax.set_xlabel('Epoch')
ax.set_ylabel('Gradient $L2$-norm')
# ax.legend()


ax = fig.add_subplot(1,2,2)


# ax = fig.add_subplot(2,2,4)
epoch_data = data[['epoch','val_loss_epoch']].dropna()
ax.plot(epoch_data['epoch'], epoch_data['val_loss_epoch'], label='Validation loss')
ax.set_title('Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
# ax.legend()

epoch_data = data[['epoch','train_loss_epoch']].dropna()
# ax.plot(step_data['step'], step_data['train_loss_step'], ',', label='Train step')
ax.plot(epoch_data['epoch'], epoch_data['train_loss_epoch'], label='Trainc´loss')
ax.set_title('Training curve')
ax.set_ylim(min(np.min(data[['train_loss_epoch', 'val_loss_epoch']]))-0.005, 0.095)

ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()


ax = fig.add_subplot(2,2,3)
step_data = data[['step','grad_2_norm_step']].dropna()
ax.hist(step_data['grad_2_norm_step'], bins=100, density=True)
ax.set_title('Histogram of gradient norm step')
ax.set_xlabel('Gradient $L2$-norm')
ax.set_ylabel('Density')

fig.tight_layout()
fig.suptitle(title)

plt.show()
