#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


experiments = [
    # (experiment_name, version, end_epoch (inclusive, None means all epochs are included))
    # ("from_bottom_v3_42", "version_1"),
    # ("from_bottom_v3_42", "version_2"),
    ("from_bottom_v8_42", "version_0", 239),
    ("from_bottom_v8_42", "version_1", 404),
    ("from_bottom_v8_42", "version_2", None),
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

assert len(versions) == 1, "only one experiment name allowed"

title = f"experiment: {experiments[0][0]}"

metrics_paths = [os.path.join("experiments", exp[0], "log", exp[1], "metrics.csv") for exp in experiments]

data = [pd.read_csv(path) for path in metrics_paths]

for i in range(len(data)):
    # filter out the epochs after end_epoch
    if experiments[i][2] is not None:
        data[i] = data[i][data[i]['epoch'] <= experiments[i][2]]

# concatenate the data
data = pd.concat(data, axis=0)

#%%


start_step = 0
start_epoch = 0


fig = plt.figure(figsize=(10,5))


ax = fig.add_subplot(2,2,1)
# step_data = data[['step','grad_2_norm_step']].dropna()
epoch_data = data[['epoch','grad_2_norm_epoch']].dropna()
# ax.plot(step_data['step'], step_data['grad_2_norm_step'], ',', label='step')
ax.plot(epoch_data['epoch'], epoch_data['grad_2_norm_epoch'])
ax.set_title('Gradient norm epoch')
ax.set_xlabel('Epoch')
ax.set_ylabel('Gradient $L2$-norm')
# ax.legend()


ax = fig.add_subplot(1,2,2)


# ax = fig.add_subplot(2,2,4)
epoch_data = data[['epoch','val_loss_epoch']].dropna()
ax.plot(epoch_data['epoch'], epoch_data['val_loss_epoch'], label='Validation loss')

epoch_data = data[['epoch','train_loss_epoch']].dropna()
ax.plot(epoch_data['epoch'], epoch_data['train_loss_epoch'], label='Train loss')

ax.set_title('Training curve')
ax.set_ylim(min(np.min(data[['train_loss_epoch', 'val_loss_epoch']]))-0.005, 0.095)

ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (unweighted $D_{KL}$)')
ax.legend()


step_data = data[['step', 'epoch', 'grad_2_norm_step']].dropna()
ax = fig.add_subplot(2,2,3)
ax.set_title('Gradient norm step distribution')
step_data_split = []

if False:
    n_splits = 4
    split_idx = int(len(step_data)/n_splits)+1
    for i in range(n_splits):
        # ax = fig.add_subplot(2,n_splits*2,n_splits*2+i+1)
        step_data_split.append(step_data.iloc[i*split_idx:(i+1)*split_idx])

else:
    split_epoch = [320]
    split_epoch = [step_data.iloc[0]['epoch']] + split_epoch
    split_epoch.append(step_data.iloc[-1]['epoch']+1)

    for i in range(len(split_epoch)-1):
        step_data_split.append(step_data[(step_data['epoch'] >= split_epoch[i]) & (step_data['epoch'] < split_epoch[i+1])])

for i in range(len(step_data_split)):
    ax.hist(step_data_split[i]['grad_2_norm_step'], bins=100, density=True, alpha = 0.4, label=f'Epochs {step_data_split[i].iloc[0]["epoch"]:.0f} to {step_data_split[i].iloc[-1]["epoch"]:.0f}')
for i in range(len(step_data_split)):
    ax.hist(step_data_split[i]['grad_2_norm_step'], bins=100, density=True, histtype='step', linewidth=1, edgecolor='black')



ax.set_xlabel('Gradient $L2$-norm')
ax.set_ylabel('Density')

ax.legend()



fig.tight_layout()

plt.show()
