#%%

from playing_with_pl import DataModule, Net, timer, CSVLogger
import pytorch_lightning as pl
import numpy as np

# generate data
n_test, n_train, n_val = 1000, 10000, 100
functions = [lambda x: x, lambda x: x ** 2, lambda x: np.sin(x)]

datamodule = DataModule(n_test, n_train, n_val, functions)


# define model
net = Net(len(functions))

#%%
# train model
max_epochs = 1
trainer_args = dict(
                    max_epochs=max_epochs,
                    accelerator="cpu",
                    devices="auto",
                    check_val_every_n_epoch=1, 
                    logger = CSVLogger("logs"),
                    # callbacks=[MyProgressBar(100), Plotter(50)],
                    enable_progress_bar=True
                    )
trainer = pl.Trainer(**trainer_args)

_ = timer("fit")
trainer.fit(net, datamodule=datamodule)
del _

print(timer.evaluate()[0])

