import torch

from utils import load_dataset, make_data, MyDataSet
from torch.utils.data import DataLoader
from pytorch_lighting_model.lighting import ContactMapEncoder
from torch.utils.data import random_split
from pytorch_lightning.callbacks import ModelCheckpoint

import pytorch_lightning as pl

if __name__ == '__main__':
    data_load = load_dataset("./dataset/dataset.pt")
    # data_load_val = load_dataset("./dataset.pt")
    enc_inputs, contact_map = make_data(data_load)
    # enc_inputs_valid, contact_map_valid = make_data(data_load_val)
    dataset = MyDataSet(enc_inputs, contact_map)
    # [mnist_train, mnist_val] = torch.load("./Example_dataset.pt")
    mnist_train, mnist_val = random_split(dataset, [2715, 200])
    torch.save([mnist_train, mnist_val], "./Example_dataset.pt")

    # [mnist_train, mnist_val] = torch.load("./Example_dataset.pt")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="sample-mnist-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
        save_last=False
    )
    loader = DataLoader(mnist_train, 64, True)
    loader_valid = DataLoader(mnist_val, 64, True)
    model = ContactMapEncoder()
    # trainer = pl.Trainer(accelerator="gpu", gpus=[0], limit_train_batches=0.5, max_epochs=500, min_epochs=499,)
    # trainer = pl.Trainer(accelerator="gpu",enable_checkpointing=False, gpus=[0], limit_train_batches=0.5, max_epochs=3, min_epochs=2)
    # trainer = pl.Trainer(accelerator="gpu", gpus=[0], limit_train_batches=0.5, max_epochs=900, min_epochs=800, callbacks=[checkpoint_callback])
    trainer = pl.Trainer(accelerator="gpu", gpus=[0], limit_train_batches=0.5, max_epochs=900, min_epochs=800, callbacks=[checkpoint_callback], resume_from_checkpoint='/root/autodl-nas/contact_map/lightning_logs/version_0/checkpoints/sample-mnist-epoch=107-val_loss=0.67.ckpt')
    trainer.fit(model, loader, loader_valid)

    # trainer.save_checkpoint("lightning_logs/example.ckpt")
