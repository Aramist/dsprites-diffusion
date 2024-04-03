import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class DSpritesDataset(Dataset):
    def __init__(self, path: Path):
        self.path = path
        self.handle = h5py.File(path, "r")

    def __len__(self):
        return len(self.handle["imgs"])

    def __getitem__(self, idx: int):
        img = self.handle["imgs"][idx, ...].astype(np.float32)
        latent = self.handle["latents/values"][idx, :].astype(np.float32)

        img = (img - img.mean()) / img.std()

        return img, latent


def get_dataloader() -> tuple[DataLoader, DataLoader]:
    dset_path = Path("dataset.hdf5")
    dset = DSpritesDataset(dset_path)
    try:
        avail_cpus = max(1, len(os.sched_getaffinity(0)) - 1)
    except:
        avail_cpus = 1

    train, test = random_split(
        dset,
        [0.95, 0.05],
        generator=torch.Generator(),
    )

    train_loader = DataLoader(train, batch_size=64, shuffle=True)
    val_loader = DataLoader(test, batch_size=64, shuffle=True)

    return train_loader, val_loader
