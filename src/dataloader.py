import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class DSpritesDataset(Dataset):
    def __init__(self, path: Path):
        self.path = path
        self.handle = h5py.File(path, "r")

        self.index: np.ndarray
        self.build_index()

    def __len__(self):
        # return len(self.handle["imgs"])
        return len(self.index)

    def build_index(self):
        mask = self.handle["latents/classes"][:, :5] == np.array([0, 0, 0, 0, 0])
        mask = mask.all(axis=1)
        indices = np.flatnonzero(mask)
        print(f"Found {len(indices)} images with the specified latent values")
        self.index = indices

    def __getitem__(self, idx: int):
        true_idx = self.index[idx]

        img = self.handle["imgs"][true_idx, ...].astype(np.float32)
        latent = self.handle["latents/values"][true_idx, :].astype(np.float32)

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

    train_loader = DataLoader(train, batch_size=8, shuffle=True)
    val_loader = DataLoader(test, batch_size=8, shuffle=True)

    return train_loader, val_loader
