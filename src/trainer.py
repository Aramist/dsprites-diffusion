from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim

from dataloader import get_dataloader
from model import DiffusionModel


class Trainer:
    def __init__(self, *, num_epochs: int = 10, learning_rate: float = 1e-3):
        self.model = DiffusionModel(var_min=0, var_max=0)
        self.num_epochs = num_epochs
        self.train_loader, self.val_loader = get_dataloader()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

    def train(self):
        for epoch in range(self.num_epochs):
            self.run_epoch()

    def run_epoch(self):
        for batch in self.train_loader:
            imgs, labels = batch
            img = imgs[0]
            plt.imshow(img)
            print(img)
            plt.show()
            # imgs = imgs.to(self.device)
            # labels = labels.to(self.device)

            # self.optimizer.zero_grad()
            # test = self.model(imgs)

    def validate(self):
        pass

    def _make_noisy_input(self, input: torch.Tensor) -> torch.Tensor:
        pass
