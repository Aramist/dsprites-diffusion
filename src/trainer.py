import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from dataloader import get_dataloader
from model import DiffusionModel


class Trainer:
    def __init__(self, *, num_epochs: int = 10, learning_rate: float = 1e-4):
        start_time = time.time()
        self.model = DiffusionModel(var_min=1e-4, var_max=1.0)
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
        end_time = time.time()
        print(f"Time taken to initialize Trainer: {end_time - start_time:.2f}s")

    def train(self):
        for epoch in range(self.num_epochs):
            print(f"Starting epoch {epoch + 1} of {self.num_epochs}")
            self.run_train_epoch()
            self.validate()

    def run_train_epoch(self):
        for batch in tqdm(self.train_loader):
            imgs, labels = batch
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            score, loss = self.model(imgs, compute_loss=True)
            loss = loss.mean()
            loss.backward()

            self.optimizer.step()

    def validate(self):
        # Run validation
        total_val_loss = 0
        num_processed = 0
        for batch in self.val_loader:
            imgs, labels = batch
            imgs = imgs.to(self.device)

            with torch.no_grad():
                _, loss = self.model(imgs, compute_loss=True)
                total_loss = loss.mean().item()
                total_val_loss += total_loss
                num_processed += len(batch)

        print(f"Mean validation loss: {total_val_loss / num_processed}")
