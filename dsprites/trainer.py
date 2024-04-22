import time
from pathlib import Path

import torch
from torch import optim
from tqdm import tqdm

from .dataloader import get_dataloader
from .model import DiffusionModel


class Trainer:
    def __init__(
        self,
        *,
        save_path: Path,
        num_epochs: int = 100,
        learning_rate: float = 1e-4,
        dset_path: Path = Path("dataset.hdf5"),
    ):
        start_time = time.time()
        self.model = DiffusionModel(var_min=1e-4, var_max=1.0)
        self.num_epochs = num_epochs
        self.train_loader, self.val_loader = get_dataloader(dset_path)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        self.model = self.model.to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        end_time = time.time()
        print(f"Time taken to initialize Trainer: {end_time - start_time:.2f}s")
        self.n_batches_processed = 0
        self.model_dir = save_path
        self.model_dir.mkdir(parents=True, exist_ok=True)

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
            self.n_batches_processed += 1

            if self.n_batches_processed % 100 == 0:
                print(f"Train loss on most recent batch: {loss.item()}")
                print(f"Saving model weights at batch #{self.n_batches_processed}")
                self.save_weights()

    def save_weights(self):
        weight_path = self.model_dir / f"model_weights_{self.n_batches_processed}.pth"
        torch.save(self.model.state_dict(), weight_path)

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
        print(f"Mean validation loss: {total_val_loss / num_processed}")
