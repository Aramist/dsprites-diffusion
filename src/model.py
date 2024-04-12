import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

from resnet import ResNet


class DiffusionModel(nn.Module):
    def __init__(self, *, var_min: float = 1e-4, var_max: float = 1.0):
        super().__init__()

        self.n_steps_per_sample = 16
        self.resnet = ResNet([16, 32, 64, 128, 256, 256], time_embedding_dim=64)
        self.var_min = var_min
        self.var_max = var_max

    def forward(self, x: torch.Tensor, compute_loss: bool = True):
        """Expects x to be a batch of one-channel images with shape (batch, height, width)"""
        # Sample random time steps
        time_steps = torch.rand(x.shape[0], self.n_steps_per_sample, device=x.device)
        # has shape (batch, num_steps_per_sample)
        batch, height, width = x.shape
        x = (
            x[:, None, ...]
            .expand(-1, self.n_steps_per_sample, -1, -1)
            .reshape(-1, height, width)
        )  # Shape: (batch * n_steps_per_sample, height, width)

        noise = torch.randn_like(x)
        noise_var = (time_steps * (self.var_max - self.var_min) + self.var_min).view(
            -1, 1, 1
        )  # Shape: (batch * n_steps_per_sample, 1 (height), 1 (width))
        scaled_noise = noise * torch.sqrt(noise_var)

        noisy_x = x + scaled_noise
        # squeeze in the channel dimension inline
        predicted_score = self.resnet(noisy_x.unsqueeze(1), time_steps.reshape(-1, 1))

        # Mean across channel dimension
        predicted_score = predicted_score.mean(dim=1).view(
            batch, self.n_steps_per_sample, height, width
        )
        # New shape: (batch, n_samples, height, width)

        if not compute_loss:
            return predicted_score
        # Write loss function:
        grad_log = (-scaled_noise / noise_var).reshape(
            batch, self.n_steps_per_sample, height, width
        )
        loss_scale = noise_var.reshape(batch, self.n_steps_per_sample)

        loss = torch.square(predicted_score - grad_log).mean(dim=(-1, -2)) / loss_scale
        return predicted_score, loss.mean(axis=1)
