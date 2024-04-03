import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

from resnet import ResNet


class DiffusionModel(nn.Module):
    def __init__(self, *, var_min: float = 1e-4, var_max: float = 1):
        super().__init__()

        self.n_steps_per_sample = 16
        self.resnet = ResNet([16, 32, 64, 128, 256, 256], time_embedding_dim=64)
        self.var_min = var_min
        self.var_max = var_max

    def forward(self, x: torch.Tensor):
        # Sample random time steps
        time_steps = torch.rand(
            x.shape[0], self.n_steps_per_sample, device=x.device
        ).unsqueeze(1)
        x = (
            x.unsqueeze(1)
            .expand(-1, self.n_steps_per_sample, -1, -1, -1)
            .reshape(-1, *x.shape[1:])
        )  # Shape: (batch * n_steps_per_sample, channels, height, width)

        noise = torch.randn_like(x)
        noise_var = (
            time_steps * (self.var_max - self.var_min) + self.var_min
        )  # Shape: (batch, n_steps_per_sample)
        scaled_noise = (noise * torch.sqrt(noise_var)).view(-1, 1, 1, 1)
        # squeeze in channel dimension
        noisy_x = x + scaled_noise
        predicted_score = self.resnet(noisy_x, time_steps)

        # Write loss function:
        grad_log = -scaled_noise / noise_var
        loss = F.mse_loss(predicted_score, grad_log)
