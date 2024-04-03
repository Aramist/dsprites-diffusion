from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, n_hidden: int, d_model: int):
        super().__init__()
        channels = [1] + [d_model] * n_hidden

        layers = [
            nn.Linear(in_c, out_c) for in_c, out_c in zip(channels[:-1], channels[1:])
        ]

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, ksize: int, time_embedding_dim: int
    ):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, ksize, padding="same"),
            nn.BatchNorm2d(out_channels),
        )

        self.time_embedding = nn.Linear(time_embedding_dim, in_channels)

        self.one_by_one = nn.Conv2d(in_channels, out_channels, 1, padding="same")

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Image x should have shape (batch, channels, height, width),
        Time t should have shape (batch, time_embedding_dim)
        """
        return F.relu(
            self.convs(x + self.time_embedding(t)[..., None, None]) + self.one_by_one(x)
        )


class ResNet(nn.Module):
    def __init__(self, channels_per_block: list[int], time_embedding_dim: int):
        super().__init__()

        blocks = []
        channels_per_block = [1] + channels_per_block
        for in_channels, out_channels in zip(
            channels_per_block[:-1], channels_per_block[1:]
        ):
            blocks.append(
                ResnetBlock(
                    in_channels, out_channels, 3, time_embedding_dim=time_embedding_dim
                )
            )
        self.blocks = nn.ModuleList(blocks)

        self.global_time_embedding = TimeEmbedding(2, time_embedding_dim)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """X should have shape (batch, channels, height, width),
        t should have shape (batch, )"""
        t = t.unsqueeze(1)
        t_embedded = self.global_time_embedding(t)
        for block in self.blocks:
            x = block(x, t_embedded)
        return x
