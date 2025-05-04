# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 21:12:58 2025

Code adapted by alberto.suarez@uam.es from
https://yang-song.net/blog/2021/score/


"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from typing import Callable, Optional, List


class GaussianRandomFourierFeatures(nn.Module):
    """Gaussian random Fourier features for encoding time steps."""

    def __init__(self, embed_dim: int, scale: float = 30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.rff_weights = nn.Parameter(
            torch.randn(embed_dim // 2) * scale,
            requires_grad=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        x_proj = x[:, None] * self.rff_weights[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.dense(x)[..., None, None]


class ScoreNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, marginal_prob_std: Callable[[Tensor], float], channels: List[int] = [32, 64, 128, 256], embed_dim: int = 256, num_c: int = 3, num_classes: Optional[int] = None, class_dropout_prob: float = 0.25):
        """Initialize a time-dependent score-based network.

        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random Fourier feature embeddings.
          num_c: Number of input channels (1 for grayscale, 3 for RGB).
        """
        super().__init__()

        # Store configuration parameters
        self.num_channels = num_c
        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob

        # Gaussian random Fourier feature embedding layer for time
        self.embed = nn.Sequential(
            GaussianRandomFourierFeatures(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

        # Class embedding layer for conditional generation
        if num_classes is not None:
            self.class_embed = nn.Embedding(num_classes, embed_dim)

        # Encoding path - downsampling blocks
        # First block - no downsampling
        self.conv1 = nn.Conv2d(
            num_c,
            channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.dense1 = Dense(embed_dim, channels[0])
        # Fewer groups for smaller channels
        self.gnorm1 = nn.GroupNorm(4, channels[0])

        # Second block - downsample by factor of 2
        self.conv2 = nn.Conv2d(
            channels[0],
            channels[1],
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, channels[1])

        # Third block - downsample by factor of 2
        self.conv3 = nn.Conv2d(
            channels[1],
            channels[2],
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, channels[2])

        # Fourth block - downsample by factor of 2
        self.conv4 = nn.Conv2d(
            channels[2],
            channels[3],
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, channels[3])

        # Decoding path
        self.tconv4 = nn.ConvTranspose2d(
            channels[3], channels[2], 3, stride=2, padding=1, output_padding=1, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, channels[2])

        self.tconv3 = nn.ConvTranspose2d(
            channels[2] + channels[2], channels[1], 3, stride=2, padding=1, output_padding=1, bias=False)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, channels[1])

        self.tconv2 = nn.ConvTranspose2d(
            channels[1] + channels[1], channels[0], 3, stride=2, padding=1, output_padding=1, bias=False)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, channels[0])

        self.tconv1 = nn.Conv2d(
            channels[0] + channels[0], num_c, 3, stride=1, padding=1)

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x: Tensor, t: Tensor, class_label: Optional[int] = None):
        # Obtain the Gaussian random Fourier feature embedding for t
        t_embed = self.act(self.embed(t))

        # Class conditioning
        embed = t_embed
        if self.num_classes is not None and class_label is not None:
            class_embed = self.class_embed(class_label)
            embed = t_embed + class_embed
            if self.training and torch.rand(1) < self.class_dropout_prob:
                embed = t_embed

        # Encoding path
        h1 = self.conv1(x)
        h1 += self.dense1(embed)
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)

        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)

        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)

        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        # Decoding path
        h = self.tconv4(h4)
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)

        # Ensure spatial dimensions match before concatenation
        if h.shape[-2:] != h3.shape[-2:]:
            h = F.interpolate(h, size=h3.shape[-2:], mode='nearest')
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)

        if h.shape[-2:] != h2.shape[-2:]:
            h = F.interpolate(h, size=h2.shape[-2:], mode='nearest')
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)

        if h.shape[-2:] != h1.shape[-2:]:
            h = F.interpolate(h, size=h1.shape[-2:], mode='nearest')
        h = self.tconv1(torch.cat([h, h1], dim=1))

        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h
