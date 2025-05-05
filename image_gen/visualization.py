"""
Visualization utilities for generative models.

This module provides functions to visualize the diffusion process,
create animations, and display generated images.
"""

import numpy as np
import torch
from matplotlib import animation
import matplotlib.pyplot as plt
from typing import Tuple, Any
from image_gen.base import GenerativeModel


def display_evolution(
    model: GenerativeModel,
    num_samples: int = 5,
    num_steps: int = 500,
    **kwargs: Any
) -> None:
    """
    Visualize the diffusion process by showing intermediate steps.

    Args:
        model: The generative model to use for image generation.
        num_samples: Number of images to generate.
        num_steps: Number of diffusion steps for generation.
        **kwargs: Additional keyword arguments for matplotlib.
    """
    captured_steps = []
    callback_frequency = max(num_steps // 10, 1)

    def progress_callback(x_t: torch.Tensor, step: int) -> None:
        if step % callback_frequency == 0 or step == num_steps:
            captured_steps.append(x_t.detach().cpu())

    model.generate(
        num_samples=num_samples,
        num_steps=num_steps,
        progress_callback=progress_callback,
        callback_frequency=callback_frequency,
        **kwargs,
    )

    # Stack and normalize images
    captured_images = torch.stack(captured_steps, dim=-1)
    captured_images = (captured_images + 1) / 2
    captured_images = np.clip(captured_images.numpy(), 0, 1)

    # Prepare figure
    num_steps_captured = captured_images.shape[-1]
    fig, axes = plt.subplots(
        num_samples, num_steps_captured, figsize=(10, int(num_samples * 1.25))
    )
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    # Plot each step
    for sample_idx in range(num_samples):
        for step_idx in range(num_steps_captured):
            ax = axes[sample_idx, step_idx]
            img = captured_images[sample_idx, ..., step_idx]
            if img.shape[0] == 1:  # Grayscale
                ax.imshow(img[0], cmap="gray")
            else:  # RGB
                ax.imshow(np.transpose(img, (1, 2, 0)))
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def create_evolution_widget(
    model: GenerativeModel, figsize: Tuple[int, int] = (6, 6), **kwargs: Any
) -> animation.FuncAnimation:
    """
    Create an interactive animation showing the diffusion process.

    Args:
        model: The generative model to use for image generation.
        figsize: Size of the animation figure.
        **kwargs: Additional keyword arguments for matplotlib.

    Returns:
        Matplotlib animation object that can be displayed in notebooks.
    """
    captured_steps = []

    def progress_callback(x_t: torch.Tensor, step: int) -> None:
        captured_steps.append(x_t.detach().cpu())

    model.generate(
        num_samples=1,
        progress_callback=progress_callback,
        callback_frequency=1,
        **kwargs,
    )

    # Process captured images
    captured_images = torch.stack(captured_steps, dim=0).squeeze(1)
    captured_images = (captured_images + 1) / 2
    captured_images = np.clip(captured_images.numpy(), 0, 1)
    captured_images = np.transpose(captured_images, (0, 2, 3, 1))

    # Create animation
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    if captured_images.shape[-1] == 1:
        img = ax.imshow(captured_images[0, ..., 0], cmap="gray")
    else:
        img = ax.imshow(captured_images[0])

    def update(frame: int):
        if captured_images.shape[-1] == 1:
            img.set_data(captured_images[frame, ..., 0])
        else:
            img.set_data(captured_images[frame])
        return [img]

    anim = animation.FuncAnimation(
        fig, update, frames=len(captured_images), interval=50, blit=True
    )

    plt.close(fig)
    return anim


def display_images(images: torch.Tensor, figsize: Tuple[int, int] = (6, 6)):
    """
    Display a grid of generated images.

    Args:
        images: Tensor of images to display (N, C, H, W).
        figsize: Size of the figure.
    """
    num_images = images.shape[0]
    row_size = int(np.sqrt(num_images))
    num_rows = int(np.ceil(num_images / row_size))
    num_channels = images.shape[1]

    images = images.permute(0, 2, 3, 1).cpu().detach().numpy()
    images = (images + 1) / 2  # Scale from [-1, 1] to [0, 1]
    images = np.clip(images, 0, 1)

    if num_images == 1:
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]
    else:
        fig, axes = plt.subplots(row_size, num_rows, figsize=figsize)
        axes = axes.flatten()

    for idx, img in enumerate(images):
        if num_channels == 1:
            axes[idx].imshow(img, cmap="gray")
        else:
            axes[idx].imshow(img)
        axes[idx].axis("off")

    for idx in range(num_images, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()
