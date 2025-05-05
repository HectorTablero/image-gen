# Imputation

This example demonstrates how to perform image inpainting using the generative model.

## Setup

Import the necessary modules and initialize the generative model.

```python
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from image_gen import GenerativeModel

# Initialize a generative model with Variance Preserving diffusion and Exponential Integrator sampler
model = GenerativeModel(
    diffusion="vp",
    sampler="exponential",
    noise_schedule="linear"
)
```

## Training

Load a dataset and train the model.

```python
# Load MNIST dataset
data = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor()
)

# Select a subset for faster training
indices = torch.where(data.targets == 3)[0]
dataset = torch.utils.data.Subset(data, indices)

# Train the model
model.train(dataset, epochs=50, batch_size=32)
```

## Imputation

Perform inpainting by providing a mask.

```python
# Generate a base image
base_image = model.generate(num_samples=1)

# Create a center rectangle mask
mask = torch.ones_like(base_image)
h, w = base_image.shape[2], base_image.shape[3]
mask[:, :, h//4:3*h//4, w//4:3*w//4] = 0

# Create a batch of images with the same mask
mask_batch = mask.repeat(16, 1, 1, 1)
base_image_batch = base_image.repeat(16, 1, 1, 1)

# Perform inpainting
results_batch = model.imputation(base_image_batch, mask_batch, n_steps=500)
display_images(results_batch)
```