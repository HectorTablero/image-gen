# Colorization

This example demonstrates how to perform image colorization using the generative model.

## Setup

Import the necessary modules and initialize the generative model.

```python
import torch
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from image_gen import GenerativeModel

# Initialize a generative model with Variance Exploding diffusion and Euler-Maruyama sampler
model = GenerativeModel(diffusion="ve", sampler="euler-maruyama")
```

## Training

Load a dataset and train the model.

```python
# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

data = datasets.CIFAR10(
    root='data',
    train=True,
    download=True,
    transform=transform
)

# Select a subset for faster training
targets = torch.tensor(data.targets)
idx = (targets == 1).nonzero().flatten()
data = torch.utils.data.Subset(data, idx)

# Train the model
model.train(data, epochs=500, batch_size=32)
```

## Colorization

Colorize a grayscale image using the trained model.

```python
# Generate a base image
generated_image = model.generate(num_samples=1)

# Convert to grayscale
gray_image = torch.mean(generated_image, dim=1, keepdim=True)

# Display original and grayscale images
display_images(generated_image)
display_images(gray_image)

# Colorize the grayscale image
colorized = model.colorize(gray_image)
display_images(colorized)

# Generate multiple color variations
gray_batch = gray_image.repeat(16, 1, 1, 1)
colorized_batch = model.colorize(gray_batch)
display_images(colorized_batch)
```