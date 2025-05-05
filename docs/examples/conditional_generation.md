# Conditional Generation

This example demonstrates how to perform conditional generation using class labels.

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

# Train the model
model.train(data, epochs=50, batch_size=32)
```

## Conditional Generation

Generate images conditioned on specific class labels.

```python
# Generate 16 samples from class 7
samples = model.generate(num_samples=16, class_labels=7)
display_images(samples)

# Generate specific classes for each sample
labels = torch.repeat_interleave(torch.arange(0, model.num_classes), 2)
samples = model.generate(num_samples=len(labels), class_labels=labels)
display_images(samples)
```

## Visualization

Visualize the effect of different guidance scales.

```python
from image_gen.visualization import create_evolution_widget

# Create an animation showing the generation process for class 9
animation = create_evolution_widget(model, class_labels=9)
animation

# Compare different guidance scales
fig, axs = plt.subplots(2, 4, figsize=(15, 8))
for i, scale in enumerate([0, 0.5, 1, 2, 3, 5, 7.5, 10]):
    samples = model.generate(num_samples=1, class_labels=6, guidance_scale=scale)
    axs[i//4, i%4].imshow(samples[0].permute(1, 2, 0), cmap="gray")
    axs[i//4, i%4].set_title(f'Scale={scale}')
plt.tight_layout()
plt.show()
```