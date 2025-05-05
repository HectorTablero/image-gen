# Basic Usage

This example demonstrates the basic usage of the GenerativeModel class for training and generating images.

## Setup

First, import the necessary modules and initialize the generative model.

```python
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from image_gen import GenerativeModel

# Initialize a generative model with Variance Exploding diffusion and Euler-Maruyama sampler
model = GenerativeModel(diffusion="ve", sampler="euler-maruyama")
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

## Generation

Generate new images using the trained model.

```python
# Generate 16 images
generated_images = model.generate(num_samples=16, n_steps=500)

# Display the generated images
from image_gen.visualization import display_images
display_images(generated_images)
```

## Saving and Loading

Save the trained model and load it later for inference.

```python
# Save the model
model.save("saved_models/mnist_model.pth")

# Load the model
model.load("saved_models/mnist_model.pth")
```

## Visualization

Visualize the generation process step by step.

```python
from image_gen.visualization import create_evolution_widget

# Create an animation showing the generation process
animation = create_evolution_widget(model)
animation
```