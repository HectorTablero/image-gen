# Image Generation Library

Welcome to the documentation for our image generation library. This library provides tools for training and using diffusion-based generative models for image generation tasks.

## Features

- Implementations of various diffusion processes
- Multiple sampling algorithms
- Support for conditional image generation
- Utilities for evaluating model quality

## Getting Started

### Installation

Clone the repository:

```bash
git clone https://github.com/HectorTablero/image-gen.git
cd image-gen
pip install -e .
```

### Basic Usage

```python
from image_gen import GenerativeModel

# Initialize a generative model
model = GenerativeModel(diffusion="ve", sampler="euler-maruyama")

# Train the model
model.train(dataset, epochs=100, batch_size=32, lr=1e-3)

# Generate images
generated_images = model.generate(num_samples=10, n_steps=500)
```

### Documentation Structure

- [Generative Model](api/generative_model.md)
- [Diffusion Processes](api/diffusion/index.md)
- [Noise Schedules](api/noise/index.md)
- [Samplers](api/samplers/index.md)
- [Metrics](api/metrics/index.md)