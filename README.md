# Diffusion-Based Image Generation

## Overview

This project implements diffusion-based generative models for image synthesis. It provides a comprehensive framework for training and using diffusion models, with support for various diffusion processes, noise schedules, and sampling methods.

## Features

- **Multiple Diffusion Processes**: Support for Variance Exploding (VE), Variance Preserving (VP), and other diffusion types
- **Customizable Noise Schedules**: Linear, Cosine, and custom noise schedules
- **Various Samplers**: Euler-Maruyama, Exponential Integrator, ODE Probability Flow, and Predictor-Corrector samplers
- **Advanced Generation Capabilities**:
  - Unconditional image generation
  - Class-conditional image generation
  - Image colorization
  - Image inpainting/imputation
- **Interactive Dashboard**: Built with Streamlit for easy model interaction

## Installation

Clone the repository:

```bash
git clone https://github.com/HectorTablero/image-gen.git
cd image-gen
pip install -e .
```

## Usage

### Basic Generation

```python
from image_gen.base import GenerativeModel

# Load a pre-trained model
model = GenerativeModel.load("path/to/model.pt")

# Generate images
images = model.generate(num_images=4, n_steps=500, seed=42)
```

### Colorization

```python
# Colorize a grayscale image
colorized = model.colorize(grayscale_image, n_steps=500)
```

### Image Inpainting

```python
# Perform inpainting with a mask
inpainted = model.imputation(image, mask, n_steps=500)
```

## Interactive Dashboard

Run the dashboard to interact with your models:

```bash
streamlit run dashboard.py
```

## Documentation

The project includes automatically generated documentation. To view it locally:

```bash
# Build the documentation
mkdocs build

# Serve the documentation locally
mkdocs serve
```

A more comprehensive version of the documentation is available at [https://deepwiki.com/HectorTablero/image-gen](https://deepwiki.com/HectorTablero/image-gen)

## Examples

Check the `examples/` directory for Jupyter notebooks demonstrating different aspects of the framework:

- `getting_started.ipynb`: Basic introduction to the framework
- `diffusers.ipynb`: Working with different diffusion processes
- `noise_schedulers.ipynb`: Exploring various noise schedules
- `samplers.ipynb`: Comparing different sampling methods
- `colorization.ipynb`: Image colorization examples
- `imputation.ipynb`: Image inpainting examples
- `class_conditioning.ipynb`: Class-conditional generation
- `evaluation.ipynb`: Evaluating model performance

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- Héctor Tablero Díaz - [hector.tablerodiaz@gmail.com](mailto:hector.tablerodiaz@gmail.com)
- Álvaro Martínez Gamo - [alva00003@gmail.com](mailto:alva00003@gmail.com)
