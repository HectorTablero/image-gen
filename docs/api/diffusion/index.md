# Diffusion Processes

Diffusion processes define how noise is added to data during training and removed during generation. They form the core of diffusion-based generative models.

## Base Diffusion

The base class for all diffusion processes. It defines the interface that all diffusion implementations must follow.

**Main Methods:**
- `forward_sde(x, t)`: Calculates the drift and diffusion coefficients for the forward SDE at time `t`.
- `forward_process(x0, t)`: Applies the forward diffusion process to input `x0` at time `t`.
- `compute_loss(score, noise, t)`: Computes the loss between predicted and actual noise.
- `backward_sde(x, t, score)`: Computes the backward SDE coefficients for sampling.

[View Implementation](base.md)

## Variance Exploding

A diffusion process where noise increases exponentially over time. Suitable for image generation tasks.

**Main Methods:**
- `forward_sde(x, t)`: Implements the forward SDE for variance exploding diffusion.
- `forward_process(x0, t)`: Applies the forward process with exponential noise increase.
- `compute_loss(score, noise, t)`: Computes loss specific to variance exploding formulation.

[View Implementation](ve.md)

## Variance Preserving

Maintains a controlled level of variance throughout the diffusion process. Commonly used in various diffusion-based generative models.

**Main Methods:**
- `forward_sde(x, t)`: Implements the forward SDE for variance preserving diffusion.
- `forward_process(x0, t)`: Applies the forward process while preserving variance.
- `compute_loss(score, noise, t)`: Computes loss specific to variance preserving formulation.

[View Implementation](vp.md)

## Sub-Variance Preserving

A variant of variance preserving diffusion with modified noise characteristics.

**Main Methods:**
- `forward_sde(x, t)`: Implements the forward SDE for sub-variance preserving diffusion.
- `forward_process(x0, t)`: Applies the forward process with controlled variance.
- `compute_loss(score, noise, t)`: Computes loss specific to this diffusion variant.

[View Implementation](sub_vp.md)