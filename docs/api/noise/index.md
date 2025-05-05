# Noise Schedules

Noise schedules control how noise is added during the diffusion process. They determine the amount of noise added at each timestep.

## Base Noise Schedule

The base class for all noise schedules. It defines the interface for noise schedule implementations.

**Main Methods:**
- `__call__(t)`: Calculates noise magnitude at time `t`.
- `integral_beta(t)`: Calculates the integral of the noise function up to time `t`.

[View Implementation](base.md)

## Linear Noise Schedule

A simple noise schedule where noise increases linearly from `beta_min` to `beta_max`.

**Main Methods:**
- `__call__(t)`: Returns linearly increasing noise values.
- `integral_beta(t)`: Computes the integral of the linear noise function.

[View Implementation](linear.md)

## Cosine Noise Schedule

A noise schedule using a cosine function for smoother transitions between noise levels.

**Main Methods:**
- `__call__(t)`: Returns noise values following a cosine curve.
- `integral_beta(t)`: Computes the integral of the cosine noise function.

[View Implementation](cosine.md)