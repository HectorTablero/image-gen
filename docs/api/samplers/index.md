# Samplers

Samplers implement different algorithms for generating samples from the learned distribution. They define how noise is removed during the sampling process.

## Base Sampler

The base class for all samplers. It defines the interface that all sampler implementations must follow.

**Main Methods:**
- `__call__(x_T, score_model, ...)`: Performs the sampling process from initial noise `x_T`.

[View Implementation](base.md)

## Euler-Maruyama

Implements the Euler-Maruyama numerical method for solving stochastic differential equations.

**Main Methods:**
- `__call__(x_T, score_model, ...)`: Performs sampling using the Euler-Maruyama method.

[View Implementation](euler_maruyama.md)

## Predictor-Corrector

Combines a predictor step with a corrector step based on Langevin dynamics for improved sampling quality.

**Main Methods:**
- `predictor_step(x_t, t_curr, ...)`: Performs a prediction step.
- `corrector_step(x_t, t, ...)`: Performs a correction step.
- `__call__(x_T, score_model, ...)`: Performs sampling using the predictor-corrector method.

[View Implementation](predictor_corrector.md)

## ODE Probability Flow

A deterministic sampling method based on the probability flow ordinary differential equation.

**Main Methods:**
- `__call__(x_T, score_model, ...)`: Performs sampling using the ODE probability flow method.

[View Implementation](ode.md)

## Exponential Integrator

An exponential integration scheme for solving stochastic differential equations with better stability properties.

**Main Methods:**
- `__call__(x_T, score_model, ...)`: Performs sampling using the exponential integrator method.

[View Implementation](exponential.md)