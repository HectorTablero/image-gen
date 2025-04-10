# from .base import BaseSampler
# from torch import Tensor


# class ODEProbabilityFlow(BaseSampler):
#     def step(self, x: Tensor, t: Tensor, score: Tensor) -> Tensor:
#         drift, diffusion = self.diffusion.forward_sde(x, t)
#         adj_drift = drift - 0.5 * diffusion**2 * score
#         return x + adj_drift * self.dt

########################################################################################################3
from .base import BaseSampler
import torch
from torch import Tensor
from typing import Callable, Optional
from tqdm.autonotebook import tqdm

class ODEProbabilityFlow(BaseSampler):
    def __call__(
        self,
        x_T: Tensor,
        score_model: Callable,
        n_steps: int = 500,
        seed: Optional[int] = None,
        callback: Optional[Callable[[Tensor, int], None]] = None,
        callback_frequency: int = 50,
        guidance: Optional[Callable[[Tensor, Tensor, Tensor], Tensor]] = None
    ) -> Tensor:
        if seed is not None:
            torch.manual_seed(seed)

        device = x_T.device
        x_t = x_T.clone()

        times = torch.linspace(1.0, 1e-3, n_steps+1, device=device)
        dt = times[0] - times[1]

        iterable = tqdm(
            range(n_steps), desc='Generating') if self.verbose else range(n_steps)

        for i in iterable:
            t_curr = times[i]
            t_batch = torch.full((x_T.shape[0],), t_curr, device=device)

            # Manejo de estabilidad numérica
            if torch.isnan(x_t).any() or torch.isinf(x_t).any():
                if self.verbose:
                    print(f"Warning: NaN/Inf detectado en paso {i}")
                x_t = torch.nan_to_num(x_t, nan=0.0, posinf=1.0, neginf=-1.0)

            # Cálculo del score
            try:
                with torch.enable_grad():
                    x_t.requires_grad_(True)
                    score = score_model(x_t, t_batch)
                    
                    if guidance is not None:
                        score = guidance(x_t, t_batch, score)
                    
                    x_t.requires_grad_(False)
                
                if torch.isnan(score).any():
                    score = torch.nan_to_num(score, nan=0.0)
            except Exception as e:
                print(f"Error en cálculo de score: {e}")
                score = torch.zeros_like(x_t)
                x_t.requires_grad_(False)

            # Cálculo de coeficientes ODE
            drift, diffusion = self.diffusion.backward_sde(x_t, t_batch, score)
            diffusion = torch.nan_to_num(diffusion, nan=1e-4)
            
            # Ajuste para Probability Flow ODE (Ecuación 5 de Song et al. 2021)
            ode_drift = drift - 0.5 * (diffusion ** 2) * score

            # Paso determinista (sin término de difusión)
            x_t = x_t + ode_drift * (-dt)
            x_t = torch.clamp(x_t, -10.0, 10.0)

            # Monitoreo y callbacks
            if self.verbose and (i % callback_frequency == 0 or torch.isnan(x_t).any()):
                print(
                    f"Step {i}: t={t_curr:.3f}, mean={x_t.mean().item():.3f}, std={x_t.std().item():.3f}")

            if callback and i % callback_frequency == 0:
                callback(x_t.detach().clone(), i)

        return x_t