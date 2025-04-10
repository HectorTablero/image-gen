# from .base import BaseSampler
# from torch import Tensor


# class PredictorCorrector(BaseSampler):
#     def step(self, x: Tensor, t: Tensor, score: Tensor) -> Tensor:
#         pass

#####################################################################################
from .base import BaseSampler
import torch
from torch import Tensor
from typing import Callable, Optional
from tqdm.autonotebook import tqdm

class PredictorCorrector(BaseSampler):
    def __init__(self, diffusion, verbose=False, n_corrector_steps=1, corrector_step_size=0.1):
        super().__init__(diffusion, verbose)
        self.n_corrector_steps = n_corrector_steps
        self.corrector_step_size = corrector_step_size

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
            
            # ========== Predictor Step (Euler-Maruyama) ==========
            t_batch = torch.full((x_T.shape[0],), t_curr, device=device)
            
            # Manejo de valores inválidos
            if torch.isnan(x_t).any() or torch.isinf(x_t).any():
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
                print(f"Error en cálculo de score (predictor): {e}")
                score = torch.zeros_like(x_t)
                x_t.requires_grad_(False)

            # Cálculo de coeficientes SDE
            drift, diffusion = self.diffusion.backward_sde(x_t, t_batch, score)
            diffusion = torch.nan_to_num(diffusion, nan=1e-4)
            
            # Paso de Euler-Maruyama
            noise = torch.randn_like(x_t)
            x_t = x_t + drift * (-dt) + diffusion * torch.sqrt(torch.abs(dt)) * noise
            x_t = torch.clamp(x_t, -10.0, 10.0)

            # ========== Corrector Step (Langevin) ==========
            t_next = times[i+1]
            for _ in range(self.n_corrector_steps):
                t_batch_corr = torch.full((x_t.shape[0],), t_next, device=device)
                
                try:
                    with torch.enable_grad():
                        x_t.requires_grad_(True)
                        score_corr = score_model(x_t, t_batch_corr)
                        
                        if guidance is not None:
                            score_corr = guidance(x_t, t_batch_corr, score_corr)
                        
                        x_t.requires_grad_(False)
                    
                    if torch.isnan(score_corr).any():
                        score_corr = torch.nan_to_num(score_corr, nan=0.0)
                except Exception as e:
                    print(f"Error en cálculo de score (corrector): {e}")
                    score_corr = torch.zeros_like(x_t)
                    x_t.requires_grad_(False)
                
                # Paso de Langevin
                # Paso de Langevin (corregido)
                noise_corr = torch.randn_like(x_t)
                corrector_factor = torch.sqrt(torch.tensor(2 * self.corrector_step_size, device=device))
                x_t = x_t + self.corrector_step_size * score_corr + corrector_factor * noise_corr
                x_t = torch.clamp(x_t, -10.0, 10.0)

                # Manejo de estabilidad
                if torch.isnan(x_t).any() or torch.isinf(x_t).any():
                    x_t = torch.nan_to_num(x_t, nan=0.0, posinf=1.0, neginf=-1.0)

            # Callback y monitoreo
            if self.verbose and (i % callback_frequency == 0 or torch.isnan(x_t).any()):
                print(f"Paso {i}: t={t_curr:.3f}, media={x_t.mean().item():.3f}, std={x_t.std().item():.3f}")

            if callback and i % callback_frequency == 0:
                callback(x_t.detach().clone(), i)

        return x_t