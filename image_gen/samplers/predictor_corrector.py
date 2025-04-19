# from .base import BaseSampler
# from torch import Tensor


# class PredictorCorrector(BaseSampler):
#     def step(self, x: Tensor, t: Tensor, score: Tensor) -> Tensor:
#         pass

#####################################################################################
from .base import BaseSampler
import torch
from torch import Tensor
from typing import Callable, Optional, Tuple, List, Union
from tqdm.autonotebook import tqdm


class PredictorCorrector(BaseSampler):
    def __init__(self, diffusion, verbose=False, corrector_steps=1, corrector_snr=0.15):
        """
        Inicializa el sampler de Predictor-Corrector.

        Args:
            diffusion: Objeto de difusión (como VarianceExploding)
            verbose: Si debe mostrar información detallada
            corrector_steps: Número de pasos de corrección por cada paso predictor
            corrector_snr: Relación señal-ruido para el paso corrector (controla la magnitud del ruido)
        """
        super().__init__(diffusion, verbose)
        self.corrector_steps = corrector_steps
        self.corrector_snr = corrector_snr
        
    def predictor_step(
        self,
        x_t: Tensor,
        t_curr: Tensor,
        t_next: Tensor,
        score: Tensor
    ) -> Tensor:
        """
        Realiza un paso de predictor (similar a Euler-Maruyama).
        """
        # Aseguramos que dt tenga las dimensiones correctas para broadcasting
        dt = (t_curr - t_next).view(-1, 1, 1, 1)
        
        # Obtenemos drift y diffusion
        drift, diffusion = self.diffusion.backward_sde(x_t, t_curr, score)
        diffusion = torch.nan_to_num(diffusion, nan=1e-4)
        noise = torch.randn_like(x_t)
        
        # Aplicamos el paso de Euler con dimensiones correctas
        dt_sqrt = torch.sqrt(torch.abs(dt))
        x_next = x_t + drift * (-dt) + diffusion * dt_sqrt * noise
        return x_next
    
    def corrector_step(
        self,
        x_t: Tensor,
        t: Tensor,
        score_model: Callable
    ) -> Tensor:
        """
        Realiza un paso de corrector (basado en Langevin dynamics).
        """
        try:
            with torch.enable_grad():
                x_t.requires_grad_(True)
                score = score_model(x_t, t)
                x_t.requires_grad_(False)
            
            if torch.isnan(score).any():
                score = torch.nan_to_num(score, nan=0.0)
                
            # Estimamos la magnitud del ruido corrector basado en el SNR
            noise_scale = torch.sqrt(torch.tensor(2.0 * self.corrector_snr, device=x_t.device))
            noise = torch.randn_like(x_t)
            
            # Calculamos la norma del score cuidadosamente
            # Usamos un valor pequeño de epsilon para evitar división por cero
            epsilon = 1e-10
            score_norm = torch.norm(score.view(score.shape[0], -1), dim=1, keepdim=True).view(-1, 1, 1, 1)
            score_norm = torch.maximum(score_norm, torch.tensor(epsilon, device=score_norm.device))
            
            # Calculamos el tamaño del paso con broadcasting correcto
            step_size = (self.corrector_snr / (score_norm ** 2)).view(-1, 1, 1, 1)
            step_size = torch.nan_to_num(step_size, nan=1e-10)
            
            # Aplicamos el paso corrector con broadcasting correcto
            sqrt_step = torch.sqrt(step_size)
            x_t_corrected = x_t + step_size * score + noise_scale * sqrt_step * noise
            return x_t_corrected
            
        except IndexError as e:
            if self.verbose:
                print(f"IndexError en corrector_step: {e}. Omitiendo corrección.")
            # Si ocurre un error de índice, simplemente devolvemos x_t sin modificar
            return x_t
    
    def __call__(
        self,
        x_T: Tensor,
        score_model: Callable,
        n_steps: int = 500,
        seed: Optional[int] = None,
        callback: Optional[Callable[[Tensor, int], None]] = None,
        callback_frequency: int = 50,
        guidance: Optional[Callable[[Tensor, Tensor], Tensor]] = None
    ) -> Tensor:

        if seed is not None:
            torch.manual_seed(seed)

        device = x_T.device
        x_t = x_T.clone()

        # Generamos tiempos
        times = torch.linspace(1.0, 1e-3, n_steps+1, device=device)

        iterable = tqdm(
            range(n_steps), desc='Generating') if self.verbose else range(n_steps)

        for i in iterable:
            t_curr = times[i]
            t_next = times[i+1]

            # Creamos tensores de tiempo con dimensiones de batch adecuadas
            batch_size = x_T.shape[0]
            t_batch = torch.full((batch_size,), t_curr, device=device)
            t_next_batch = torch.full((batch_size,), t_next, device=device)

            if torch.isnan(x_t).any() or torch.isinf(x_t).any():
                if self.verbose:
                    print(f"Warning: NaN or Inf values detected in x_t at step {i}")
                x_t = torch.nan_to_num(x_t, nan=0.0, posinf=1.0, neginf=-1.0)

            # Paso 1: Predictor
            try:
                with torch.enable_grad():
                    x_t.requires_grad_(True)
                    score = score_model(x_t, t_batch)
                    x_t.requires_grad_(False)

                if torch.isnan(score).any():
                    if self.verbose:
                        print(f"Warning: NaN values in score at step {i}, t={t_curr}")
                    score = torch.nan_to_num(score, nan=0.0)
            except Exception as e:
                print(f"Error computing score at step {i}, t={t_curr}: {e}")
                score = torch.zeros_like(x_t)
                x_t.requires_grad_(False)
            
            # Aplicamos el paso predictor
            x_t = self.predictor_step(x_t, t_batch, t_next_batch, score)
            
            # Paso 2: Corrector (Langevin MCMC)
            # Aseguramos que el paso corrector maneje correctamente los class labels
            try:
                for j in range(self.corrector_steps):
                    x_t = self.corrector_step(x_t, t_next_batch, score_model)
            except Exception as e:
                if self.verbose:
                    print(f"Error en el paso corrector: {e}. Continuando sin corrección.")
            
            if guidance is not None:
                try:
                    x_t = guidance(x_t, t_next)
                except Exception as e:
                    if self.verbose:
                        print(f"Error en guidance: {e}. Continuando sin aplicar guía.")

            # Estabilización
            x_t = torch.clamp(x_t, -10.0, 10.0)

            # Logging y callbacks
            if self.verbose and (i % callback_frequency == 0 or torch.isnan(x_t).any()):
                print(
                    f"Step {i}: t={t_curr:.3f}, mean={x_t.mean().item():.3f}, std={x_t.std().item():.3f}")

            if callback and i % callback_frequency == 0:
                callback(x_t.detach().clone(), i)

        return x_t