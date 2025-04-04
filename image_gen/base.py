from .diffusion import BaseDiffusion, VarianceExploding, VariancePreserving, SubVariancePreserving
from .samplers import BaseSampler, EulerMaruyama, ExponentialIntegrator, ODEProbabilityFlow, PredictorCorrector
from .noise import BaseNoiseSchedule, LinearNoiseSchedule, CosineNoiseSchedule
from .score_model import ScoreNet
from typing import Optional, Union, Literal
import torch
from torch.optim import Adam
from torch import Tensor
from tqdm.autonotebook import tqdm
import warnings


class GenerativeModel(torch.nn.Module):
    def __init__(self,
                 diffusion: Optional[Union[BaseDiffusion, type,
                                           Literal["ve", "vp", "sub-vp"]]] = "ve",
                 sampler: Optional[Union[BaseSampler, type,
                                         Literal["euler-maruyama", "exponential", "ode", "predictor-corrector"]]] = "euler-maruyama",
                 noise_schedule: Optional[Union[BaseNoiseSchedule, type, Literal["linear", "cosine"]]] = None):
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        diffusion_map = {
            "ve": VarianceExploding,
            "vp": VariancePreserving,
            "sub-vp": SubVariancePreserving,
        }
        noise_schedule_map = {
            "linear": LinearNoiseSchedule,
            "cosine": CosineNoiseSchedule,
        }
        sampler_map = {
            "euler-maruyama": EulerMaruyama,
            "exponential": ExponentialIntegrator,
            "ode": ODEProbabilityFlow,
            "predictor-corrector": PredictorCorrector,
        }

        if diffusion is None:
            diffusion = "ve"

        if isinstance(diffusion, str):
            diffusion_key = diffusion.lower()
            try:
                diffusion = diffusion_map[diffusion_key]
            except KeyError:
                raise ValueError(f"Unknown diffusion string: {diffusion}")

        if sampler is None:
            sampler = "euler-maruyama"

        if isinstance(sampler, str):
            sampler_key = sampler.lower()
            try:
                sampler = sampler_map[sampler_key]
            except KeyError:
                raise ValueError(f"Unknown sampler string: {sampler}")

        if noise_schedule is None and ((isinstance(diffusion, type) or isinstance(diffusion, BaseDiffusion)) and diffusion.NEEDS_NOISE_SCHEDULE):
            noise_schedule = "linear"

        if isinstance(noise_schedule, str):
            ns_key = noise_schedule.lower()
            try:
                noise_schedule = noise_schedule_map[ns_key]
            except KeyError:
                raise ValueError(
                    f"Unknown noise_schedule string: {noise_schedule}")

        if isinstance(diffusion, type):
            if diffusion.NEEDS_NOISE_SCHEDULE:
                if isinstance(noise_schedule, type):
                    ns_inst = noise_schedule()
                else:
                    ns_inst = noise_schedule
                self.diffusion = diffusion(ns_inst)
            else:
                if noise_schedule is not None:
                    warnings.warn(
                        f"{diffusion.__name__} does not require a noise schedule. The provided noise schedule will be ignored.",
                        UserWarning
                    )
                self.diffusion = diffusion()
        else:
            if not diffusion.NEEDS_NOISE_SCHEDULE and noise_schedule is not None:
                warnings.warn(
                    f"{diffusion.__class__.__name__} does not require a noise schedule. The provided noise schedule will be ignored.",
                    UserWarning
                )
            self.diffusion = diffusion

        if isinstance(sampler, type):
            self.sampler = sampler(self.diffusion)
        else:
            self.sampler = sampler

    def _build_default_model(self, shape=(3, 32, 32)) -> ScoreNet:
        self.num_c = shape[0]
        self.shape = (shape[1], shape[2])
        self.model = ScoreNet(
            marginal_prob_std=self.diffusion.schedule, num_c=shape[0])
        if self.device.type == "cuda":
            self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.model(x, t)

    def loss_function(self, x0: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        t = torch.rand(x0.shape[0], device=x0.device) * (1.0 - eps) + eps

        xt, noise = self.diffusion.forward_process(x0, t)

        score = self.model(xt, t)

        beta_t = self.diffusion.schedule(t).view(
            x0.shape[0], *([1]*(x0.dim()-1)))
        mse_per_example = torch.sum(
            (beta_t * score + noise)**2, dim=list(range(1, x0.dim())))
        return torch.mean(mse_per_example)

    def train(self, dataset, epochs=100, batch_size=32, lr=1e-3):
        first = dataset[0]
        first = first[0] if isinstance(first, (list, tuple)) else first
        self._build_default_model(shape=first.shape)
        optimizer = Adam(self.model.parameters(), lr=lr)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True)

        epoch_bar = tqdm(range(epochs), desc='Training')
        for epoch in epoch_bar:
            avg_loss = 0.0
            num_items = 0

            batch_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}', leave=False)
            for batch in batch_bar:
                x0 = batch[0] if isinstance(batch, (list, tuple)) else batch
                x0 = x0.to(self.device)

                optimizer.zero_grad()
                loss = self.loss_function(x0)
                loss.backward()
                optimizer.step()

                avg_loss += loss.item() * x0.shape[0]
                num_items += x0.shape[0]
                batch_bar.set_postfix(loss=loss.item())

            epoch_bar.set_postfix(avg_loss=avg_loss/num_items)

    def generate(self, num_samples: int, n_steps: int = 500, seed: int = 0) -> torch.Tensor:
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError(
                "Model not initialized. Please load or train the model first.")

        device = next(self.model.parameters()).device
        x_T = torch.randn(num_samples, self.num_c, *self.shape, device=device)

        self.model.eval()
        with torch.no_grad():
            samples = self.sampler(
                x_T=x_T,
                score_model=self.model,
                n_steps=n_steps,
                seed=seed
            )

        self.model.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (samples.clamp(-1, 1) + 1) / 2

    def save(self, path: str):
        torch.save({
            'model_state': self.model.state_dict(),
            # 'diffusion_config': self.diffusion.config(),
            # 'sampler_config': self.sampler.config(),
            'shape': self.shape
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path)
        checkpoint_channels = checkpoint.get(
            'num_channels', 1)  # Default to 1 if not specified
        self.shape = checkpoint.get('shape', (32, 32))

        # Check if we need to rebuild the model with the correct number of channels
        if not hasattr(self, 'model') or not hasattr(self, 'num_c') or self.num_c != checkpoint_channels:
            print(
                f"Rebuilding model to match checkpoint channels: {checkpoint_channels}")
            self._build_default_model(shape=(checkpoint_channels, *self.shape))

        try:
            # First try regular loading
            self.model.load_state_dict(checkpoint['model_state'])
        except RuntimeError as e:
            print(f"Warning: Failed to load model state with error: {e}")
            print("Attempting to load with channel adaptation...")

            # If that fails, try partial loading with adaptation
            self._load_with_channel_adaptation(checkpoint['model_state'])

        if not hasattr(self, 'diffusion'):
            self.diffusion = BaseDiffusion.from_config(
                checkpoint['diffusion_config'])
        if not hasattr(self, 'sampler'):
            self.sampler = BaseSampler.from_config(
                checkpoint['sampler_config'])

    def _load_with_channel_adaptation(self, state_dict):
        """Load state dict with adaptation for different channel counts."""
        current_state = self.model.state_dict()

        # Filter and potentially adapt weights
        adapted_state = {}
        for key, checkpoint_param in state_dict.items():
            if key in current_state:
                current_param = current_state[key]

                # If shapes match exactly, use as is
                if checkpoint_param.shape == current_param.shape:
                    adapted_state[key] = checkpoint_param
                    continue

                # Handle first conv layer (input channels adaptation)
                if 'conv1.weight' in key:
                    if current_param.shape[1] > checkpoint_param.shape[1]:
                        # RGB model loading grayscale weights
                        # Repeat the grayscale channel across RGB
                        adapted_param = checkpoint_param.repeat(
                            1, current_param.shape[1], 1, 1)
                        # Normalize to maintain the same magnitude
                        adapted_param = adapted_param / current_param.shape[1]
                        adapted_state[key] = adapted_param
                    else:
                        # Skip this case (can't adapt from RGB to grayscale)
                        print(
                            f"Skipping {key}: can't adapt from more to fewer channels")

                # Handle final conv/tconv layer (output channels adaptation)
                elif 'tconv1.weight' in key or 'tconv1.bias' in key:
                    if 'weight' in key:
                        if current_param.shape[0] > checkpoint_param.shape[0]:
                            # Repeat the single channel for RGB output
                            adapted_param = checkpoint_param.repeat(
                                current_param.shape[0], 1, 1, 1)
                            adapted_state[key] = adapted_param
                        else:
                            # Use just the first channel if going from RGB to grayscale
                            adapted_state[key] = checkpoint_param[:current_param.shape[0]]
                    elif 'bias' in key:
                        if current_param.shape[0] > checkpoint_param.shape[0]:
                            # Repeat bias for RGB
                            adapted_state[key] = checkpoint_param.repeat(
                                current_param.shape[0])
                        else:
                            # Use subset of bias values
                            adapted_state[key] = checkpoint_param[:current_param.shape[0]]

                # For other layers, if dimensions don't match, skip
                else:
                    print(
                        f"Skipping {key}: shape mismatch - checkpoint {checkpoint_param.shape} vs model {current_param.shape}")
            else:
                print(f"Parameter {key} not found in current model")

        # Load the adapted state dict
        if adapted_state:
            # Use strict=False to load partial state dict
            self.model.load_state_dict(adapted_state, strict=False)
            print(
                f"Loaded {len(adapted_state)}/{len(state_dict)} parameters with adaptation")
