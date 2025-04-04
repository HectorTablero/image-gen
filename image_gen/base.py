from .diffusion import BaseDiffusion, VarianceExploding, VariancePreserving, SubVariancePreserving
from .samplers import BaseSampler, EulerMaruyama, ExponentialIntegrator, ODEProbabilityFlow, PredictorCorrector
from .noise import BaseNoiseSchedule, LinearNoiseSchedule, CosineNoiseSchedule
from .score_model import ScoreNet
from typing import Callable, Optional, Union, Literal
import torch
from torch.optim import Adam
from torch import Tensor
from tqdm.autonotebook import tqdm
import warnings


class GenerativeModel:
    def __init__(self,
                 diffusion: Optional[Union[BaseDiffusion, type,
                                           Literal["ve", "vp", "sub-vp"]]] = "ve",
                 sampler: Optional[Union[BaseSampler, type,
                                         Literal["euler-maruyama", "exponential", "ode", "predictor-corrector"]]] = "euler-maruyama",
                 noise_schedule: Optional[Union[BaseNoiseSchedule,
                                                type, Literal["linear", "cosine"]]] = None,
                 verbose: bool = True) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.verbose = verbose

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
        self.sampler.verbose = verbose

    def _progress(self, iterable, **kwargs):
        return tqdm(iterable, **kwargs) if self.verbose else iterable

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

        epoch_bar = self._progress(range(epochs), desc='Training')
        for epoch in epoch_bar:
            avg_loss = 0.0
            num_items = 0

            batch_bar = self._progress(
                dataloader, desc=f'Epoch {epoch+1}', leave=False)
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

    def generate(self,
                 num_samples: int,
                 n_steps: int = 500,
                 seed: int = 0,
                 selected_class: Optional[int] = None,
                 progress_callback: Optional[Callable[[Tensor, int], None]] = None) -> torch.Tensor:
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError(
                "Model not initialized. Please load or train the model first.")

        # TODO: Add class-conditional generation support

        device = next(self.model.parameters()).device
        x_T = torch.randn(num_samples, self.num_c, *self.shape, device=device)

        self.model.eval()
        with torch.no_grad():
            samples = self.sampler(
                x_T=x_T,
                score_model=self.model,
                n_steps=n_steps,
                seed=seed,
                callback=progress_callback
            )

        self.model.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return samples

    def colorize(self, x: Tensor) -> Tensor:
        # Check model compatibility
        if not hasattr(self, 'num_c') or self.num_c != 3:
            raise ValueError("Colorization requires a 3-channel model")

        # Convert input to grayscale (Y channel)
        if x.shape[1] == 3:
            y_target = self._rgb_to_grayscale(x)
        elif x.shape[1] == 1:
            y_target = x
        else:
            raise ValueError("Input must be 1 or 3 channels")

        # Move to device
        y_target = y_target.to(self.device)

        # Define guidance function
        lambda_param = 200  # Tune this empirically

        def luminance_guidance(x_t: Tensor, t: Tensor, score: Tensor):
            batch_size = x_t.shape[0]

            # Estimate denoised image (x₀) from current x_t
            with torch.no_grad():
                sigma_t = self.diffusion.schedule(t).view(
                    batch_size, *([1]*(x_t.dim()-1)))
                x0_estimate = x_t + sigma_t**2 * score  # VE-specific

            # Compute luminance loss gradient
            x0_estimate.requires_grad_(True)
            y_pred = self._rgb_to_grayscale(x0_estimate)
            loss = torch.mean((y_pred - y_target)**2, dim=[1, 2, 3])

            # Adjust score with gradient
            grad = torch.autograd.grad(
                outputs=torch.sum(loss),
                inputs=x0_estimate,
                retain_graph=True
            )[0]

            return score + lambda_param * grad

        # Generate with guidance
        return self.generate(
            num_samples=x.shape[0],
            sampler_kwargs={'guidance': luminance_guidance}  # Pass to sampler
        )

    @staticmethod
    def _rgb_to_grayscale(img: Tensor) -> Tensor:
        """Convert RGB tensor (B,3,H,W) to grayscale (B,1,H,W)"""
        return 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]

    def imputation(self, x: Tensor, mask: Tensor) -> Tensor:
        # TODO: Implement imputation logic
        pass

    def save(self, path: str):
        save_data = {
            'model_state': self.model.state_dict(),
            'shape': self.shape,
            'diffusion_type': self.diffusion.__class__.__name__.lower(),
            'sampler_type': self.sampler.__class__.__name__.lower(),
            'num_channels': self.num_c,
        }

        if hasattr(self.diffusion, 'config'):
            save_data['diffusion_config'] = self.diffusion.config()

        if self.diffusion.NEEDS_NOISE_SCHEDULE:
            save_data['noise_schedule_type'] = self.diffusion.schedule.__class__.__name__.lower()
            if hasattr(self.diffusion.schedule, 'config'):
                save_data['noise_schedule_config'] = self.diffusion.schedule.config()

        torch.save(save_data, path)

    def _rebuild_diffusion(self, checkpoint: dict):
        diffusion_map = {
            VarianceExploding.__name__.lower(): VarianceExploding,
            VariancePreserving.__name__.lower(): VariancePreserving,
            SubVariancePreserving.__name__.lower(): SubVariancePreserving,
        }

        diff_type = checkpoint.get('diffusion_type', 've')
        diffusion_cls = diffusion_map[diff_type]

        schedule = None
        if diffusion_cls.NEEDS_NOISE_SCHEDULE:
            schedule = self._rebuild_noise_schedule(checkpoint)

        config = checkpoint.get('diffusion_config', {})

        if diffusion_cls.NEEDS_NOISE_SCHEDULE:
            self.diffusion = diffusion_cls(schedule, **config)
        else:
            self.diffusion = diffusion_cls(**config)

    def _rebuild_noise_schedule(self, checkpoint: dict) -> BaseNoiseSchedule:
        schedule_map = {
            LinearNoiseSchedule.__name__.lower(): LinearNoiseSchedule,
            CosineNoiseSchedule.__name__.lower(): CosineNoiseSchedule,
        }

        schedule_type = checkpoint.get('noise_schedule_type', 'linear')
        schedule_cls = schedule_map[schedule_type]
        config = checkpoint.get('noise_schedule_config', {})
        return schedule_cls(**config)

    def _rebuild_sampler(self, checkpoint: dict):
        sampler_map = {
            EulerMaruyama.__name__.lower(): EulerMaruyama,
            ExponentialIntegrator.__name__.lower(): ExponentialIntegrator,
            ODEProbabilityFlow.__name__.lower(): ODEProbabilityFlow,
            PredictorCorrector.__name__.lower(): PredictorCorrector,
        }

        sampler_type = checkpoint.get('sampler_type', 'euler-maruyama')
        sampler_cls = sampler_map[sampler_type]
        self.sampler = sampler_cls(self.diffusion, verbose=self.verbose)

    def load(self, path: str):
        checkpoint = torch.load(path)

        self._rebuild_diffusion(checkpoint)
        self._rebuild_sampler(checkpoint)

        checkpoint_channels = checkpoint.get(
            'num_channels', 1)  # Default to grayscale
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

    def __str__(self) -> str:
        components = [
            f"Input shape: {getattr(self, 'shape', 'Not initialized')}",
            f"Channels: {getattr(self, 'num_c', 'Not initialized')}",
            f"Diffusion: {str(self.diffusion) if hasattr(self, 'diffusion') else 'None'}",
            f"Sampler: {str(self.sampler) if hasattr(self, 'sampler') else 'None'}"
        ]

        if hasattr(self, 'diffusion') and self.diffusion.NEEDS_NOISE_SCHEDULE:
            components.insert(
                3, f"Noise Schedule: {str(self.diffusion.schedule)}")

        return "GenerativeModel(\n    " + "\n    ".join(components) + "\n)"
