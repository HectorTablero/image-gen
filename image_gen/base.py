from .diffusion import BaseDiffusion, VarianceExploding, VariancePreserving, SubVariancePreserving
from .samplers import BaseSampler, EulerMaruyama, ExponentialIntegrator, ODEProbabilityFlow, PredictorCorrector
from .noise import BaseNoiseSchedule, LinearNoiseSchedule, CosineNoiseSchedule
from .score_model import ScoreNet
from typing import Callable, Optional, Union, Literal, List, Tuple, Iterable, Dict, Any
import torch
from torch.optim import Adam
from torch import Tensor
from tqdm.autonotebook import tqdm
import warnings


MODEL_VERSION = 2


class GenerativeModel:
    DIFFUSION_MAP = {
        "ve": VarianceExploding,
        "vp": VariancePreserving,
        "sub-vp": SubVariancePreserving,
    }
    NOISE_SCHEDULE_MAP = {
        "linear": LinearNoiseSchedule,
        "cosine": CosineNoiseSchedule,
    }
    SAMPLER_MAP = {
        "euler-maruyama": EulerMaruyama,
        "exponential": ExponentialIntegrator,
        "ode": ODEProbabilityFlow,
        "predictor-corrector": PredictorCorrector,
    }

    def __init__(self,
                 diffusion: Optional[Union[BaseDiffusion, type,
                                           Literal["ve", "vp", "sub-vp"]]] = "ve",
                 sampler: Optional[Union[BaseSampler, type,
                                         Literal["euler-maruyama", "exponential", "ode", "predictor-corrector"]]] = "euler-maruyama",
                 noise_schedule: Optional[Union[BaseNoiseSchedule,
                                                type, Literal["linear", "cosine"]]] = None,
                 verbose: bool = True) -> None:
        self.model = None
        self.verbose = verbose

        if diffusion is None:
            diffusion = "ve"

        if isinstance(diffusion, str):
            diffusion_key = diffusion.lower()
            try:
                diffusion = GenerativeModel.DIFFUSION_MAP[diffusion_key]
            except KeyError:
                raise ValueError(f"Unknown diffusion string: {diffusion}")

        if sampler is None:
            sampler = "euler-maruyama"

        if isinstance(sampler, str):
            sampler_key = sampler.lower()
            try:
                sampler = GenerativeModel.SAMPLER_MAP[sampler_key]
            except KeyError:
                raise ValueError(f"Unknown sampler string: {sampler}")

        if noise_schedule is None and ((isinstance(diffusion, type) or isinstance(diffusion, BaseDiffusion)) and diffusion.NEEDS_NOISE_SCHEDULE):
            noise_schedule = "linear"

        if isinstance(noise_schedule, str):
            ns_key = noise_schedule.lower()
            try:
                noise_schedule = GenerativeModel.NOISE_SCHEDULE_MAP[ns_key]
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

        self.stored_labels = None
        self._label_map = None
        self.version = MODEL_VERSION

    @property
    def device(self) -> torch.device:
        """Dynamic device property based on model parameters."""
        if self.model is not None:
            return next(self.model.parameters()).device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def version(self) -> int:
        """Model version."""
        return self.version

    @property
    def labels(self) -> List[str]:
        """Stored labels."""
        return self._label_map if self._label_map is not None else []

    def _progress(self, iterable: Iterable, **kwargs: Dict[str, Any]) -> Iterable:
        return tqdm(iterable, **kwargs) if self.verbose else iterable

    def _build_default_model(self, shape: Tuple[int, int, int] = (3, 32, 32)):
        device = self.device  # Creating the ScoreNet changes the device, so this line is necessary
        self.num_c = shape[0]
        self.shape = (shape[1], shape[2])
        self.model = ScoreNet(
            marginal_prob_std=self.diffusion.schedule, num_c=shape[0], num_classes=self.num_classes)
        if self.device.type == "cuda":
            self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(device)

    def loss_function(self, x0: torch.Tensor, eps: float = 1e-5, class_labels: Optional[Tensor] = None) -> torch.Tensor:
        t = torch.rand(x0.shape[0], device=x0.device) * (1.0 - eps) + eps

        xt, noise = self.diffusion.forward_process(x0, t)

        score = self.model(xt, t, class_label=class_labels)

        beta_t = self.diffusion.schedule(t).view(
            x0.shape[0], *([1]*(x0.dim()-1)))
        mse_per_example = torch.sum(
            (beta_t * score + noise)**2, dim=list(range(1, x0.dim())))
        return torch.mean(mse_per_example)

    def train(
        self,
        dataset: Union[
            torch.utils.data.Dataset,
            List[Union[Tensor, Tuple[Tensor, Tensor]]]
        ],
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3
    ) -> None:
        first = dataset[0]

        has_labels = isinstance(
            first, (list, tuple)) and len(first) > 1
        if has_labels:
            all_labels = [
                label if isinstance(label, Tensor) else torch.tensor(label)
                for _, label in dataset
            ]
            all_labels_tensor = torch.cat([lbl.view(-1) for lbl in all_labels])
            unique_labels = sorted(all_labels_tensor.unique().tolist())

            self.num_classes = len(unique_labels)
            self.stored_labels = unique_labels
            self._label_map = {str(label): idx for idx,
                               label in enumerate(unique_labels)}
        else:
            self.num_classes = None

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
                dataloader, desc=f'Epoch {epoch + 1}', leave=False)
            for batch in batch_bar:
                if has_labels:
                    x0, labels = batch[0], batch[1]
                    labels = labels.to(self.device)
                else:
                    x0 = batch
                    labels = None

                x0 = x0.to(self.device)

                optimizer.zero_grad()

                if self.num_classes is not None:
                    loss = self.loss_function(x0, class_labels=labels)
                else:
                    loss = self.loss_function(x0)

                loss.backward()
                optimizer.step()

                avg_loss += loss.item() * x0.shape[0]
                num_items += x0.shape[0]
                batch_bar.set_postfix(loss=loss.item())

            epoch_bar.set_postfix(avg_loss=avg_loss / num_items)

    def set_labels(self, labels: List[str]):
        # Check if model has class conditioning
        if not hasattr(self, 'num_classes') or self.num_classes is None:
            warnings.warn(
                "Model not initialized for class conditioning - labels will have no effect")
            return

        # Check if we have stored numeric labels
        if not hasattr(self, 'stored_labels') or self.stored_labels is None:
            warnings.warn(
                "No class labels stored from training - cannot map string labels")
            return

        # Validate input length
        if len(labels) != len(self.stored_labels):
            raise ValueError(
                f"Length mismatch: got {len(labels)} string labels, "
                f"but model has {len(self.stored_labels)} classes. "
                f"Current numeric labels: {self.stored_labels}"
            )

        # Create new mapping
        self._label_map = {
            string_label: numeric_label
            for numeric_label, string_label in zip(self.stored_labels, labels)
        }

    def class_conditional_score(self, class_labels: Union[int, Tensor], num_samples: int, guidance_scale: float = 3.0) -> Callable[[Tensor, Tensor], Tensor]:
        if class_labels is None:
            return self.model

        processed_labels = None
        if self.num_classes is None:
            warnings.warn(
                "Ignoring class_labels - model not initialized for class conditioning")
            return self.model

        # Convert to tensor and ensure proper type (torch.long)
        if isinstance(class_labels, int):
            class_labels = torch.full(
                (num_samples,), class_labels, dtype=torch.long)
        elif isinstance(class_labels, list):
            class_labels = torch.tensor(class_labels, dtype=torch.long)
        elif isinstance(class_labels, Tensor):
            class_labels = class_labels.long()  # Convert to long if not already
        else:
            raise ValueError(
                "class_labels must be int, list or Tensor")

        class_labels = class_labels.to(self.device)

        # Validate labels
        if hasattr(self, 'stored_labels') and self.stored_labels is not None:
            invalid_mask = ~torch.isin(class_labels, torch.tensor(
                self.stored_labels, device=self.device))
            if invalid_mask.any():
                warnings.warn(
                    f"Invalid labels detected. Valid labels: {self.stored_labels}")
                # Replace invalid with first valid label
                class_labels[invalid_mask] = self.stored_labels[0]

        processed_labels = class_labels.to(self.device)

        def guided_score(x: Tensor, t: Tensor) -> Tensor:
            uncond_score = self.model(x, t, class_label=None)

            # Conditional score - ensure we pass proper labels
            if processed_labels is not None:
                # Ensure we have enough labels for the batch
                if len(processed_labels) != x.shape[0]:
                    # If single label provided, repeat it for batch
                    if len(processed_labels) == 1:
                        current_labels = processed_labels.expand(
                            x.shape[0])
                    else:
                        raise ValueError(
                            "Number of labels must match batch size or be 1")
                else:
                    current_labels = processed_labels

                cond_score = self.model(x, t, class_label=current_labels)
            else:
                cond_score = uncond_score

            return uncond_score + guidance_scale * (cond_score - uncond_score)

        return guided_score

    def generate(self,
                 num_samples: int,
                 n_steps: int = 500,
                 seed: Optional[int] = None,
                 class_labels: Optional[Union[int, Tensor]] = None,
                 progress_callback: Optional[Callable[[
                     Tensor, int], None]] = None
                 ) -> torch.Tensor:
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError(
                "Model not initialized. Please load or train the model first.")

        score_func = self.class_conditional_score(class_labels, num_samples)

        x_T = torch.randn(num_samples, self.num_c, *
                          self.shape, device=self.device)

        self.model.eval()
        with torch.no_grad():
            samples = self.sampler(
                x_T=x_T,
                score_model=score_func,
                n_steps=n_steps,
                seed=seed,
                callback=progress_callback
            )

        self.model.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return samples

    def colorize(self, x: Tensor, n_steps: int = 500,
                 seed: Optional[int] = None,
                 class_labels: Optional[Union[int, Tensor]] = None,
                 progress_callback: Optional[Callable[[Tensor, int], None]] = None) -> Tensor:
        """Colorize grayscale images using YUV-space luminance enforcement"""
        if not hasattr(self, 'num_c') or self.num_c != 3:
            raise ValueError("Colorization requires a 3-channel model")

        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension
        if x.shape[1] == 3:
            y_target = self._rgb_to_grayscale(x)
        elif x.shape[1] == 1:
            y_target = x
        else:
            raise ValueError("Input must be 1 or 3 channels")

        y_target = (y_target - y_target.min()) / \
            (y_target.max() - y_target.min() + 1e-8)

        y_target = y_target.to(self.device).float()
        batch_size, _, h, w = y_target.shape

        with torch.no_grad():
            uv = torch.rand(batch_size, 2, h, w, device=self.device) * \
                0.5 - 0.25
            yuv = torch.cat([y_target, uv], dim=1)
            x_init = self._yuv_to_rgb(yuv)

            t_T = torch.ones(batch_size, device=self.device)
            x_T, _ = self.diffusion.forward_process(x_init, t_T)

        def enforce_luminance(x_t: Tensor, t: Tensor) -> Tensor:
            """Enforce Y channel while preserving UV color information"""
            with torch.no_grad():
                yuv = self._rgb_to_yuv(x_t)

                yuv[:, 0:1] = y_target

                enforced_rgb = self._yuv_to_rgb(yuv)

                alpha = t.item() / self.diffusion.schedule.max_t
                return enforced_rgb * (1 - alpha) + x_t * alpha

        score_func = self.class_conditional_score(class_labels, x.shape[0])

        self.model.eval()
        with torch.no_grad():
            samples = self.sampler(
                x_T=x_T,
                score_model=score_func,
                n_steps=n_steps,
                guidance=enforce_luminance,
                callback=progress_callback,
                seed=seed
            )

        self.model.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return samples

    @staticmethod
    def _rgb_to_yuv(img: Tensor) -> Tensor:
        """Convert RGB tensor (B,3,H,W) to YUV (B,3,H,W)"""
        r, g, b = img.chunk(3, dim=1)
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = 0.492 * (b - y) + 0.5
        v = 0.877 * (r - y) + 0.5
        return torch.cat([y, u, v], dim=1)

    @staticmethod
    def _yuv_to_rgb(yuv: Tensor) -> Tensor:
        """Convert YUV tensor (B,3,H,W) to RGB (B,3,H,W)"""
        y, u, v = yuv.chunk(3, dim=1)
        u = (u - 0.5) / 0.492
        v = (v - 0.5) / 0.877

        r = y + v
        b = y + u
        g = (y - 0.299 * r - 0.114 * b) / 0.587
        return torch.clamp(torch.cat([r, g, b], dim=1), 0.0, 1.0)

    def imputation(self, x: Tensor, mask: Tensor, n_steps: int = 500,
                   seed: Optional[int] = None,
                   class_labels: Optional[Union[int, Tensor]] = None,
                   progress_callback: Optional[Callable[[Tensor, int], None]] = None) -> Tensor:
        """Image inpainting with mask-guided generation and proper normalization"""
        if x.shape[-2:] != mask.shape[-2:]:
            raise ValueError(
                "Image and mask must have same spatial dimensions")
        if mask.shape[1] != 1:
            raise ValueError("Mask must be single-channel")

        batch_size, channels, _, _ = x.shape

        input_min = x.min()
        input_max = x.max()

        x_normalized = (x - input_min) / (input_max - input_min + 1e-8) * 2 - 1

        generate_mask = mask.to(self.device).bool()
        generate_mask = generate_mask.expand(-1, channels, -1, -1)
        preserve_mask = ~generate_mask

        with torch.no_grad():
            x_init = x_normalized.clone().to(self.device)
            noise = torch.randn_like(x_normalized)
            x_T = torch.where(generate_mask, noise, x_init)
            t_T = torch.ones(batch_size, device=self.device)
            x_T, _ = self.diffusion.forward_process(x_T, t_T)

        def inpaint_guidance(x_t: Tensor, _: Tensor) -> Tensor:
            with torch.no_grad():
                return torch.where(preserve_mask, x_normalized, x_t)

        score_func = self.class_conditional_score(class_labels, batch_size)

        self.model.eval()
        with torch.no_grad():
            samples_normalized = self.sampler(
                x_T=x_T,
                score_model=score_func,
                n_steps=n_steps,
                guidance=inpaint_guidance,
                callback=progress_callback,
                seed=seed
            )

        combined_normalized = torch.where(
            generate_mask, samples_normalized, x_normalized)
        result = (combined_normalized + 1) / 2 * \
            (input_max - input_min) + input_min

        self.model.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    def save(self, path: str):
        save_data = {
            'model_state': self.model.state_dict(),
            'shape': self.shape,
            'diffusion_type': self.diffusion.__class__.__name__.lower(),
            'sampler_type': self.sampler.__class__.__name__.lower(),
            'num_channels': self.num_c,
            'stored_labels': self.stored_labels,
            'label_map': self._label_map,
            'model_version': MODEL_VERSION,
        }

        if hasattr(self.diffusion, 'config'):
            save_data['diffusion_config'] = self.diffusion.config()

        if self.diffusion.NEEDS_NOISE_SCHEDULE:
            save_data['noise_schedule_type'] = self.diffusion.schedule.__class__.__name__.lower()
            if hasattr(self.diffusion.schedule, 'config'):
                save_data['noise_schedule_config'] = self.diffusion.schedule.config()

        torch.save(save_data, path)

    def _rebuild_diffusion(self, checkpoint: Dict[str, Any]):
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

    def _rebuild_noise_schedule(self, checkpoint: Dict[str, Any]) -> BaseNoiseSchedule:
        schedule_map = {
            LinearNoiseSchedule.__name__.lower(): LinearNoiseSchedule,
            CosineNoiseSchedule.__name__.lower(): CosineNoiseSchedule,
        }

        schedule_type = checkpoint.get('noise_schedule_type', 'linear')
        schedule_cls = schedule_map[schedule_type]
        config = checkpoint.get('noise_schedule_config', {})
        return schedule_cls(**config)

    def _rebuild_sampler(self, checkpoint: Dict[str, Any]):
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
        # TODO: Version-based model loading

        self.model = None

        checkpoint = torch.load(path)

        self._rebuild_diffusion(checkpoint)
        self._rebuild_sampler(checkpoint)

        self.stored_labels = checkpoint.get('stored_labels')
        self.num_classes = len(
            self.stored_labels) if self.stored_labels is not None else None
        self._label_map = checkpoint.get('label_map')
        self.version = checkpoint.get('model_version')

        checkpoint_channels = checkpoint.get(
            'num_channels', 1)  # Default to grayscale
        self.shape = checkpoint.get('shape', (32, 32))

        self._build_default_model(shape=(checkpoint_channels, *self.shape))

        try:
            self.model.load_state_dict(checkpoint['model_state'])
        except RuntimeError as e:
            try:
                new_state_dict = {
                    k.replace('module.', ''): v for k, v in checkpoint['model_state'].items()}
                self.model.load_state_dict(new_state_dict)
            except RuntimeError as e2:
                print(f"Warning: Failed to load model state with error: {e}")

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
