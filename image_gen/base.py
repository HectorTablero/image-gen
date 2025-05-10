"""Generative model implementation for diffusion-based image generation.

This module implements a generative model framework that supports multiple
diffusion processes, sampling methods, and noise schedules for image generation.
"""

from .diffusion import BaseDiffusion, VarianceExploding, VariancePreserving, SubVariancePreserving
from .samplers import BaseSampler, EulerMaruyama, ExponentialIntegrator, ODEProbabilityFlow, PredictorCorrector
from .noise import BaseNoiseSchedule, LinearNoiseSchedule, CosineNoiseSchedule
from .metrics import BaseMetric, BitsPerDimension, FrechetInceptionDistance, InceptionScore
from .utils import CustomClassWrapper, get_class_source
from .score_model import ScoreNet
from typing import Callable, Optional, Union, Literal, List, Tuple, Iterable, Dict, Any
import torch
from torch.optim import Adam
from torch import Tensor
from tqdm.autonotebook import tqdm
import warnings


MODEL_VERSION = 3


# TODO: For custom classes, instead of training with them directly, create
# a wrapper class to check if they have any unallowed dependencies


class GenerativeModel:
    """Generative model for diffusion-based image generation.

    This class implements a framework for training and using generative diffusion models
    for tasks such as image generation, colorization, and inpainting.

    Attributes:
        diffusion: The diffusion process to use.
        sampler: The sampling algorithm for generation.
        model: The underlying score network model.
        device: The device on which the model is running.
    """

    DIFFUSION_MAP = {
        "variance exploding": VarianceExploding,
        "varianceexploding": VarianceExploding,
        "ve": VarianceExploding,
        "variance preserving": VariancePreserving,
        "variancepreserving": VariancePreserving,
        "vp": VariancePreserving,
        "sub-variance preserving": SubVariancePreserving,
        "sub variance preserving": SubVariancePreserving,
        "subvariancepreserving": SubVariancePreserving,
        "sub-vp": SubVariancePreserving,
        "subvp": SubVariancePreserving,
        "svp": SubVariancePreserving,
    }
    NOISE_SCHEDULE_MAP = {
        "linear noise schedule": LinearNoiseSchedule,
        "linearnoiseschedule": LinearNoiseSchedule,
        "linear": LinearNoiseSchedule,
        "lin": LinearNoiseSchedule,
        "l": LinearNoiseSchedule,
        "cosine noise schedule": CosineNoiseSchedule,
        "cosinenoiseschedule": CosineNoiseSchedule,
        "cosine": CosineNoiseSchedule,
        "cos": CosineNoiseSchedule,
        "c": CosineNoiseSchedule,
    }
    SAMPLER_MAP = {
        "euler-maruyama": EulerMaruyama,
        "euler maruyama": EulerMaruyama,
        "eulermaruyama": EulerMaruyama,
        "euler": EulerMaruyama,
        "em": EulerMaruyama,
        "exponential integrator": ExponentialIntegrator,
        "exponentialintegrator": ExponentialIntegrator,
        "exponential": ExponentialIntegrator,
        "exp": ExponentialIntegrator,
        "ode probability flow": ODEProbabilityFlow,
        "odeprobabilityflow": ODEProbabilityFlow,
        "ode flow": ODEProbabilityFlow,
        "ode": ODEProbabilityFlow,
        "predictor-corrector": PredictorCorrector,
        "predictor corrector": PredictorCorrector,
        "predictorcorrector": PredictorCorrector,
        "pred": PredictorCorrector,
    }
    METRIC_MAP = {
        "bits per dimension": BitsPerDimension,
        "bitsperdimension": BitsPerDimension,
        "bpd": BitsPerDimension,
        "fréchet inception distance": FrechetInceptionDistance,
        "frechet inception distance": FrechetInceptionDistance,
        "frechetinceptiondistance": FrechetInceptionDistance,
        "frechet": FrechetInceptionDistance,
        "fréchet": FrechetInceptionDistance,
        "fid": FrechetInceptionDistance,
        "inception score": InceptionScore,
        "inceptionscore": InceptionScore,
        "inception": InceptionScore,
        "is": InceptionScore,
    }

    def __init__(
        self,
        diffusion: Optional[Union[BaseDiffusion, type,
                                  Literal["ve", "vp", "sub-vp", "svp"]]] = "ve",
        sampler: Optional[Union[BaseSampler, type,
                                Literal["euler-maruyama", "euler", "em",
                                        "exponential", "exp", "ode",
                                        "predictor-corrector", "pred"]]] = "euler-maruyama",
        noise_schedule: Optional[Union[BaseNoiseSchedule, type,
                                       Literal["linear", "lin", "cosine", "cos"]]] = None,
        verbose: bool = True
    ) -> None:
        """Initializes the generative model.

        Args:
            diffusion: The diffusion process to use. Can be a string identifier,
                a diffusion class, or a diffusion instance.
            sampler: The sampling algorithm to use. Can be a string identifier,
                a sampler class, or a sampler instance.
            noise_schedule: The noise schedule to use. Only required for diffusion
                processes that need a noise schedule.
            verbose: Whether to display progress bars during generation and training.

        Raises:
            ValueError: If an unknown diffusion or sampler string is provided.
            TypeError: If the diffusion or sampler has an invalid type.
        """
        self._model = None
        self._verbose = verbose
        self._num_classes = None  # Initialize this attribute
        self._stored_labels = None
        self._label_map = None
        self._version = MODEL_VERSION
        self._num_channels = None
        self._shape = None  # Changed from _input_shape to _shape

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

        if noise_schedule is None and ((isinstance(diffusion, type) or
                                        isinstance(diffusion, BaseDiffusion)) and
                                       diffusion.NEEDS_NOISE_SCHEDULE):
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
                        f"{diffusion.__name__} does not require a noise schedule. "
                        f"The provided noise schedule will be ignored.",
                        UserWarning
                    )
                self.diffusion = diffusion()
        else:
            if not diffusion.NEEDS_NOISE_SCHEDULE and noise_schedule is not None:
                warnings.warn(
                    f"{diffusion.__class__.__name__} does not require a noise schedule. "
                    f"The provided noise schedule will be ignored.",
                    UserWarning
                )
            self.diffusion = diffusion

        if isinstance(sampler, type):
            self.sampler = sampler(self.diffusion)
        else:
            self.sampler = sampler
        self.sampler.verbose = verbose

        self._stored_labels = None
        self._label_map = None
        self._version = MODEL_VERSION

        self._num_channels = None
        self._input_shape = None

        self._custom_sampler = None
        self._custom_diffusion = None
        self._custom_schedule = None

    @property
    def device(self) -> torch.device:
        """Device on which the model is running."""
        if self._model is not None:
            return next(self._model.parameters()).device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def version(self) -> int:
        """Version of the model."""
        return self._version

    @property
    def num_channels(self) -> int:
        """Number of input channels (read-only)."""
        return self._num_channels if self._num_channels is not None else 0

    @property
    def shape(self) -> Tuple[int, int]:
        """Spatial dimensions of the input (height, width) (read-only)."""
        return self._shape if self._shape is not None else (0, 0)

    @property
    def stored_labels(self) -> Tuple[Any, ...]:
        """Numeric class labels from training data (read-only)."""
        return tuple(self._stored_labels) if self._stored_labels is not None else ()

    @property
    def num_classes(self) -> Optional[int]:
        """Number of classes (read-only). None if not class-conditional."""
        return self._num_classes

    @property
    def labels(self) -> List[str]:
        """String labels for classes."""
        return self._label_map if self._label_map is not None else []

    @property
    def model(self) -> Optional[ScoreNet]:
        """The underlying score model (read-only)."""
        return self._model

    @property
    def verbose(self) -> bool:
        """Whether to display progress bars during operations."""
        return self._verbose

    @property
    def noise_schedule(self) -> BaseNoiseSchedule:
        """The noise schedule used by the diffusion process."""
        return self.diffusion.schedule if hasattr(self.diffusion, 'schedule') else None

    @verbose.setter
    def verbose(self, value: bool):
        """Sets the verbose flag for the model and sampler.

        Args:
            value: Whether to display progress bars.
        """
        self._verbose = value
        if hasattr(self.sampler, 'verbose'):
            self.sampler.verbose = value

    @property
    def diffusion(self) -> BaseDiffusion:
        """The diffusion process (read-only after training)"""
        return self._diffusion

    @diffusion.setter
    def diffusion(self, value: Union[BaseDiffusion, type, str]):
        """Sets the diffusion process.

        Args:
            value: The diffusion process to use.

        Raises:
            ValueError: If the diffusion is not a subclass of BaseDiffusion.
            TypeError: If the diffusion has an invalid type.
        """
        if self._model is not None:
            warnings.warn(
                "Diffusion cannot be changed after training", UserWarning)
            return

        if isinstance(value, str):
            value = self.DIFFUSION_MAP.get(value.lower(), VarianceExploding)

        if isinstance(value, type):
            if issubclass(value, BaseDiffusion):
                if value.NEEDS_NOISE_SCHEDULE:
                    ns = LinearNoiseSchedule()
                    self._diffusion = value(ns)
                else:
                    self._diffusion = value()
            else:
                raise ValueError("Must subclass BaseDiffusion")
        elif isinstance(value, BaseDiffusion):
            self._diffusion = value
        else:
            raise TypeError("Invalid diffusion type")

    @property
    def sampler(self) -> BaseSampler:
        """The sampling algorithm (always settable)"""
        return self._sampler

    @sampler.setter
    def sampler(self, value: Union[BaseSampler, type, str]):
        """Sets the sampling algorithm.

        Args:
            value: The sampler to use.

        Raises:
            ValueError: If the sampler is not a subclass of BaseSampler.
            TypeError: If the sampler has an invalid type.
        """
        if isinstance(value, str):
            value = self.SAMPLER_MAP.get(value.lower(), EulerMaruyama)

        if isinstance(value, type):
            if issubclass(value, BaseSampler):
                self._sampler = value(self.diffusion, verbose=self.verbose)
            else:
                # Dashboard breaks without this line (wtf?)
                value == issubclass(value, BaseSampler)
                raise ValueError("Must subclass BaseSampler")
        elif isinstance(value, BaseSampler):
            self._sampler = value
            self._sampler.verbose = self.verbose
        else:
            raise TypeError("Invalid sampler type")

        self._sampler.verbose = self.verbose

    def _progress(self, iterable: Iterable, **kwargs: Dict[str, Any]) -> Iterable:
        """Wraps an iterable with a progress bar if verbose is enabled.

        Args:
            iterable: The iterable to wrap.
            **kwargs: Additional arguments to pass to tqdm.

        Returns:
            The wrapped iterable, or the original if verbose is disabled.
        """
        return tqdm(iterable, **kwargs) if self._verbose else iterable

    def _build_default_model(self, shape: Tuple[int, int, int] = (3, 32, 32)):
        """Builds the default score model.

        Args:
            shape: The input shape (channels, height, width).
        """
        device = self.device  # Creating the ScoreNet changes the device, so this line is necessary
        self._num_channels = shape[0]
        self._shape = (shape[1], shape[2])
        self._model = ScoreNet(
            marginal_prob_std=self.diffusion.schedule,
            num_c=shape[0],
            num_classes=self.num_classes
        )
        if self.device.type == "cuda":
            self._model = torch.nn.DataParallel(self.model)
        self._model = self.model.to(device)

    def loss_function(self, x0: torch.Tensor, eps: float = 1e-5,
                      class_labels: Optional[Tensor] = None) -> torch.Tensor:
        """Computes the loss for training the score model.

        Args:
            x0: The input data.
            eps: Small constant to avoid numerical issues.
            class_labels: Class labels for conditional generation.

        Returns:
            The computed loss value.
        """
        t = torch.rand(x0.shape[0], device=x0.device) * (1.0 - eps) + eps
        xt, noise = self.diffusion.forward_process(x0, t)
        score = self.model(xt, t, class_label=class_labels)
        loss_per_example = self.diffusion.compute_loss(score, noise, t)
        return torch.mean(loss_per_example)

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
        """Trains the score model.

        Args:
            dataset: The dataset to train on. Can be a torch Dataset or a list
                of tensors or (tensor, label) tuples.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            lr: Learning rate for the optimizer.
        """
        first = dataset[0]

        has_labels = isinstance(first, (list, tuple)) and len(first) > 1
        if has_labels:
            all_labels = [
                label if isinstance(label, Tensor) else torch.tensor(label)
                for _, label in dataset
            ]
            all_labels_tensor = torch.cat([lbl.view(-1) for lbl in all_labels])
            self._stored_labels = sorted(all_labels_tensor.unique().tolist())

            # Create mapping from original labels to 0-based indices
            self._label_to_index = {
                lbl: idx for idx, lbl in enumerate(self.stored_labels)
            }
            self._num_classes = len(self.stored_labels)

            # Map all labels to indices
            self._mapped_labels = torch.tensor([
                self._label_to_index[lbl.item()]
                for lbl in all_labels_tensor
            ])
        else:
            self._num_classes = None

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
                    x0, original_labels = batch[0], batch[1]
                    # Convert original labels to mapped indices
                    labels = torch.tensor([
                        self._label_to_index[lbl.item()]
                        for lbl in original_labels
                    ], device=self.device)
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
                # batch_bar.set_postfix(loss=loss.item())

            # epoch_bar.set_postfix(avg_loss=avg_loss / num_items)

    def set_labels(self, labels: List[str]) -> None:
        """Sets string labels for the model's classes.

        Args:
            labels: List of string labels, one per class.

        Raises:
            ValueError: If the number of labels doesn't match the number of classes.
        """
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

    def score(self, real: Tensor, generated: Tensor,
              metrics: List[Union[str, BaseMetric]] = ["bpd", "fid", "is"],
              *args: Any, **kwargs: Any) -> Dict[str, float]:
        """Evaluates the model using various metrics.

        Args:
            real: Real data samples.
            generated: Generated data samples.
            metrics: List of metrics to compute. Can be strings or BaseMetric instances.
            *args: Additional arguments for metrics.
            **kwargs: Additional keyword arguments for metrics.

        Returns:
            Dictionary mapping metric names to scores.

        Raises:
            Exception: If metrics is empty or not a list.
        """
        if not isinstance(metrics, list) or len(metrics) == 0:
            raise Exception(
                "Scores must be a non-empty list.")

        calculated_scores = {}
        for score in metrics:
            # Instantiate the class
            if isinstance(score, str) and score.lower() in GenerativeModel.METRIC_MAP:
                score = GenerativeModel.METRIC_MAP[score.lower()](self)
            elif isinstance(score, type):
                score = score(self)

            if not isinstance(score, BaseMetric):
                warnings.warn(f'"{score}" is not a metric, skipping...')
                continue

            if score.name in calculated_scores:
                warnings.warn(
                    f'A score with the name of "{score.name}" has already been calculated, but it will be overwritten.')
            calculated_scores[score.name] = score(
                real, generated, *args, **kwargs)

        return calculated_scores

    def _class_conditional_score(self, class_labels: Union[int, Tensor],
                                 num_samples: int,
                                 guidance_scale: float = 3.0) -> Callable[[Tensor, Tensor], Tensor]:
        """Creates a class-conditional score function.

        Args:
            class_labels: Class labels for conditional generation.
            num_samples: Number of samples to generate.
            guidance_scale: Scale factor for classifier-free guidance.

        Returns:
            A function that computes the score for a given input and time.

        Raises:
            ValueError: If class_labels has an invalid type.
        """
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
            """Computes the guided score for classifier-free guidance.

            Args:
                x: The input tensor.
                t: The time tensor.

            Returns:
                The guided score.
            """
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
                 guidance_scale: float = 3.0,
                 progress_callback: Optional[Callable[[
                     Tensor, int], None]] = None,
                 callback_frequency: int = 50
                 ) -> torch.Tensor:
        """Generates samples from the model.

        Args:
            num_samples: Number of samples to generate.
            n_steps: Number of sampling steps.
            seed: Random seed for reproducibility.
            class_labels: Class labels for conditional generation.
            guidance_scale: Scale factor for classifier-free guidance.
            progress_callback: Function to call with intermediate results.
            callback_frequency: How often to call the progress callback.

        Returns:
            The generated samples.

        Raises:
            ValueError: If the model is not initialized.
        """
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError(
                "Model not initialized. Please load or train the model first.")

        score_func = self._class_conditional_score(
            class_labels, num_samples, guidance_scale=guidance_scale)

        x_T = torch.randn(num_samples, self.num_channels, *
                          self.shape, device=self.device)

        self.model.eval()
        with torch.no_grad():
            samples = self.sampler(
                x_T=x_T,
                score_model=score_func,
                n_steps=n_steps,
                seed=seed,
                callback=progress_callback,
                callback_frequency=callback_frequency
            )

        self.model.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return samples

    def colorize(self, x: Tensor, n_steps: int = 500,
                 seed: Optional[int] = None,
                 class_labels: Optional[Union[int, Tensor]] = None,
                 progress_callback: Optional[Callable[[Tensor, int], None]] = None) -> Tensor:
        """Colorizes grayscale images using YUV-space luminance enforcement.

        Args:
            x: Grayscale input image(s).
            n_steps: Number of sampling steps.
            seed: Random seed for reproducibility.
            class_labels: Class labels for conditional generation.
            progress_callback: Function to call with intermediate results.

        Returns:
            The colorized images.

        Raises:
            ValueError: If the model doesn't have 3 channels or the input has invalid shape.
        """
        if not hasattr(self, 'num_channels') or self.num_channels != 3:
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
            """Enforces Y channel while preserving UV color information.

            Args:
                x_t: Current RGB image.
                t: Current time step.

            Returns:
                Modified RGB image with enforced Y channel.
            """
            with torch.no_grad():
                yuv = self._rgb_to_yuv(x_t)
                yuv[:, 0:1] = y_target
                enforced_rgb = self._yuv_to_rgb(yuv)
                alpha = t.item() / n_steps
                return enforced_rgb * (1 - alpha) + x_t * alpha

        score_func = self._class_conditional_score(class_labels, x.shape[0])

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
    def _rgb_to_grayscale(img: Tensor) -> Tensor:
        """Convert RGB image tensor to grayscale.

        Args:
            img: Input tensor (B, 3, H, W) in range [0,1] or [-1,1]

        Returns:
            Grayscale tensor (B, 1, H, W)
        """
        if img.min() < 0:  # If in [-1,1] range, normalize to [0,1]
            img = (img + 1) / 2

        # Use standard RGB to grayscale conversion weights
        gray = 0.2989 * img[:, 0] + 0.5870 * img[:, 1] + 0.1140 * img[:, 2]
        return gray.unsqueeze(1)  # Add channel dimension

    @staticmethod
    def _rgb_to_yuv(img: Tensor) -> Tensor:
        """Converts RGB tensor (B,3,H,W) to YUV (B,3,H,W).

        Args:
            img: RGB image tensor.

        Returns:
            YUV image tensor.
        """
        r, g, b = img.chunk(3, dim=1)
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = 0.492 * (b - y) + 0.5
        v = 0.877 * (r - y) + 0.5
        return torch.cat([y, u, v], dim=1)

    @staticmethod
    def _yuv_to_rgb(yuv: Tensor) -> Tensor:
        """Converts YUV tensor (B,3,H,W) to RGB (B,3,H,W).

        Args:
            yuv: YUV image tensor.

        Returns:
            RGB image tensor.
        """
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
        """Performs image inpainting with mask-guided generation.

        Args:
            x: Input image(s) with missing regions.
            mask: Binary mask where 1 indicates pixels to generate (missing regions).
            n_steps: Number of sampling steps.
            seed: Random seed for reproducibility.
            class_labels: Class labels for conditional generation.
            progress_callback: Function to call with intermediate results.

        Returns:
            Inpainted image(s).

        Raises:
            ValueError: If image and mask dimensions don't match.
        """
        if x.shape[-2:] != mask.shape[-2:]:
            raise ValueError(
                "Image and mask must have same spatial dimensions")
        if mask.shape[1] != 1:
            raise ValueError("Mask must be single-channel")

        batch_size, original_channels, _, _ = x.shape

        input_min = x.min()
        input_max = x.max()

        x_normalized = (x - input_min) / (input_max - input_min + 1e-8) * 2 - 1
        x_normalized = x_normalized.to(self.device)

        # Convert to grayscale if model expects 1 channel but input has more
        if self.num_channels == 1 and original_channels != 1:
            x_normalized = x_normalized.mean(dim=1, keepdim=True)

        generate_mask = mask.to(self.device).bool()
        generate_mask = generate_mask.expand(-1,
                                             1, -1, -1).to(self.device)
        preserve_mask = ~generate_mask

        with torch.no_grad():
            x_init = x_normalized.clone().to(self.device)
            noise = torch.randn_like(x_normalized).to(self.device)
            x_T = torch.where(generate_mask, noise, x_init)
            t_T = torch.ones(batch_size, device=self.device)
            x_T, _ = self.diffusion.forward_process(x_T, t_T)

        def inpaint_guidance(x_t: Tensor, t: Tensor) -> Tensor:
            """Preserves known pixels in the image during sampling."""
            with torch.no_grad():
                return torch.where(preserve_mask, x_normalized, x_t)

        score_func = self._class_conditional_score(class_labels, batch_size)

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

    def save(self, path: str) -> None:
        """Saves the model to the specified path.

        Args:
            path: Path where to save the model.
        """
        save_data = {
            'model_state': self.model.state_dict(),
            'shape': self.shape,
            'diffusion_type': self.diffusion.__class__.__name__.lower(),
            'sampler_type': self.sampler.__class__.__name__.lower(),
            'num_channels': self.num_channels,
            'stored_labels': self.stored_labels,
            'label_map': self._label_map,
            'model_version': MODEL_VERSION,
        }

        if hasattr(self.diffusion, 'config'):
            save_data['diffusion_config'] = self.diffusion.config()
        if save_data["diffusion_type"] not in GenerativeModel.DIFFUSION_MAP:
            save_data["diffusion_code"] = get_class_source(
                self.diffusion.__class__)

        if self.diffusion.NEEDS_NOISE_SCHEDULE:
            save_data['noise_schedule_type'] = self.diffusion.schedule.__class__.__name__.lower()
            if hasattr(self.diffusion.schedule, 'config'):
                save_data['noise_schedule_config'] = self.diffusion.schedule.config()
            if save_data["noise_schedule_type"] not in GenerativeModel.NOISE_SCHEDULE_MAP:
                save_data["noise_schedule_code"] = get_class_source(
                    self.noise_schedule.__class__)

        if hasattr(self.sampler, 'config'):
            save_data['sampler_config'] = self.sampler.config()
        if save_data["sampler_type"] not in GenerativeModel.SAMPLER_MAP:
            save_data["sampler_code"] = get_class_source(
                self.sampler.__class__)

        torch.save(save_data, path)

    def _rebuild_diffusion(self, checkpoint: Dict[str, Any], unsafe: bool = False):
        """Rebuilds the diffusion process from a checkpoint.

        Args:
            checkpoint: The checkpoint data.
            unsafe: Whether to allow loading custom code.

        Raises:
            Exception: If the checkpoint contains custom code and unsafe is False.
        """
        default_diffusion = VarianceExploding.__name__.lower()
        diffusion_type = checkpoint.get("diffusion_type", default_diffusion)
        diffusion_cls = GenerativeModel.DIFFUSION_MAP.get(diffusion_type)

        if diffusion_cls is None:
            diffusion_code = checkpoint.get("diffusion_code")
            if diffusion_type != default_diffusion and diffusion_code is not None:
                if unsafe:
                    self._custom_diffusion = diffusion_code
                    diffusion_cls = lambda *args, **kwargs: CustomClassWrapper(
                        diffusion_code, *args, **kwargs)
                    warnings.warn(
                        "This model has been instantiated with a custom diffuser. "
                        "Please verify the safety of the code before calling any methods "
                        "of the GenerativeModel. It can be viewed with "
                        "GenerativeModel.show_custom_code(), and won't be run until needed.")
                else:
                    raise Exception(
                        "The saved model uses a custom diffuser, which is not allowed for "
                        "safety reasons. If you want to load the custom class, use "
                        "model.load(path, override=True, unsafe=True).")

        schedule = self._rebuild_noise_schedule(checkpoint, unsafe=unsafe)
        config = checkpoint.get('diffusion_config', {})
        self._diffusion = diffusion_cls(schedule, **config)

    def _rebuild_noise_schedule(self, checkpoint: Dict[str, Any], unsafe: bool = False) -> BaseNoiseSchedule:
        """Rebuilds the noise schedule from a checkpoint.

        Args:
            checkpoint: The checkpoint data.
            unsafe: Whether to allow loading custom code.

        Returns:
            The rebuilt noise schedule.

        Raises:
            Exception: If the checkpoint contains custom code and unsafe is False.
        """
        default_schedule = LinearNoiseSchedule.__name__.lower()
        schedule_type = checkpoint.get("noise_schedule_type", default_schedule)
        schedule_cls = GenerativeModel.NOISE_SCHEDULE_MAP.get(schedule_type)

        if schedule_cls is None:
            schedule_code = checkpoint.get("noise_schedule_code")
            if schedule_type != default_schedule and schedule_code is not None:
                if unsafe:
                    self._custom_schedule = schedule_code
                    schedule_cls = lambda *args, **kwargs: CustomClassWrapper(
                        schedule_code, *args, **kwargs)
                    warnings.warn(
                        "This model has been instantiated with a custom schedule. "
                        "Please verify the safety of the code before calling any methods "
                        "of the GenerativeModel. It can be viewed with "
                        "GenerativeModel.show_custom_code(), and won't be run until needed.")
                else:
                    raise Exception(
                        "The saved model uses a custom schedule, which is not allowed for "
                        "safety reasons. If you want to load the custom class, use "
                        "model.load(path, override=True, unsafe=True).")

        config = checkpoint.get('noise_schedule_config', {})
        return schedule_cls(**config)

    def _rebuild_sampler(self, checkpoint: Dict[str, Any], unsafe: bool = False):
        """Rebuilds the sampler from a checkpoint.

        Args:
            checkpoint: The checkpoint data.
            unsafe: Whether to allow loading custom code.

        Raises:
            Exception: If the checkpoint contains custom code and unsafe is False.
        """
        default_sampler = EulerMaruyama.__name__.lower()
        sampler_type = checkpoint.get("sampler_type", default_sampler)
        sampler_cls = GenerativeModel.SAMPLER_MAP.get(sampler_type)

        if sampler_cls is None:
            sampler_code = checkpoint.get("sampler_code")
            if sampler_type != default_sampler and sampler_code is not None:
                if unsafe:
                    self._custom_sampler = sampler_code
                    sampler_cls = lambda *args, **kwargs: CustomClassWrapper(
                        sampler_code, *args, **kwargs)
                    warnings.warn(
                        "This model has been instantiated with a custom sampler. "
                        "Please verify the safety of the code before calling any methods "
                        "of the GenerativeModel. It can be viewed with "
                        "GenerativeModel.show_custom_code(), and won't be run until needed.")
                else:
                    raise Exception(
                        "The saved model uses a custom sampler, which is not allowed for "
                        "safety reasons. If you want to load the custom class, use "
                        "model.load(path, override=True, unsafe=True).")

        if self._sampler.__class__ != sampler_cls:
            warnings.warn(
                f"The model was initialized with sampler {self._sampler.__class__.__name__}, "
                f"but the saved model has {sampler_cls.__name__}. The sampler will be set to "
                f"{sampler_cls.__name__}. If you don't want this behaviour, use "
                f"model.load(path, override=False)."
            )
        config = checkpoint.get('sampler_config', {})
        self._sampler = sampler_cls(
            self.diffusion, **config, verbose=self._verbose)

    def get_custom_code(self) -> dict:
        """Returns any custom code components used by the model.

        Returns:
            Dictionary mapping component names to their source code.
        """
        custom_components = {}

        if self._custom_diffusion is not None:
            custom_components["diffusion"] = self._custom_diffusion
        if self._custom_schedule is not None:
            custom_components["noise_schedule"] = self._custom_schedule
        if self._custom_sampler is not None:
            custom_components["sampler"] = self._custom_sampler

        return custom_components

    def load(self, path: str, override: bool = True, unsafe: bool = False, device: Literal["cpu", "cuda"] = "cuda") -> None:
        """Loads a saved model from the specified path.

        Args:
            path: Path to the saved model file.
            override: If True, overwrites the current sampler with the saved one.
            unsafe: If True, allows loading custom code components (potentially unsafe).

        Raises:
            RuntimeError: If the model state dictionary cannot be loaded properly.
        """
        self._model = None

        # Determine the device to load the checkpoint
        map_location = 'cuda' if (
            device == "cuda" and torch.cuda.is_available()) else 'cpu'
        try:
            checkpoint = torch.load(path, map_location=map_location)
        except RuntimeError as e:
            return self.load(path, override=override, unsafe=unsafe, device="cpu")
        self._version = checkpoint.get('model_version')

        self._custom_sampler = None
        self._custom_diffusion = None
        self._custom_schedule = None
        self._rebuild_diffusion(checkpoint, unsafe=unsafe)
        if override:
            self._rebuild_sampler(checkpoint, unsafe=unsafe)

        self._stored_labels = checkpoint.get('stored_labels')
        self._num_classes = (
            len(self.stored_labels) if self.stored_labels is not None else None
        )
        self._label_map = checkpoint.get('label_map')

        # Default to grayscale if channels not specified
        checkpoint_channels = checkpoint.get('num_channels', 1)
        self._shape = checkpoint.get('shape', (32, 32))

        self._build_default_model(shape=(checkpoint_channels, *self._shape))

        try:
            # Load only keys that exist in both models
            model_dict = self.model.state_dict()
            # Filter checkpoint keys that exist in the current model
            pretrained_dict = {
                k: v
                for k, v in checkpoint['model_state'].items()
                if k in model_dict
            }
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict, strict=False)
        except RuntimeError as original_error:
            try:
                # Try with keys without "module." prefix (happens with DataParallel)
                new_state_dict = {
                    k.replace('module.', ''): v
                    for k, v in checkpoint['model_state'].items()
                }
                model_dict = self.model.state_dict()
                pretrained_dict = {
                    k: v
                    for k, v in new_state_dict.items()
                    if k in model_dict
                }
                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict, strict=False)
            except RuntimeError as secondary_error:
                # Log both errors for better debugging
                error_msg = (
                    f"Failed to load model state. Original error: {original_error}. "
                    f"Secondary error: {secondary_error}"
                )
                print(f"Warning: {error_msg}")

    def _class_only_load(self, path: str) -> None:
        """Loads the components of the model from the specified path.
        Useful for getting data without loading the entire model.

        Args:
            path: Path to the saved model file.
        """
        self._model = None

        # Determine the device to load the checkpoint
        checkpoint = torch.load(path, map_location="cpu")
        self._version = checkpoint.get('model_version')

        self._custom_sampler = None
        self._custom_diffusion = None
        self._custom_schedule = None
        self._rebuild_diffusion(checkpoint, unsafe=True)

        self._stored_labels = checkpoint.get('stored_labels')
        self._num_classes = (
            len(self.stored_labels) if self.stored_labels is not None else None
        )
        self._label_map = checkpoint.get('label_map')

        # Default to grayscale if channels not specified
        checkpoint_channels = checkpoint.get('num_channels', 1)
        self._shape = checkpoint.get('shape', (32, 32))
