from .diffusion import BaseDiffusion
from .samplers import BaseSampler
from .noise import BaseNoiseSchedule
from .score_model import ScoreNet
from typing import Optional, List
import torch
from torch.optim import Adam
from torch import Tensor
from tqdm.autonotebook import tqdm


class GenerativeModel(torch.nn.Module):
    def __init__(self,
                 diffusion: BaseDiffusion,
                 sampler: BaseSampler,
                 noise_schedule: BaseNoiseSchedule,
                 model: Optional[torch.nn.Module] = None,
                 image_size: tuple = (32, 32)):
        super().__init__()
        self.schedule = noise_schedule
        self.diffusion = diffusion
        self.sampler = sampler
        self.shape = image_size
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = model if model else self._build_default_model()

        if self.device.type == "cuda":
            self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

    def _build_default_model(self) -> ScoreNet:
        return ScoreNet(marginal_prob_std=self.schedule)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t_embed = t.float() / self.schedule.max_t
        t_embed = t_embed.view(-1, 1, 1, 1).expand(-1, 1, *self.shape)
        x = torch.cat([x, t_embed], dim=1)
        return self.model(x)

    def loss_function(self, x0: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        t = torch.rand(x0.shape[0], device=x0.device) * (1.0 - eps) + eps
        t_scaled = t * self.schedule.max_t

        xt, noise = self.diffusion.forward_process(x0, t_scaled)

        score = self.model(xt, t_scaled)

        beta_t = self.diffusion.schedule(t_scaled).view(
            x0.shape[0], *([1]*(x0.dim()-1)))
        mse_per_example = torch.sum(
            (beta_t * score + noise)**2, dim=[1, 2, 3])
        return torch.mean(mse_per_example)

    def train(self, dataset, epochs=100, batch_size=32, lr=1e-4):
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

    def generate(self, num_samples: int) -> torch.Tensor:
        """Generation with automatic device handling"""
        xt = torch.randn(num_samples, 3, *self.shape, device=self.device)

        for t in tqdm(range(self.schedule.max_t),
                      desc='Generating'):
            t_tensor = torch.full((num_samples,), t, device=self.device)
            score = self.model(xt, t_tensor)
            xt = self.sampler.step(xt, t, score)

        return xt.clamp(-1, 1)

    def save(self, path: str):
        torch.save({
            'model_state': self.model.state_dict(),
            # 'diffusion_config': self.diffusion.config(),
            # 'sampler_config': self.sampler.config(),
            'shape': self.shape
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.shape = checkpoint.get('shape', (32, 32))

        if not hasattr(self, 'diffusion'):
            self.diffusion = BaseDiffusion.from_config(
                checkpoint['diffusion_config'])
        if not hasattr(self, 'sampler'):
            self.sampler = BaseSampler.from_config(
                checkpoint['sampler_config'])
