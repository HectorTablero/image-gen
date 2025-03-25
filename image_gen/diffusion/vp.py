from .base import BaseDiffusion


class VariancePreserving(BaseDiffusion):
    """Variance Preserving (VP) diffusion process"""

    def forward_process(self, x, t):
        return None

    def reverse_process(self, x, t):
        return None

    def get_schedule(self, t):
        return None
