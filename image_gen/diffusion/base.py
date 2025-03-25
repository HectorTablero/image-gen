from abc import ABC, abstractmethod


class BaseDiffusion(ABC):
    """Abstract base class for diffusion processes"""

    @abstractmethod
    def forward_process(self, x, t):
        """Apply forward diffusion process"""
        pass

    @abstractmethod
    def reverse_process(self, x, t):
        """Apply reverse diffusion process"""
        pass

    @abstractmethod
    def get_schedule(self, t):
        """Get noise schedule parameters for time step t"""
        pass
