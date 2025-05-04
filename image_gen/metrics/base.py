from __future__ import annotations
from abc import ABC, abstractmethod
from torch import Tensor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..base import GenerativeModel


class BaseMetric(ABC):
    def __init__(self, model: "GenerativeModel"):
        self.model = model

    @abstractmethod
    def __call__(self, real: Tensor, generated: Tensor, *args, **kwargs) -> float:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def is_lower_better(self) -> bool:
        ...

    def config(self) -> dict:
        return {}

    def __str__(self) -> str:
        config = self.config()
        params = ", ".join(f"{k}: {v}" for k, v in config.items())
        return f"{self._class_name}({params})"

    @property
    def _class_name(self) -> str:
        # This will be automatically overridden in custom classes made by users
        return self.__class__.__name__
