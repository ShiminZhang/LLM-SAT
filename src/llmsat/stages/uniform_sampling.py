from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..data.base import AlgorithmData


@dataclass
class UniformAlgorithmSampler:
    """Uniformly sample algorithms from a pool."""

    def sample(self, algorithms: List[AlgorithmData], n: int) -> List[AlgorithmData]:  # pragma: no cover
        raise NotImplementedError

