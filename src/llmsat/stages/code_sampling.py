from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..data.base import CodeData


@dataclass
class HeuristicCodeSampler:
    """Sample heuristic code data uniformly or by rule."""

    def sample(self, data: List[CodeData], n: int) -> List[CodeData]:  # pragma: no cover
        raise NotImplementedError

