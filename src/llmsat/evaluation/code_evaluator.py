from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..data.base import CodeData


@dataclass
class CodeEvaluator:
    """Evaluate code outputs on a benchmark.

    Default benchmark is assumed to be "SAT2025".
    Implementation is omitted; only interface remains.
    """

    benchmark: str = "SAT2025"

    def evaluate(self, data: CodeData) -> List[int]:  # noqa: D401
        """Return list of par2 scores for provided data."""
        raise NotImplementedError

