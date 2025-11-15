from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..data.base import AlgorithmData
from ..utils.types import NestedPar2Scores


@dataclass
class AlgorithmEvaluator:
    """Aggregate evaluation over algorithms.

    Implementation omitted; provides only the method signature.
    """

    def evaluate(self, algorithm: AlgorithmData, code_results: NestedPar2Scores) -> List[int]:
        """Return a list of avg par2 scores for the algorithm."""
        raise NotImplementedError

