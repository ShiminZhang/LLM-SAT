from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..data.base import AlgorithmData


@dataclass
class AlgorithmGenerator:
    """Generate algorithm candidates.

    Consists of prompt design and data parsing sub-steps.
    Implementation is intentionally omitted.
    """

    def prompt_design(self, spec: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def data_parser(self, raw: Any) -> AlgorithmData:  # pragma: no cover
        raise NotImplementedError

