from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..data.base import CodeData


@dataclass
class HeuristicCodeDataGenerator:
    """Generate heuristic code data.

    Contains prompt design and data parsing sub-steps.
    """

    def prompt_design(self, spec: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def data_parser(self, raw: Any) -> CodeData:  # pragma: no cover
        raise NotImplementedError

