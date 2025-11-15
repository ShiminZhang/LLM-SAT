from __future__ import annotations

from dataclasses import dataclass

from ..data.base import AlgorithmData


@dataclass
class AlgorithmLabeler:
    """Label algorithms with metadata or outcomes."""

    def label(self, algorithm: AlgorithmData) -> AlgorithmData:  # pragma: no cover
        raise NotImplementedError

