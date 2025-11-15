from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .base import Data


class SupportsFilter(Protocol):
    def filter(self, data: Data) -> Data:  # pragma: no cover - interface only
        ...


@dataclass
class DataFilter:
    """Filter and sampling by embedding clusters.

    Attributes
    - k: number of clusters
    - top: number sampled from each cluster

    Implementation intentionally omitted.
    """

    k: int
    top: int

    def filter(self, data: Data) -> Data:  # noqa: D401
        """Return filtered/sampled data (declaration only)."""
        raise NotImplementedError

