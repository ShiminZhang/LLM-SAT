"""Common type aliases and protocol placeholders."""
from __future__ import annotations

from typing import List, Sequence

Par2Scores = List[int]
NestedPar2Scores = List[Par2Scores]

__all__ = [
    "Par2Scores",
    "NestedPar2Scores",
]

