from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Data:
    """
    Base data container.

    Represents a generic JSON-like payload used across the project.
    """

    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlgorithmData(Data):
    """
    Algorithm description and metadata.

    Example structure:
        {"algorithm": str, "par2": str, "error_rate": float, ...}
    """


@dataclass
class CodeData(Data):
    """
    Code-related data and metadata.

    Example structure:
        {"algorithm": str, "par2": str, "error_rate": float, ...}
    """


@dataclass
class DPOData(Data):
    """Pairwise preference data for DPO training.

    Example jsonl fields: {"prompt": str, "chosen": str, "reject": str}
    """

