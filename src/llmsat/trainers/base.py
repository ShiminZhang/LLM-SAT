from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseTrainer:
    """Abstract trainer placeholder."""

    model_name: Optional[str] = None

    def train(self) -> None:  # pragma: no cover - declaration only
        """Run training with current configuration."""
        raise NotImplementedError

