from __future__ import annotations

from dataclasses import dataclass

from .base import BaseTrainer


@dataclass
class DPOTrainer(BaseTrainer):
    """Direct Preference Optimization trainer placeholder."""

    sft_ratio: float = 0.0

    def train(self) -> None:  # pragma: no cover - declaration only
        raise NotImplementedError

