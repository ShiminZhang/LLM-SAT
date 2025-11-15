from __future__ import annotations

from dataclasses import dataclass

from .base import BaseTrainer


@dataclass
class RLSFTrainer(BaseTrainer):
    """Reinforcement Learning from Synthetic Feedback trainer placeholder."""

    def train(self) -> None:  # pragma: no cover - declaration only
        raise NotImplementedError

