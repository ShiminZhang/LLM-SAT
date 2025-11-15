from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..data.base import CodeData
from ..evaluation.code_evaluator import CodeEvaluator
from ..stages import HeuristicCodeDataGenerator, HeuristicCodeSampler
from ..trainers.rlsf import RLSFTrainer


@dataclass
class TrainCoderPipeline:
    """High-level pipeline for coder training.

    Steps (as in diagram):
    - Heuristic code data generation (prompt design, parser)
    - Sample heuristic code data
    - Train coder (RLSF trainer)
    - Evaluation
    """

    generator: HeuristicCodeDataGenerator
    sampler: HeuristicCodeSampler
    trainer: RLSFTrainer
    evaluator: CodeEvaluator

    def run(self, pool: Sequence[CodeData]) -> None:  # pragma: no cover - declaration only
        """Execute the pipeline."""
        raise NotImplementedError

