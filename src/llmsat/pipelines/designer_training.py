from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..data.base import AlgorithmData
from ..data.filter import DataFilter
from ..evaluation.algorithm_evaluator import AlgorithmEvaluator
from ..stages import AlgorithmGenerator, AlgorithmLabeler, UniformAlgorithmSampler
from ..trainers.dpo import DPOTrainer


@dataclass
class TrainDesignerPipeline:
    """High-level pipeline for designer training.

    Steps (as in diagram):
    - Algorithm generation (prompt design, data parser)
    - Uniformly sample algorithms
    - Label algorithms
    - Train designer (DPO trainer)
    - Evaluation
    """

    generator: AlgorithmGenerator
    sampler: UniformAlgorithmSampler
    labeler: AlgorithmLabeler
    trainer: DPOTrainer
    evaluator: AlgorithmEvaluator
    data_filter: DataFilter

    def run(self, pool: Sequence[AlgorithmData]) -> None:  # pragma: no cover - declaration only
        """Execute the pipeline."""
        raise NotImplementedError

