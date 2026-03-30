"""Typed schemas for experience pool records and API contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, Union


PoolName = Literal["algorithm", "mutation", "combination"]
"""Valid experience pool names."""


class OutcomeLabel(str, Enum):
    """Outcome partition labels for experience pools.

    `GOOD` indicates successful transformations.
    `BAD` indicates failed or harmful transformations.
    """

    GOOD = "good"
    BAD = "bad"


@dataclass(frozen=True)
class AlgorithmExperienceRecord:
    """Schema for algorithm-level bad experiences.

    Fields:
        algorithm_id: Optional unique ID for this algorithm.
        algorithm_description: Natural-language algorithm description that performed poorly.
        analysis: Text explanation of why the algorithm is bad.
    """

    algorithm_description: str
    analysis: str
    algorithm_id: Optional[str] = None


@dataclass(frozen=True)
class MutationExperienceRecord:
    """Schema for mutation experiences.

    Fields:
        leader_algorithm_id: Optional leader algorithm ID.
        member_algorithm_id: Optional member algorithm ID.
        leader_algorithm_description: Source leader algorithm description before mutation.
        member_algorithm_description: Mutated member algorithm description.
        step: Mutation step label that identifies which step produced the member.
        analysis: Text explanation of why the mutation is better or worse.
    """

    leader_algorithm_description: str
    member_algorithm_description: str
    step: str
    analysis: str
    leader_algorithm_id: Optional[str] = None
    member_algorithm_id: Optional[str] = None


@dataclass(frozen=True)
class CombinationExperienceRecord:
    """Schema for crossover/combination experiences.

    Fields:
        parent_alg1_id: Optional first parent algorithm ID.
        parent_alg2_id: Optional second parent algorithm ID.
        new_algorithm_id: Optional offspring/new algorithm ID.
        parent_alg1_description: First parent algorithm description.
        parent_alg2_description: Second parent algorithm description.
        new_algorithm_description: Offspring/combined algorithm description.
        analysis: Text explanation of why combination is better or worse.
    """

    parent_alg1_description: str
    parent_alg2_description: str
    new_algorithm_description: str
    analysis: str
    parent_alg1_id: Optional[str] = None
    parent_alg2_id: Optional[str] = None
    new_algorithm_id: Optional[str] = None


ExperienceRecord = Union[
    AlgorithmExperienceRecord,
    MutationExperienceRecord,
    CombinationExperienceRecord,
]
"""Union of all supported experience record payloads."""


@dataclass(frozen=True)
class PersistReceipt:
    """Return payload for `persist()` operations.

    Fields:
        record_id: Deterministic ID for this record.
        pool_name: Target pool name.
        outcome: Outcome partition where record was written.
        created: True when new record inserted, False when deduplicated.
        partition_size: Number of records in the partition after write.
    """

    record_id: str
    pool_name: PoolName
    outcome: OutcomeLabel
    created: bool
    partition_size: int


@dataclass(frozen=True)
class RetrievedExperience:
    """Return payload for `retrieve()` operations.

    Fields:
        record_id: Unique record identifier.
        pool_name: Source pool name.
        outcome: Source partition label.
        score: Similarity score from FAISS inner-product search.
        payload: Original typed record payload.
    """

    record_id: str
    pool_name: PoolName
    outcome: OutcomeLabel
    score: float
    payload: ExperienceRecord


@dataclass(frozen=True)
class OutcomeExperienceSearchSection:
    """Structured per-outcome result for unified pool search.

    Fields:
        outcome: The outcome partition requested.
        supported: Whether this pool supports the requested outcome.
        requested_retrieve_k: Requested retrieval count for this outcome.
        requested_sample_k: Requested random sampling count for this outcome.
        retrieved: Items returned by `retrieve()`.
        sampled: Items returned by `sample()`.
        unique: Deduplicated union of `retrieved` and `sampled`.
        error: Optional message when this outcome is unsupported or failed.
    """

    outcome: OutcomeLabel
    supported: bool
    requested_retrieve_k: int
    requested_sample_k: int
    retrieved: list[RetrievedExperience]
    sampled: list[RetrievedExperience]
    unique: list[RetrievedExperience]
    error: Optional[str] = None


@dataclass(frozen=True)
class ExperiencePoolSearchResult:
    """Structured output of unified pool search API.

    Fields:
        pool_name: Target pool searched.
        query_text: Original query provided by caller.
        good: Per-outcome search section for GOOD partition.
        bad: Per-outcome search section for BAD partition.
        all_unique: Deduplicated union across good/bad sections.
    """

    pool_name: PoolName
    query_text: Union[str, list[str]]
    good: OutcomeExperienceSearchSection
    bad: OutcomeExperienceSearchSection
    all_unique: list[RetrievedExperience]


OutcomeQuery = Optional[OutcomeLabel]
"""Optional outcome filter in retrieval. `None` means search all valid outcomes."""
