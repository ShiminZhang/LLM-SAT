"""Experience pool

This package provides a manager over three pools:
1) algorithm (bad-only experiences)
2) mutation (good and bad experiences)
3) combination (good and bad experiences)
"""

from .manager import ExperiencePoolManager, search_experience_pool
from .types import (
    AlgorithmExperienceRecord,
    CombinationExperienceRecord,
    ExperiencePoolSearchResult,
    MutationExperienceRecord,
    OutcomeExperienceSearchSection,
    OutcomeLabel,
    PersistReceipt,
    PoolName,
    RetrievedExperience,
)

__all__ = [
    "ExperiencePoolManager",
    "search_experience_pool",
    "PoolName",
    "OutcomeLabel",
    "AlgorithmExperienceRecord",
    "MutationExperienceRecord",
    "CombinationExperienceRecord",
    "PersistReceipt",
    "RetrievedExperience",
    "OutcomeExperienceSearchSection",
    "ExperiencePoolSearchResult",
]
