"""Unified manager API for all experience pools.

This is the main entry point to access and update algorithm, mutation,
and combination memory banks through one interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .layout import get_data_root
from .pools import (
    AlgorithmExperiencePool,
    BaseExperiencePool,
    CombinationExperiencePool,
    MutationExperiencePool,
)
from .runtime import SharedRuntime
from .types import (
    ExperiencePoolSearchResult,
    ExperienceRecord,
    OutcomeExperienceSearchSection,
    OutcomeLabel,
    OutcomeQuery,
    PersistReceipt,
    PoolName,
    RetrievedExperience,
)


def _dedupe_hits_by_record_id(hits: List[RetrievedExperience]) -> List[RetrievedExperience]:
    """Keep first occurrence per record_id while preserving order."""

    seen: set[str] = set()
    deduped: List[RetrievedExperience] = []
    for hit in hits:
        if hit.record_id in seen:
            continue
        seen.add(hit.record_id)
        deduped.append(hit)
    return deduped


def _build_outcome_section(
    manager: "ExperiencePoolManager",
    pool_name: PoolName,
    query_text: str | list[str],
    outcome: OutcomeLabel,
    retrieve_k: int,
    sample_k: int,
) -> OutcomeExperienceSearchSection:
    """Build one GOOD/BAD section for unified pool search."""

    pool = manager.get_pool(pool_name)
    safe_retrieve_k = max(0, retrieve_k)
    safe_sample_k = max(0, sample_k)

    # Explicit opt-out: if caller requests zero retrieve and zero sample for an
    # outcome, return an empty section silently (no warning even if unsupported).
    if safe_retrieve_k == 0 and safe_sample_k == 0:
        return OutcomeExperienceSearchSection(
            outcome=outcome,
            supported=(outcome in pool.allowed_outcomes),
            requested_retrieve_k=0,
            requested_sample_k=0,
            retrieved=[],
            sampled=[],
            unique=[],
            error=None,
        )

    if outcome not in pool.allowed_outcomes:
        msg = (
            "[search_experience_pool][WARNING] Unsupported outcome "
            f"'{outcome.value}' for pool '{pool_name}'. Returning empty section."
        )
        print(msg)
        return OutcomeExperienceSearchSection(
            outcome=outcome,
            supported=False,
            requested_retrieve_k=safe_retrieve_k,
            requested_sample_k=safe_sample_k,
            retrieved=[],
            sampled=[],
            unique=[],
            error=msg,
        )

    retrieved: List[RetrievedExperience] = []
    sampled: List[RetrievedExperience] = []
    errors: List[str] = []

    if safe_retrieve_k > 0:
        try:
            if pool_name == "algorithm":
                # Special case: algorithm pool does not use semantic retrieval in
                # unified search; retrieval request is fulfilled by random sample.
                retrieved = manager.sample(
                    pool_name=pool_name,
                    top_k=safe_retrieve_k,
                    outcome=outcome,
                    balanced=False,
                )
            else:
                retrieved = manager.retrieve(
                    pool_name=pool_name,
                    query_text=query_text,
                    top_k=safe_retrieve_k,
                    outcome=outcome,
                    balanced=False,
                )
        except Exception as exc:  # noqa: BLE001
            msg = (
                "[search_experience_pool][WARNING] retrieve failed for "
                f"pool='{pool_name}', outcome='{outcome.value}': {type(exc).__name__}: {exc}"
            )
            print(msg)
            errors.append(msg)

    if safe_sample_k > 0:
        try:
            sampled = manager.sample(
                pool_name=pool_name,
                top_k=safe_sample_k,
                outcome=outcome,
                balanced=False,
            )
        except Exception as exc:  # noqa: BLE001
            msg = (
                "[search_experience_pool][WARNING] sample failed for "
                f"pool='{pool_name}', outcome='{outcome.value}': {type(exc).__name__}: {exc}"
            )
            print(msg)
            errors.append(msg)

    # Precedence rule: retrieval results first, then sampled extras.
    unique = _dedupe_hits_by_record_id([*retrieved, *sampled])

    return OutcomeExperienceSearchSection(
        outcome=outcome,
        supported=True,
        requested_retrieve_k=safe_retrieve_k,
        requested_sample_k=safe_sample_k,
        retrieved=retrieved,
        sampled=sampled,
        unique=unique,
        error="\n".join(errors) if errors else None,
    )


def search_experience_pool(
    manager: "ExperiencePoolManager",
    pool_name: PoolName,
    query_text: str | list[str],
    retrieve_good_k: int,
    retrieve_bad_k: int,
    sample_good_k: int,
    sample_bad_k: int,
) -> ExperiencePoolSearchResult:
    """Unified external API to retrieve+sample experiences for one pool.

    Args:
        manager: Experience pool manager instance.
        pool_name: Target pool to search.
        query_text: Query text used for retrieval.
        retrieve_good_k: Number of items to retrieve from GOOD partition.
        retrieve_bad_k: Number of items to retrieve from BAD partition.
        sample_good_k: Number of items to randomly sample from GOOD partition.
        sample_bad_k: Number of items to randomly sample from BAD partition.

    Returns:
        ExperiencePoolSearchResult: Structured per-outcome and combined results.
    """

    good = _build_outcome_section(
        manager=manager,
        pool_name=pool_name,
        query_text=query_text,
        outcome=OutcomeLabel.GOOD,
        retrieve_k=retrieve_good_k,
        sample_k=sample_good_k,
    )
    bad = _build_outcome_section(
        manager=manager,
        pool_name=pool_name,
        query_text=query_text,
        outcome=OutcomeLabel.BAD,
        retrieve_k=retrieve_bad_k,
        sample_k=sample_bad_k,
    )

    all_unique: List[RetrievedExperience] = []
    seen: set[tuple[OutcomeLabel, str]] = set()
    for item in [*good.unique, *bad.unique]:
        key = (item.outcome, item.record_id)
        if key in seen:
            continue
        seen.add(key)
        all_unique.append(item)

    return ExperiencePoolSearchResult(
        pool_name=pool_name,
        query_text=query_text,
        good=good,
        bad=bad,
        all_unique=all_unique,
    )


class ExperiencePoolManager:
    """Unified façade for all experience pools.

    Responsibilities:
    - Ensure embedding model and FAISS repository are loaded once per process.
    - Route `persist()` and `retrieve()` requests to the correct pool handler.
    - Offer a stable API for future `update()` integration in pipeline stages.
    """

    def __init__(self, data_root: str | Path | None = None) -> None:
        """Initialize manager and pool registry.

        Args:
            data_root: Optional custom root directory for experience data.
                Defaults to `src/experience_pool/data`.
        """

        self.data_root = get_data_root(data_root)
        self.runtime = SharedRuntime.get_instance(self.data_root)
        self._pools: Dict[PoolName, BaseExperiencePool] = {
            "algorithm": AlgorithmExperiencePool(self.runtime),
            "mutation": MutationExperiencePool(self.runtime),
            "combination": CombinationExperiencePool(self.runtime),
        }

    def get_pool(self, pool_name: PoolName) -> BaseExperiencePool:
        """Return pool handler by name.

        Args:
            pool_name: Name of pool (`algorithm`, `mutation`, `combination`).

        Returns:
            BaseExperiencePool: Concrete pool handler.
        """

        return self._pools[pool_name]

    def persist(
        self,
        pool_name: PoolName,
        record: ExperienceRecord,
        outcome: OutcomeLabel,
    ) -> PersistReceipt:
        """Persist one experience record into a target pool partition.

        Args:
            pool_name: Target pool name.
            record: Typed payload matching the target pool schema.
            outcome: Partition label (`good` or `bad`).

        Returns:
            PersistReceipt: Write metadata and deduplication status.
        """

        return self.get_pool(pool_name).persist(record=record, outcome=outcome)

    def retrieve(
        self,
        pool_name: PoolName,
        query_text: str | list[str],
        top_k: int = 5,
        outcome: OutcomeQuery = None,
        balanced: bool = True,
    ) -> List[RetrievedExperience]:
        """Retrieve similar records from a target pool.

        Args:
            pool_name: Source pool name.
            query_text: Query text for semantic retrieval.
                Recommended format by pool:
                - `algorithm`: pass only the current algorithm description text.
                                - `mutation`: pass exactly:
                                    "Leader Algorithm Description: <desc>\nMutation Step: <step>"
                - `combination`: pass exactly:
                  "Parent Algorithm 1: <desc1>\nParent Algorithm 2: <desc2>"
            top_k: Maximum number of records to return.
            outcome: Optional partition filter. If `None`, search all valid
                partitions for the pool.
            balanced: If searching multiple partitions, attempt balanced return
                counts across them.

        Returns:
            list[RetrievedExperience]: Ranked retrieval outputs.
        """

        return self.get_pool(pool_name).retrieve(
            query_text=query_text,
            top_k=top_k,
            outcome=outcome,
            balanced=balanced,
        )

    def sample(
        self,
        pool_name: PoolName,
        top_k: int = 5,
        outcome: OutcomeQuery = None,
        balanced: bool = False,
    ) -> List[RetrievedExperience]:
        """Randomly sample records from a target pool.

        Args:
            pool_name: Source pool name.
            top_k: Maximum number of sampled records.
            outcome: Optional partition filter. If `None`, sample from all valid
                partitions for the pool.
            balanced: If sampling from multiple partitions, attempt balanced
                counts across partitions.

        Returns:
            list[RetrievedExperience]: Random sampled records.
        """

        return self.get_pool(pool_name).sample(
            top_k=top_k,
            outcome=outcome,
            balanced=balanced,
        )

    def search_experience_pool(
        self,
        pool_name: PoolName,
        query_text: str | list[str],
        retrieve_good_k: int,
        retrieve_bad_k: int,
        sample_good_k: int,
        sample_bad_k: int,
    ) -> ExperiencePoolSearchResult:
        """Unified external API to retrieve+sample GOOD/BAD experiences.

        This method never raises for unsupported outcomes. Unsupported sections
        are returned empty with warning messages.
        """

        return search_experience_pool(
            manager=self,
            pool_name=pool_name,
            query_text=query_text,
            retrieve_good_k=retrieve_good_k,
            retrieve_bad_k=retrieve_bad_k,
            sample_good_k=sample_good_k,
            sample_bad_k=sample_bad_k,
        )

    def update(self, pool_name: PoolName | None = None, *args, **kwargs):
        """Placeholder API for future event-based pool updates.

        Args:
            pool_name: Optional target pool name. If provided, forwards update
                call to that pool's `update()` implementation.
            *args: Reserved for future event payloads.
            **kwargs: Reserved for future event payloads.

        Returns:
            Any: Pool-specific update return payload, or `None` when no pool
            is specified.

        Notes:
            Backward-compatible behavior is preserved: if `pool_name` is not
            provided, this method returns `None`.
        """

        if pool_name is None:
            return None
        return self.get_pool(pool_name).update(*args, **kwargs)
