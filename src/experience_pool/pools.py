"""Pool implementations for algorithm, mutation, and combination experiences."""

from __future__ import annotations

import hashlib
import difflib
from itertools import combinations
import json
import math
import random
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from llmsat.utils.aws import get_code_result
from llmsat.utils.chatgpt_helper import get_llm_response

from .runtime import SharedRuntime
from .types import (
    AlgorithmExperienceRecord,
    CombinationExperienceRecord,
    ExperienceRecord,
    MutationExperienceRecord,
    OutcomeLabel,
    OutcomeQuery,
    PersistReceipt,
    PoolName,
    RetrievedExperience,
)


def _stable_record_id(pool_name: PoolName, outcome: OutcomeLabel, record: ExperienceRecord) -> str:
    """Create deterministic record ID from pool, outcome, and payload.

    Args:
        pool_name: Pool namespace.
        outcome: Outcome partition.
        record: Typed record payload.

    Returns:
        str: SHA256 hex digest used as persistent record ID.
    """

    canonical = {
        "pool": pool_name,
        "outcome": outcome.value,
        "record": asdict(record),
    }
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode("utf-8")).hexdigest()


class BaseExperiencePool:
    """Abstract base class implementing shared persist/retrieve behavior.

    Subclasses define schema validation and query-text construction.
    """

    pool_name: PoolName
    allowed_outcomes: tuple[OutcomeLabel, ...]
    similarity_dedupe_threshold: float = 0.95

    def __init__(self, runtime: SharedRuntime) -> None:
        """Initialize pool with shared runtime.

        Args:
            runtime: Shared process-wide runtime with embedding and FAISS services.
        """

        self.runtime = runtime

    @staticmethod
    def _short_text(text: str, max_len: int = 140) -> str:
        """Return one-line shortened text for concise logs."""

        if not isinstance(text, str):
            return ""
        compact = " ".join(text.split())
        if len(compact) <= max_len:
            return compact
        return compact[: max_len - 3] + "..."

    def validate_record(self, record: ExperienceRecord) -> None:
        """Validate record schema for this pool.

        Args:
            record: Candidate typed record payload.

        Returns:
            None

        Raises:
            TypeError: If record type does not match the pool schema.
        """

        raise NotImplementedError

    def to_index_text(self, record: ExperienceRecord) -> str:
        """Convert a record payload into text used for embedding/indexing.

        Args:
            record: Typed pool record.

        Returns:
            str: Concatenated text representation for semantic search.
        """

        raise NotImplementedError

    def _validate_outcome(self, outcome: OutcomeLabel) -> None:
        """Ensure requested outcome partition is valid for this pool.

        Args:
            outcome: Requested outcome partition.

        Returns:
            None

        Raises:
            ValueError: If outcome is not allowed for this pool.
        """

        if outcome not in self.allowed_outcomes:
            valid = ", ".join(x.value for x in self.allowed_outcomes)
            raise ValueError(f"Invalid outcome '{outcome.value}' for pool '{self.pool_name}'. Valid: {valid}")

    def persist(self, record: ExperienceRecord, outcome: OutcomeLabel) -> PersistReceipt:
        """Persist one experience record into this pool and FAISS partition.

        Args:
            record: Typed record payload for the current pool schema.
            outcome: Target outcome partition.

        Returns:
            PersistReceipt: Metadata about insertion result and partition size.
        """

        self.validate_record(record)
        self._validate_outcome(outcome)

        text = self.to_index_text(record)
        vector = self.runtime.embedding.encode([text])[0]
        record_id = _stable_record_id(self.pool_name, outcome, record)

        partition = self.runtime.index_repo.get_partition(self.pool_name, outcome)

        if record_id in partition.records:
            return PersistReceipt(
                record_id=record_id,
                pool_name=self.pool_name,
                outcome=outcome,
                created=False,
                partition_size=partition.size(),
            )

        created = partition.add(record_id=record_id, record=record, vector=vector)
        if not self._is_partition_consistent(partition):
            print(
                "[ExperiencePool.persist][WARNING] Inconsistent partition sizes detected "
                f"for pool='{self.pool_name}', outcome='{outcome.value}'. "
                "Rebuilding index and metadata mapping."
            )
            self._rebuild_partition(partition)
        else:
            partition.save()

        return PersistReceipt(
            record_id=record_id,
            pool_name=self.pool_name,
            outcome=outcome,
            created=created,
            partition_size=partition.size(),
        )

    def _is_partition_consistent(self, partition) -> bool:
        """Check consistency across FAISS index, id-map, and records dictionary.

        Args:
            partition: Partition object from FAISS repository.

        Returns:
            bool: True when all three structures have identical counts.
        """

        index_size = int(partition.index.ntotal)
        # id_map_size = len(partition.id_map)
        records_size = len(partition.records)
        return index_size == records_size # == id_map_size

    def _rebuild_partition(self, partition) -> None:
        """Rebuild partition index and metadata when sizes are inconsistent.

        Rebuild strategy:
        - Use `records` as source of truth.
        - Reconstruct typed records.
        - Recompute embeddings from current pool index text rules.
        - Recreate FAISS index and id-map from rebuilt records.

        Args:
            partition: Partition object from FAISS repository.

        Returns:
            None
        """

        rebuilt_ids: List[str] = []
        rebuilt_records: Dict[str, dict] = {}
        rebuilt_texts: List[str] = []

        for rec_id in sorted(partition.records.keys()):
            payload_dict = partition.records.get(rec_id)
            if not isinstance(payload_dict, dict):
                print(
                    "[ExperiencePool.persist][WARNING] Skipping malformed record payload "
                    f"during rebuild: record_id='{rec_id}'"
                )
                continue

            try:
                typed_record = self._dict_to_record(payload_dict)
                index_text = self.to_index_text(typed_record)
            except Exception as exc:  # noqa: BLE001
                print(
                    "[ExperiencePool.persist][WARNING] Skipping record during rebuild "
                    f"record_id='{rec_id}' due to {type(exc).__name__}: {exc}"
                )
                continue

            rebuilt_ids.append(rec_id)
            rebuilt_records[rec_id] = payload_dict
            rebuilt_texts.append(index_text)

        new_index = partition._faiss.IndexFlatIP(partition.embedding_dim)
        if rebuilt_texts:
            vectors = self.runtime.embedding.encode(rebuilt_texts)
            new_index.add(vectors)

        partition.index = new_index
        partition.id_map = rebuilt_ids
        partition.records = rebuilt_records
        partition.save()

    def retrieve(
        self,
        query_text: str,
        top_k: int,
        outcome: OutcomeQuery = None,
        balanced: bool = False,
        verbose: bool = True,
    ) -> List[RetrievedExperience]:
        """Retrieve semantically similar experiences from this pool.

        Args:
            query_text: Natural-language query text describing current generation context.
                Expected formats:
                - Algorithm pool: current algorithm description only.
                - Mutation pool: use exactly:
                  "Leader Algorithm Description: <desc>\nMutation Step: <step>".
                - Combination pool: "Parent Algorithm 1: <desc1>\nParent Algorithm 2: <desc2>".
            top_k: Maximum number of results to return.
            outcome: Optional outcome filter. `None` searches all valid partitions.
            balanced: When searching multiple partitions, try to return balanced
                counts across partitions. This should not be used unless special cases.

        Returns:
            list[RetrievedExperience]: Ranked retrieval results with similarity scores.
        """

        if top_k <= 0:
            if verbose:
                print(
                    f"[{self.__class__.__name__}.retrieve] pool='{self.pool_name}' skipped: top_k={top_k}"
                )
            return []

        if verbose:
            outcome_str = outcome.value if outcome is not None else "ALL"
            print(
                f"[{self.__class__.__name__}.retrieve] pool='{self.pool_name}' "
                f"start: top_k={top_k}, outcome={outcome_str}, balanced={balanced}, "
                f"query='{self._short_text(query_text)}'"
            )

        query_vector = self.runtime.embedding.encode([query_text])[0]

        if outcome is not None:
            self._validate_outcome(outcome)
            outcomes = [outcome]
        else:
            outcomes = list(self.allowed_outcomes)

        if len(outcomes) == 1:
            result = self._search_partition(query_vector=query_vector, top_k=top_k, outcome=outcomes[0])
            if verbose:
                print(
                    f"[{self.__class__.__name__}.retrieve] pool='{self.pool_name}' done: "
                    f"returned={len(result)}"
                )
            return result

        if balanced:
            per_partition = max(1, top_k // len(outcomes))
            partial: List[RetrievedExperience] = []
            for label in outcomes:
                partial.extend(
                    self._search_partition(query_vector=query_vector, top_k=per_partition, outcome=label)
                )

            partial.sort(key=lambda x: x.score, reverse=True)
            if len(partial) >= top_k:
                result = partial[:top_k]
                if verbose:
                    print(
                        f"[{self.__class__.__name__}.retrieve] pool='{self.pool_name}' done: "
                        f"returned={len(result)} (balanced)"
                    )
                return result

            # Backfill from all partitions if balanced split did not produce enough.
            merged = self._search_all_partitions(query_vector=query_vector, top_k=top_k)
            if verbose:
                print(
                    f"[{self.__class__.__name__}.retrieve] pool='{self.pool_name}' done: "
                    f"returned={len(merged)} (balanced+backfill)"
                )
            return merged

        merged = self._search_all_partitions(query_vector=query_vector, top_k=top_k)
        if verbose:
            print(
                f"[{self.__class__.__name__}.retrieve] pool='{self.pool_name}' done: returned={len(merged)}"
            )
        return merged

    def sample(
        self,
        top_k: int,
        outcome: OutcomeQuery = None,
        balanced: bool = False,
    ) -> List[RetrievedExperience]:
        """Randomly sample existing experiences from this pool.

        Args:
            top_k: Maximum number of sampled records.
            outcome: Optional outcome filter. `None` samples across all valid
                partitions.
            balanced: If sampling across multiple partitions, attempt to sample
                balanced counts across partitions.

        Returns:
            list[RetrievedExperience]: Random sampled records. `score` is set to
                `0.0` because sampling is not similarity-ranked.
        """

        if top_k <= 0:
            return []

        if outcome is not None:
            self._validate_outcome(outcome)
            outcomes = [outcome]
        else:
            outcomes = list(self.allowed_outcomes)

        if len(outcomes) == 1:
            return self._sample_partition(top_k=top_k, outcome=outcomes[0])

        if balanced:
            per_partition = max(1, top_k // len(outcomes))
            partial: List[RetrievedExperience] = []
            seen: set[tuple[OutcomeLabel, str]] = set()

            for label in outcomes:
                picks = self._sample_partition(top_k=per_partition, outcome=label)
                for item in picks:
                    key = (item.outcome, item.record_id)
                    if key not in seen:
                        seen.add(key)
                        partial.append(item)

            if len(partial) >= top_k:
                random.shuffle(partial)
                return partial[:top_k]

            # Backfill with global random sample from all partitions.
            backfill = self._sample_all_partitions(top_k=top_k)
            for item in backfill:
                key = (item.outcome, item.record_id)
                if key not in seen:
                    seen.add(key)
                    partial.append(item)
                if len(partial) >= top_k:
                    break

            return partial

        return self._sample_all_partitions(top_k=top_k)

    def _search_partition(
        self,
        query_vector,
        top_k: int,
        outcome: OutcomeLabel,
    ) -> List[RetrievedExperience]:
        """Search a single outcome partition.

        Args:
            query_vector: Query embedding vector.
            top_k: Maximum number of records.
            outcome: Target outcome partition.

        Returns:
            list[RetrievedExperience]: Ranked records from this partition.
        """

        partition = self.runtime.index_repo.get_partition(self.pool_name, outcome)
        raw = partition.search(query_vector=query_vector, top_k=top_k)

        return [
            RetrievedExperience(
                record_id=record_id,
                pool_name=self.pool_name,
                outcome=outcome,
                score=score,
                payload=self._dict_to_record(payload_dict),
            )
            for record_id, score, payload_dict in raw
        ]

    def _search_all_partitions(self, query_vector, top_k: int) -> List[RetrievedExperience]:
        """Search all valid partitions and return top globally ranked results.

        Args:
            query_vector: Query embedding vector.
            top_k: Maximum number of returned results.

        Returns:
            list[RetrievedExperience]: Top results merged across partitions.
        """

        merged: List[RetrievedExperience] = []
        for label in self.allowed_outcomes:
            merged.extend(self._search_partition(query_vector=query_vector, top_k=top_k, outcome=label))
        merged.sort(key=lambda x: x.score, reverse=True)
        return merged[:top_k]

    def _sample_partition(
        self,
        top_k: int,
        outcome: OutcomeLabel,
    ) -> List[RetrievedExperience]:
        """Sample random records from one outcome partition."""

        partition = self.runtime.index_repo.get_partition(self.pool_name, outcome)
        items = list(partition.records.items())
        if not items:
            return []

        k = min(top_k, len(items))
        sampled = random.sample(items, k=k)
        return [
            RetrievedExperience(
                record_id=record_id,
                pool_name=self.pool_name,
                outcome=outcome,
                score=0.0,
                payload=self._dict_to_record(payload_dict),
            )
            for record_id, payload_dict in sampled
        ]

    def _sample_all_partitions(self, top_k: int) -> List[RetrievedExperience]:
        """Sample random records across all valid partitions."""

        merged: List[tuple[OutcomeLabel, str, dict]] = []
        for label in self.allowed_outcomes:
            partition = self.runtime.index_repo.get_partition(self.pool_name, label)
            merged.extend((label, record_id, payload_dict) for record_id, payload_dict in partition.records.items())

        if not merged:
            return []

        k = min(top_k, len(merged))
        sampled = random.sample(merged, k=k)
        return [
            RetrievedExperience(
                record_id=record_id,
                pool_name=self.pool_name,
                outcome=outcome,
                score=0.0,
                payload=self._dict_to_record(payload_dict),
            )
            for outcome, record_id, payload_dict in sampled
        ]

    def _dict_to_record(self, payload_dict: dict) -> ExperienceRecord:
        """Convert stored dictionary payload back to typed record.

        Args:
            payload_dict: Dictionary payload loaded from storage.

        Returns:
            ExperienceRecord: Typed record instance.
        """

        raise NotImplementedError

    def update(self, *args, **kwargs) -> Any:
        """Placeholder update API.

        Args:
            *args: Reserved for future update payloads.
            **kwargs: Reserved for future update payloads.

        Returns:
            Any: Pool-specific update return payload.

        Notes:
            Exact update logic is intentionally left empty per current
            implementation scope.
        """

        return None

    @staticmethod
    def _iter_algorithm_dirs(parent_dir: Path):
        """Yield `(folder, id_suffix)` for `algorithm_<id>` directories.

        Args:
            parent_dir: Parent directory to scan.

        Yields:
            tuple[Path, str]: Algorithm folder path and extracted ID suffix.
        """

        if not parent_dir.exists():
            return

        for child in sorted(parent_dir.iterdir()):
            if not child.is_dir() or not child.name.startswith("algorithm_"):
                continue
            algo_id = child.name[len("algorithm_") :]
            if not algo_id:
                continue
            yield child, algo_id

    def _safe_load_json(self, path: Path) -> Optional[dict]:
        """Load JSON file safely.

        Args:
            path: JSON file path.

        Returns:
            Optional[dict]: Parsed payload dictionary, or `None` on failure.
        """

        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(
                f"[{self.__class__.__name__}.update][WARNING] "
                f"Failed to parse JSON at {path}: {type(exc).__name__}: {exc}"
            )
            return None

    @staticmethod
    def _is_nonempty_str(value: Any) -> bool:
        """Return True when value is a non-empty string after strip."""

        return isinstance(value, str) and bool(value.strip())

    @staticmethod
    def _extract_representative_par2(payload: dict) -> Optional[float]:
        """Extract representative PAR2 score from `raw_par2_score[4]`.

        Args:
            payload: Algorithm metadata dictionary.

        Returns:
            Optional[float]: Representative PAR2, or `None` if invalid.
        """

        raw_par2 = payload.get("raw_par2_score")
        if not isinstance(raw_par2, list) or len(raw_par2) != 5:
            return None

        score = raw_par2[4]
        if score is None:
            return None

        try:
            return float(score)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _is_finite_number(value: Any) -> bool:
        """Return True when value is a finite int/float (excluding bool)."""

        if isinstance(value, bool):
            return False
        if not isinstance(value, (int, float)):
            return False
        return math.isfinite(float(value))

    def _is_valid_score_list(self, payload: dict, field_name: str) -> bool:
        """Validate score list field as exactly 5 finite numeric values."""

        raw = payload.get(field_name)
        if not isinstance(raw, list) or len(raw) != 5:
            return False
        return all(self._is_finite_number(x) for x in raw)

    @staticmethod
    def _extract_singleton_str_from_fields(payload: dict, field_names: tuple[str, ...]) -> Optional[str]:
        """Extract one non-empty string from a singleton list field."""

        for field in field_names:
            raw = payload.get(field)

            if isinstance(raw, list) and len(raw) == 1:
                value = raw[0]
                if isinstance(value, str) and value.strip():
                    return value.strip()

        return None

    @staticmethod
    def _extract_json_object_from_text(text: str) -> Optional[dict]:
        """Best-effort extraction of a top-level JSON object from model text."""

        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

        if not isinstance(text, str):
            return None

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        maybe_json = text[start : end + 1]
        try:
            obj = json.loads(maybe_json)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    @staticmethod
    def _extract_json_from_fenced_block(text: str) -> Optional[str]:
        """Extract JSON payload from a ```json fenced block."""

        if not isinstance(text, str):
            return None

        pattern = re.compile(r"```json\s*(\{.*?\})\s*```", flags=re.IGNORECASE | re.DOTALL)
        match = pattern.search(text)
        if not match:
            return None
        return match.group(1)

    @staticmethod
    def _relative_change(reference: float, candidate: float, improve: bool) -> Optional[float]:
        """Compute relative change `(reference-candidate)/abs(reference)` or inverse."""

        if reference == 0:
            return None

        if improve:
            return (reference - candidate) / abs(reference)
        return (candidate - reference) / abs(reference)


class AlgorithmExperiencePool(BaseExperiencePool):
    """Algorithm pool storing bad algorithm experiences only."""

    pool_name: PoolName = "algorithm"
    allowed_outcomes = (OutcomeLabel.BAD,)
    _PROMPT_DIR = Path(__file__).resolve().parents[2] / "data" / "prompts"
    _PROMPT_DEGRADED_FILE = _PROMPT_DIR / "algorithm_analysis_degraded.txt"

    def validate_record(self, record: ExperienceRecord) -> None:
        """Validate algorithm experience payload type.

        Args:
            record: Candidate record payload.

        Returns:
            None

        Raises:
            TypeError: If record is not `AlgorithmExperienceRecord`.
        """

        if not isinstance(record, AlgorithmExperienceRecord):
            raise TypeError("AlgorithmExperiencePool expects AlgorithmExperienceRecord")

    def to_index_text(self, record: ExperienceRecord) -> str:
        """Build index text from algorithm record.

        Args:
            record: Validated `AlgorithmExperienceRecord`.

        Returns:
            str: Search text composed of algorithm description only.

        Notes:
            Per design, the algorithm pool embedding is computed only from
            `algorithm_description` (not from `analysis`).
        """

        assert isinstance(record, AlgorithmExperienceRecord)
        return record.algorithm_description

    def _dict_to_record(self, payload_dict: dict) -> ExperienceRecord:
        """Deserialize dictionary into `AlgorithmExperienceRecord`.

        Args:
            payload_dict: Persisted payload dictionary.

        Returns:
            ExperienceRecord: `AlgorithmExperienceRecord` instance.
        """

        normalized = dict(payload_dict)
        normalized.setdefault("algorithm_id", None)
        return AlgorithmExperienceRecord(**normalized)

    @staticmethod
    def _build_code_diff(
        baseline_code: str,
        candidate_code: str,
        baseline_label: str,
        candidate_label: str,
        max_chars: int = 16000,
    ) -> str:
        """Build unified diff text from baseline/candidate code strings."""

        lines1 = baseline_code.splitlines(keepends=True)
        lines2 = candidate_code.splitlines(keepends=True)
        diff_iter = difflib.unified_diff(
            lines1,
            lines2,
            fromfile=baseline_label,
            tofile=candidate_label,
        )
        diff_text = "".join(diff_iter)
        # if len(diff_text) > max_chars:
        #     return diff_text[:max_chars] + "\n... [diff truncated]"
        return diff_text

    def _load_analysis_prompt_template(self) -> str:
        """Load degraded prompt template text from data/prompts with fallback."""

        try:
            text = self._PROMPT_DEGRADED_FILE.read_text(encoding="utf-8")
            if text.strip():
                return text
        except Exception as exc:  # noqa: BLE001
            print(
                "[AlgorithmExperiencePool.update][WARNING] Failed to load prompt template "
                f"at '{self._PROMPT_DEGRADED_FILE}': {type(exc).__name__}: {exc}"
            )

        return (
            "You are analyzing algorithm outcomes for SAT solver heuristics.\n"
            "ALL provided items are DEGRADATION cases versus a baseline.\n"
            "Return ONLY one valid JSON object in ```json ... ``` with exactly {{EXPECTED_COUNT}} analyses.\n"
            "Use key 'pairs' with items: {\"pair_key\": \"...\", \"analysis\": \"...\"}.\n"
            "\n"
            "Batch data:\n"
            "{{BATCH_CONTENT}}\n"
        )

    def _build_analysis_prompt_for_batch(
        self,
        batch: List[Dict[str, Any]],
        baseline_par2: float,
        baseline_code: str,
    ) -> tuple[str, Dict[str, str]]:
        """Create one LLM prompt and local->actual pair key map for a batch."""

        prompt_template = self._load_analysis_prompt_template()
        lines: List[str] = []
        prompt_key_to_actual: Dict[str, str] = {}

        for idx, candidate in enumerate(batch, 1):
            prompt_key = f"pair_{idx}"
            prompt_key_to_actual[prompt_key] = candidate["pair_key"]

            diff_text = self._build_code_diff(
                baseline_code=baseline_code,
                candidate_code=candidate["candidate_code"],
                baseline_label="baseline_algorithm",
                candidate_label="candidate_algorithm",
            )

            lines.extend(
                [
                    f"PAIR {idx}",
                    f"pair_key: {prompt_key}",
                    f"baseline_par2: {baseline_par2}",
                    f"candidate_par2: {candidate['candidate_par2']}",
                    "candidate_description:",
                    candidate["algorithm_description"],
                    "code_diff_baseline_to_candidate:",
                    diff_text if diff_text.strip() else "[no diff]",
                    "",
                    "\n\n\n",
                ]
            )

        batch_content = "\n\n".join(lines)
        prompt = (
            prompt_template.replace("{{EXPECTED_COUNT}}", str(len(batch)))
            .replace("{{BATCH_CONTENT}}", batch_content)
        )
        return prompt, prompt_key_to_actual

    def _parse_batch_analysis_response(
        self,
        response_text: str,
        expected_keys: List[str],
    ) -> Dict[str, str]:
        """Parse model JSON response into pair_key->analysis mapping."""

        json_text = self._extract_json_from_fenced_block(response_text) or response_text
        parsed = self._extract_json_object_from_text(json_text)
        if parsed is None:
            return None

        pairs = parsed.get("pairs")
        if not isinstance(pairs, list):
            return {}

        out: Dict[str, str] = {}
        expected = set(expected_keys)
        for item in pairs:
            if not isinstance(item, dict):
                continue
            pair_key = item.get("pair_key")
            analysis = item.get("analysis")
            if (
                isinstance(pair_key, str)
                and pair_key in expected
                and isinstance(analysis, str)
                and analysis.strip()
            ):
                out[pair_key] = analysis.strip()
        return out

    def _generate_algorithm_batch_analyses(
        self,
        candidates: List[Dict[str, Any]],
        baseline_par2: float,
        baseline_code: str,
        batch_size: int = 5,
        debug: bool = False,
    ) -> Dict[str, str]:
        """Generate analyses in batches and return candidate_key->analysis mapping."""

        if not candidates:
            return {}

        if batch_size <= 0:
            batch_size = 1

        ordered = sorted(candidates, key=lambda x: (x["candidate_par2"], x["algorithm_id"]))
        analyses: Dict[str, str] = {}

        for i in range(0, len(ordered), batch_size):
            batch = ordered[i : i + batch_size]
            prompt, prompt_key_to_actual = self._build_analysis_prompt_for_batch(
                batch=batch,
                baseline_par2=baseline_par2,
                baseline_code=baseline_code,
            )
            expected_prompt_keys = list(prompt_key_to_actual.keys())

            if debug:
                print("\n[DEBUG] === BATCH PROMPT for BAD ===")
                print(prompt)
                print("[DEBUG] ==============================\n")

            parsed: Dict[str, str] = {}
            max_attempts = 5
            for attempt in range(1, max_attempts + 1):
                try:
                    response_text = get_llm_response(
                        prompt=prompt,
                        system_message=(
                            "You are an expert SAT solver engineer. "
                            "Return valid JSON wrapped in ```json ... ``` only."
                        ),
                        model="gpt-5.4-2026-03-05",
                        temperature=0.7,
                    )

                    if debug:
                        print(f"\n[DEBUG] === ALGORITHM RAW RESPONSE (attempt={attempt}) ===")
                        print(response_text)
                        print("[DEBUG] ===============================================\n")

                    parsed = self._parse_batch_analysis_response(
                        response_text=response_text,
                        expected_keys=expected_prompt_keys,
                    )

                    if debug:
                        print(f"\n[DEBUG] === ALGORITHM PARSED RESPONSE (attempt={attempt}) ===")
                        print(parsed)
                        print("[DEBUG] ================================================\n")

                except Exception as exc:  # noqa: BLE001
                    print(
                        "[AlgorithmExperiencePool.update][WARNING] LLM batch analysis failed "
                        f"for attempt={attempt}: {type(exc).__name__}: {exc}"
                    )
                    parsed = {}

                parsed_actual = {
                    prompt_key_to_actual[k]: v
                    for k, v in parsed.items()
                    if k in prompt_key_to_actual
                }

                if len(parsed_actual) == len(batch):
                    break

            for candidate in batch:
                key = candidate["pair_key"]
                if key in parsed_actual:
                    analyses[key] = parsed_actual[key]

        return analyses

    def _get_code_string_from_aws(
        self,
        code_id: str,
        cache: Dict[str, Optional[str]],
    ) -> Optional[str]:
        """Fetch code string by code_id from AWS-backed DB."""

        if code_id in cache:
            return cache[code_id]

        try:
            result = get_code_result(code_id)
        except Exception as exc:  # noqa: BLE001
            print(
                "[AlgorithmExperiencePool.update][WARNING] AWS lookup failed "
                f"for code_id='{code_id}': {type(exc).__name__}: {exc}"
            )
            cache[code_id] = None
            return None

        code = getattr(result, "code", None) if result is not None else None
        if not isinstance(code, str) or not code.strip():
            cache[code_id] = None
            return None

        cache[code_id] = code
        return code

    def update(
        self,
        input_dir: str | Path,
        baseline_par2: float,
        baseline_code: str,
        batch_size: int = 5,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """Parse algorithms under leaders/members and persist bad-vs-baseline records.

        Expected layout under `input_dir`:
        - optional `<input_dir>/leaders/algorithm_<id>/<id>.json`
        - optional `<input_dir>/members/algorithm_<id>/<id>.json`

        Candidate validity:
        - non-empty `id` and `description`
        - singleton `code_id_list`
        - valid `raw_par2_score` (exactly 5 finite numbers)
        - code exists in AWS for extracted code-id

        Classification:
        - BAD candidate if representative candidate PAR2 > baseline PAR2.
        - Otherwise treated as neutral and skipped.
        - BAD analyses are generated in batches (default size 5), but each
          candidate still receives an individual analysis entry.

        Args:
            input_dir: Path to run directory with `leaders/` and `members/`.
            baseline_par2: PAR2 score of baseline algorithm.
            baseline_code: Source code of baseline algorithm.
            batch_size: Number of bad candidates per LLM batch call.
            debug: Print prompt/response and selected candidates when `True`.

        Returns:
            dict[str, Any]: Update summary counters and diagnostics.

        Raises:
            ValueError: If no `leaders/` or `members/` subfolders exist.
        """

        root = Path(input_dir)
        print(
            f"[AlgorithmExperiencePool.update] start: input_dir='{root}', batch_size={max(1, int(batch_size))}"
        )
        leaders_dir = root / "leaders"
        members_dir = root / "members"

        has_leaders = leaders_dir.exists() and leaders_dir.is_dir()
        has_members = members_dir.exists() and members_dir.is_dir()
        if not has_leaders and not has_members:
            raise ValueError(
                f"Missing leaders/members directories under '{root}'. At least one must exist."
            )

        if not isinstance(baseline_code, str) or not baseline_code.strip():
            raise ValueError("baseline_code must be a non-empty string")

        try:
            baseline_par2_f = float(baseline_par2)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"baseline_par2 must be numeric, got {baseline_par2!r}") from exc

        if not math.isfinite(baseline_par2_f):
            raise ValueError(f"baseline_par2 must be finite, got {baseline_par2!r}")
        if baseline_par2_f == 0:
            raise ValueError("baseline_par2 must be non-zero for relative comparison")

        summary: Dict[str, Any] = {
            "input_dir": str(root),
            "baseline_par2": baseline_par2_f,
            "batch_size": max(1, int(batch_size)),
            "algorithms_seen": 0,
            "algorithms_loaded": 0,
            "bad_candidates": 0,
            "selected_bad": 0,
            "generated_bad": 0,
            "analysis_failed": 0,
            "persisted_bad": 0,
            "persisted_created": 0,
            "persisted_deduped": 0,
            "neutral_skipped": 0,
            "invalid_skipped": 0,
            "missing_code_skipped": 0,
            "errors": [],
        }

        code_cache: Dict[str, Optional[str]] = {}
        bad_candidates: List[Dict[str, Any]] = []

        scan_dirs: List[Path] = []
        if has_leaders:
            scan_dirs.append(leaders_dir)
        if has_members:
            scan_dirs.append(members_dir)

        for partition_dir in scan_dirs:
            for algo_dir, folder_algo_id in self._iter_algorithm_dirs(partition_dir):
                summary["algorithms_seen"] += 1

                json_path = algo_dir / f"{folder_algo_id}.json"
                if not json_path.exists():
                    summary["invalid_skipped"] += 1
                    continue

                payload = self._safe_load_json(json_path)
                if payload is None:
                    summary["invalid_skipped"] += 1
                    continue

                algo_id = payload.get("id")
                description = payload.get("description")
                code_id = self._extract_singleton_str_from_fields(payload, ("code_id_list",))

                if code_id is None:
                    summary["invalid_skipped"] += 1
                    continue

                code = self._get_code_string_from_aws(code_id=code_id, cache=code_cache)
                if code is None:
                    summary["missing_code_skipped"] += 1
                    continue

                if not self._is_valid_score_list(payload, "raw_par2_score"):
                    summary["invalid_skipped"] += 1
                    continue

                score = self._extract_representative_par2(payload)

                if not self._is_nonempty_str(algo_id):
                    summary["invalid_skipped"] += 1
                    continue
                if algo_id != folder_algo_id:
                    err = (
                        "[AlgorithmExperiencePool.update][ERROR] "
                        f"ID mismatch at {json_path}. Folder expects '{folder_algo_id}' but payload id is '{algo_id}'."
                    )
                    summary["errors"].append(err)
                    summary["invalid_skipped"] += 1
                    continue
                if not self._is_nonempty_str(description):
                    summary["invalid_skipped"] += 1
                    continue
                if score is None:
                    summary["invalid_skipped"] += 1
                    continue

                summary["algorithms_loaded"] += 1

                score_f = float(score)
                if score_f <= baseline_par2_f:
                    summary["neutral_skipped"] += 1
                    continue

                relative_change = self._relative_change(baseline_par2_f, score_f, improve=False)
                if relative_change is None:
                    summary["invalid_skipped"] += 1
                    continue

                bad_candidates.append(
                    {
                        "pair_key": f"{partition_dir.name}__{algo_id}",
                        "algorithm_id": algo_id,
                        "algorithm_description": description,
                        "algorithm_code_id": code_id,
                        "candidate_code": code,
                        "candidate_par2": score_f,
                        "relative_change": float(relative_change),
                        "json_path": str(json_path),
                        "partition": partition_dir.name,
                    }
                )

        bad_candidates.sort(key=lambda x: (-x["relative_change"], x["algorithm_id"]))
        summary["bad_candidates"] = len(bad_candidates)
        summary["selected_bad"] = len(bad_candidates)
        selected_bad = bad_candidates

        if debug:
            print("\n[DEBUG] Selected BAD algorithms:")
            for cand in bad_candidates:
                print(
                    f" - {cand['algorithm_id']}: "
                    f"par2={cand['candidate_par2']}, rel_change={cand['relative_change']:.4f}, "
                    f"source={cand['partition']}"
                )
            print()

        analysis_map = self._generate_algorithm_batch_analyses(
            candidates=selected_bad,
            baseline_par2=baseline_par2_f,
            baseline_code=baseline_code,
            batch_size=max(1, int(batch_size)),
            debug=debug,
        )
        print(
            f"[AlgorithmExperiencePool.update] analysis stage done: generated={len(analysis_map)}/{len(selected_bad)}"
        )

        for cand in selected_bad:
            generated_analysis = analysis_map.get(cand["pair_key"])
            if not generated_analysis:
                summary["analysis_failed"] += 1
                continue

            summary["generated_bad"] += 1

            record = AlgorithmExperienceRecord(
                algorithm_description=cand["algorithm_description"],
                analysis=generated_analysis,
                algorithm_id=cand["algorithm_id"],
            )

            receipt = self.persist(record=record, outcome=OutcomeLabel.BAD)
            summary["persisted_bad"] += 1

            if receipt.created:
                summary["persisted_created"] += 1
            else:
                summary["persisted_deduped"] += 1

        print("\n[AlgorithmExperiencePool] Update Summary:")
        print(
            f"  - Algorithms loaded: {summary['algorithms_loaded']} "
            f"({summary['invalid_skipped']} invalid, {summary['missing_code_skipped']} missing code)"
        )
        print(
            f"  - Bad candidates found: {summary['bad_candidates']} "
            f"({summary['neutral_skipped']} neutral vs baseline)"
        )
        print(
            f"  - Analyses generated: {summary['generated_bad']} BAD "
            f"({summary['analysis_failed']} failed)"
        )
        print(
            f"  - Persisted experiences: {summary['persisted_bad']} BAD"
        )
        print(
            f"  - Persistence details: {summary['persisted_created']} created, "
            f"{summary['persisted_deduped']} deduped"
        )

        return summary


class MutationExperiencePool(BaseExperiencePool):
    """Mutation pool storing good and bad mutation experiences."""

    pool_name: PoolName = "mutation"
    allowed_outcomes = (OutcomeLabel.GOOD, OutcomeLabel.BAD)
    _PROMPT_DIR = Path(__file__).resolve().parents[2] / "data" / "prompts"
    _PROMPT_IMPROVED_FILE = _PROMPT_DIR / "mutation_analysis_improved.txt"
    _PROMPT_DEGRADED_FILE = _PROMPT_DIR / "mutation_analysis_degraded.txt"

    def validate_record(self, record: ExperienceRecord) -> None:
        """Validate mutation experience payload type.

        Args:
            record: Candidate record payload.

        Returns:
            None

        Raises:
            TypeError: If record is not `MutationExperienceRecord`.
        """

        if not isinstance(record, MutationExperienceRecord):
            raise TypeError("MutationExperiencePool expects MutationExperienceRecord")

    def to_index_text(self, record: ExperienceRecord) -> str:
        """Build index text from mutation record.

        Args:
            record: Validated `MutationExperienceRecord`.

        Returns:
            str: Search text composed of leader description and mutation step.

        Notes:
            Per design, the mutation pool embedding is computed from:
            "Leader Algorithm Description: ...\nMutation Step: ...".
            Query using the same labeled format.
        """

        assert isinstance(record, MutationExperienceRecord)
        return (
            f"Leader Algorithm Description: {record.leader_algorithm_description}\n"
            f"Mutation Step: {record.step}"
        )

    def _dict_to_record(self, payload_dict: dict) -> ExperienceRecord:
        """Deserialize dictionary into `MutationExperienceRecord`.

        Args:
            payload_dict: Persisted payload dictionary.

        Returns:
            ExperienceRecord: `MutationExperienceRecord` instance.
        """

        normalized = dict(payload_dict)
        normalized.setdefault("leader_algorithm_id", None)
        normalized.setdefault("member_algorithm_id", None)
        normalized.setdefault("leader_raw_par2", None)
        normalized.setdefault("member_raw_par2", None)
        return MutationExperienceRecord(**normalized)

    @staticmethod
    def _chunk_list(items: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
        """Split a list into fixed-size batches."""

        if batch_size <= 0:
            batch_size = 1
        return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

    @staticmethod
    def _build_code_diff(
        leader_code: str,
        member_code: str,
        leader_label: str,
        member_label: str,
        max_chars: int = 16000,
    ) -> str:
        """Build unified diff text from leader/member code strings."""

        lines1 = leader_code.splitlines(keepends=True)
        lines2 = member_code.splitlines(keepends=True)
        diff_iter = difflib.unified_diff(
            lines1,
            lines2,
            fromfile=leader_label,
            tofile=member_label,
        )
        diff_text = "".join(diff_iter)
        # if len(diff_text) > max_chars:
        #     return diff_text[:max_chars] + "\n... [diff truncated]"
        return diff_text

    def _build_analysis_prompt_for_batch(
        self,
        batch: List[Dict[str, Any]],
        outcome: OutcomeLabel,
    ) -> tuple[str, Dict[str, str]]:
        """Create one LLM prompt and local->actual pair key map for a batch."""

        prompt_template = self._load_analysis_prompt_template(outcome=outcome)

        lines: List[str] = []
        prompt_key_to_actual: Dict[str, str] = {}

        for idx, pair in enumerate(batch, 1):
            prompt_key = f"pair_{idx}"
            prompt_key_to_actual[prompt_key] = pair["pair_key"]
            diff_text = self._build_code_diff(
                leader_code=pair.get("leader_code", ""),
                member_code=pair.get("member_code", ""),
                leader_label=f"leader_algorithm",
                member_label=f"member_algorithm",
            )
            lines.extend(
                [
                    f"PAIR {idx}",
                    f"pair_key: {prompt_key}",
                    f"leader_par2: {pair['leader_par2']}",
                    f"member_par2: {pair['member_par2']}",
                    f"relative_change: {pair['relative_change']*100}%",
                    "leader_description:",
                    pair["leader_algorithm_description"],
                    "step_mutated:",
                    pair["step"],
                    "member_description:",
                    pair["member_algorithm_description"],
                    "code_diff:",
                    diff_text if diff_text.strip() else "[no diff]",
                    "",
                    "\n\n\n",
                ]
            )

        batch_content = "\n\n".join(lines)
        expected_count = str(len(batch))
        prompt_text = (
            prompt_template.replace("{{EXPECTED_COUNT}}", expected_count)
            .replace("{{BATCH_CONTENT}}", batch_content)
        )
        return prompt_text, prompt_key_to_actual

    def _load_analysis_prompt_template(self, outcome: OutcomeLabel) -> str:
        """Load prompt template text from data/prompts with safe fallback."""

        prompt_path = (
            self._PROMPT_IMPROVED_FILE if outcome == OutcomeLabel.GOOD else self._PROMPT_DEGRADED_FILE
        )

        try:
            text = prompt_path.read_text(encoding="utf-8")
            if text.strip():
                return text
        except Exception as exc:  # noqa: BLE001
            print(
                "[MutationExperiencePool.update][WARNING] Failed to load prompt template "
                f"at '{prompt_path}': {type(exc).__name__}: {exc}"
            )

        return (
            "You are analyzing SAT-solver mutation pairs.\n"
            "Return a valid JSON object wrapped in ```json ... ``` with exactly {{EXPECTED_COUNT}} pair analyses.\n"
            "Use key 'pairs' with items of shape: {\"pair_key\": \"...\", \"analysis\": \"...\"}.\n"
            "\n"
            "Batch data:\n"
            "{{BATCH_CONTENT}}\n"
        )

    def _parse_batch_analysis_response(
        self,
        response_text: str,
        expected_keys: List[str],
    ) -> Dict[str, str]:
        """Parse model JSON response into pair_key->analysis mapping."""

        json_text = self._extract_json_from_fenced_block(response_text) or response_text
        parsed = self._extract_json_object_from_text(json_text)
        if parsed is None:
            return {}

        pairs = parsed.get("pairs")
        if not isinstance(pairs, list):
            return {}

        out: Dict[str, str] = {}
        expected = set(expected_keys)
        for item in pairs:
            if not isinstance(item, dict):
                continue
            pair_key = item.get("pair_key")
            analysis = item.get("analysis")
            if (
                isinstance(pair_key, str)
                and pair_key in expected
                and isinstance(analysis, str)
                and analysis.strip()
            ):
                out[pair_key] = analysis.strip()
        return out

    def _generate_mutation_batch_analyses(
        self,
        candidates: List[Dict[str, Any]],
        outcome: OutcomeLabel,
        batch_size: int = 5,
        debug: bool = False,
    ) -> Dict[str, str]:
        """Generate analyses in batches; each batch contains 5 pairs by member PAR2."""

        if not candidates:
            return {}

        # Sort by member PAR2 so first batch has best member PAR2 values.
        ordered = sorted(candidates, key=lambda x: (x["member_par2"], x["member_algorithm_id"]))
        analyses: Dict[str, str] = {}

        for batch in self._chunk_list(ordered, batch_size=batch_size):
            prompt, prompt_key_to_actual = self._build_analysis_prompt_for_batch(
                batch=batch,
                outcome=outcome,
            )
            expected_prompt_keys = list(prompt_key_to_actual.keys())

            if debug:
                print(f"\n[DEBUG] === BATCH PROMPT for {outcome.value} ===")
                print(prompt)
                print("[DEBUG] =======================================\n")

            parsed: Dict[str, str] = {}
            max_attempts = 5
            for attempt in range(1, max_attempts + 1):
                try:
                    response_text = get_llm_response(
                        prompt=prompt,
                        system_message=(
                            "You are an expert SAT solver engineer. "
                            "Return valid JSON wrapped in ```json ... ``` only."
                        ),
                        model="gemini-3-flash-preview",
                        temperature=0.7,
                    )

                    if debug:
                        print(f"\n[DEBUG] === RAW RESPONSE (ATTEMPT {attempt}) for {outcome.value} ===")
                        print(response_text)
                        print("[DEBUG] ==================================================\n")

                    parsed = self._parse_batch_analysis_response(
                        response_text=response_text,
                        expected_keys=expected_prompt_keys,
                    )

                    if debug:
                        print(f"\n[DEBUG] === PARSED RESPONSE (ATTEMPT {attempt}) for {outcome.value} ===")
                        print(parsed)
                        print("[DEBUG] =====================================================\n")

                except Exception as exc:  # noqa: BLE001
                    print(
                        "[MutationExperiencePool.update][WARNING] LLM batch analysis failed "
                        f"for outcome='{outcome.value}', attempt={attempt}: {type(exc).__name__}: {exc}"
                    )
                    parsed = {}

                parsed_actual = {
                    prompt_key_to_actual[k]: v
                    for k, v in parsed.items()
                    if k in prompt_key_to_actual
                }

                if len(parsed_actual) == len(batch):
                    break

            # Only keep successfully generated analyses.
            for pair in batch:
                key = pair["pair_key"]
                if key in parsed_actual:
                    analyses[key] = parsed_actual[key]

        return analyses

    def _get_code_string_from_aws(
        self,
        code_id: str,
        cache: Dict[str, Optional[str]],
    ) -> Optional[str]:
        """Fetch code string by code_id from AWS-backed DB.

        Returns:
            Optional[str]: Non-empty code string when available; otherwise None.
        """

        if code_id in cache:
            return cache[code_id]

        try:
            result = get_code_result(code_id)
        except Exception as exc:  # noqa: BLE001
            print(
                "[MutationExperiencePool.update][WARNING] AWS lookup failed "
                f"for code_id='{code_id}': {type(exc).__name__}: {exc}"
            )
            cache[code_id] = None
            return None

        code = getattr(result, "code", None) if result is not None else None
        if not isinstance(code, str) or not code.strip():
            cache[code_id] = None
            return None

        cache[code_id] = code
        return code

    def update(
        self,
        input_dir: str | Path,
        threshold: float = 0.10,
        top_k_good: int = 5,
        top_k_bad: int = 5,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """Parse mutation run directory and persist top-ranked good/bad pairs.

        The expected directory shape is:
        - `<input_dir>/leaders/algorithm_<id>/<id>.json`
        - `<input_dir>/members/algorithm_<id>/<id>.json`

        Pair construction rules:
        - Scan all members.
        - Keep only valid members with:
            - singleton code-id list field (`code_id`/`code_id_list`/`code_ids`)
            - code string exists in AWS database for that code id and is non-empty
            - singleton `parent_id` list
            - valid `raw_par2_score` list (5 finite numbers)
            - non-empty `description`
            - non-empty `step`
        - Resolve member parent from leaders map and keep only valid leaders with:
            - singleton code-id list field (`code_id`/`code_id_list`/`code_ids`)
            - code string exists in AWS database for that code id and is non-empty
            - valid `raw_par2_score` list (5 finite numbers)
            - non-empty `description`
            - representative PAR2 extracted by `_extract_representative_par2`

        Classification and ranking:
        - GOOD pair: member_par2 < leader_par2
          rank score = (leader_par2 - member_par2) / abs(leader_par2)
        - BAD pair: member_par2 > leader_par2
          rank score = (member_par2 - leader_par2) / abs(leader_par2)
        - Equal representative PAR2: neutral (ignored)
        - Rank both GOOD and BAD candidate sets by descending rank score.
        - Persist only top-K from each side.

        Args:
            input_dir: Path to run directory with `leaders/` and `members/`.
            threshold: Backward-compatible argument kept for API stability.
                Not used in current ranking/classification.
            top_k_good: Number of top-ranked GOOD pairs to persist.
            top_k_bad: Number of top-ranked BAD pairs to persist.

        Returns:
            dict[str, Any]: Update summary counters and diagnostics.

        Raises:
            ValueError: If `leaders/` or `members/` subfolders are missing.
        """

        root = Path(input_dir)
        print(
            f"[MutationExperiencePool.update] start: input_dir='{root}', "
            f"top_k_good={max(0, int(top_k_good))}, top_k_bad={max(0, int(top_k_bad))}"
        )
        leaders_dir = root / "leaders"
        members_dir = root / "members"

        if not leaders_dir.exists() or not leaders_dir.is_dir():
            raise ValueError(f"Missing leaders directory: {leaders_dir}")
        if not members_dir.exists() or not members_dir.is_dir():
            raise ValueError(f"Missing members directory: {members_dir}")

        summary: Dict[str, Any] = {
            "input_dir": str(root),
            "threshold": threshold,
            "top_k_good": max(0, int(top_k_good)),
            "top_k_bad": max(0, int(top_k_bad)),
            "leaders_seen": 0,
            "leaders_loaded": 0,
            "members_seen": 0,
            "members_loaded": 0,
            "good_candidates": 0,
            "bad_candidates": 0,
            "selected_good": 0,
            "selected_bad": 0,
            "persisted_good": 0,
            "persisted_bad": 0,
            "persisted_created": 0,
            "persisted_deduped": 0,
            "neutral_skipped": 0,
            "invalid_skipped": 0,
            "missing_code_skipped": 0,
            "missing_leader_skipped": 0,
            "parent_cardinality_errors": 0,
            "errors": [],
        }

        leader_map: Dict[str, Dict[str, Any]] = {}
        code_cache: Dict[str, Optional[str]] = {}
        good_candidates: List[Dict[str, Any]] = []
        bad_candidates: List[Dict[str, Any]] = []

        # ------------------------------
        # Load leaders
        # ------------------------------
        for algo_dir, folder_algo_id in self._iter_algorithm_dirs(leaders_dir):
            summary["leaders_seen"] += 1
            json_path = algo_dir / f"{folder_algo_id}.json"
            if not json_path.exists():
                summary["invalid_skipped"] += 1
                continue

            payload = self._safe_load_json(json_path)
            if payload is None:
                summary["invalid_skipped"] += 1
                continue

            algo_id = payload.get("id")
            description = payload.get("description")
            function_name = payload.get("function_name")
            code_id = self._extract_singleton_str_from_fields(
                payload,
                ("code_id_list",),
            )

            if code_id is None:
                summary["invalid_skipped"] += 1
                continue

            leader_code = self._get_code_string_from_aws(code_id=code_id, cache=code_cache)
            if leader_code is None:
                summary["missing_code_skipped"] += 1
                continue

            # Leader validation requires raw_par2_score to be a numeric
            # length-5 list. Representative PAR2 still comes from the shared
            # `_extract_representative_par2` helper.
            if not self._is_valid_score_list(payload, "raw_par2_score"):
                summary["invalid_skipped"] += 1
                continue

            score = self._extract_representative_par2(payload)

            if not self._is_nonempty_str(algo_id):
                summary["invalid_skipped"] += 1
                continue
            if algo_id != folder_algo_id:
                print(
                    "[MutationExperiencePool.update][ERROR] "
                    f"ID mismatch at {json_path}. Folder expects '{folder_algo_id}' but payload id is '{algo_id}'."
                )
                summary["invalid_skipped"] += 1
                continue

            if not self._is_nonempty_str(description):
                summary["invalid_skipped"] += 1
                continue
            if score is None:
                print(
                    "[MutationExperiencePool.update][ERROR] "
                    f"Invalid par2 score found for leader at {json_path}: {score}"
                )
                summary["invalid_skipped"] += 1
                continue

            summary["leaders_loaded"] += 1
            leader_map[algo_id] = {
                "id": algo_id,
                "code_id": code_id,
                "code": leader_code,
                "function_name": function_name if isinstance(function_name, str) else "",
                "description": description,
                "score": score,
                "raw_par2": payload.get("raw_par2_score"),
                "json_path": str(json_path),
            }

        # ------------------------------
        # Load members + classify
        # ------------------------------
        for algo_dir, folder_algo_id in self._iter_algorithm_dirs(members_dir):
            summary["members_seen"] += 1
            json_path = algo_dir / f"{folder_algo_id}.json"
            if not json_path.exists():
                summary["invalid_skipped"] += 1
                continue

            payload = self._safe_load_json(json_path)
            if payload is None:
                summary["invalid_skipped"] += 1
                continue

            member_id = payload.get("id")
            member_desc = payload.get("description")
            # Backward/forward compatibility:
            # - older payloads used `step`
            # - current generation payloads use `mutation_step`
            member_step = payload.get("step")
            if not self._is_nonempty_str(member_step):
                member_step = payload.get("mutation_step")
            member_function_name = payload.get("function_name")
            member_code_id = self._extract_singleton_str_from_fields(
                payload,
                ("code_id_list",),
            )

            if member_code_id is None:
                summary["invalid_skipped"] += 1
                continue

            member_code = self._get_code_string_from_aws(code_id=member_code_id, cache=code_cache)
            if member_code is None:
                summary["missing_code_skipped"] += 1
                continue

            if not self._is_valid_score_list(payload, "raw_par2_score"):
                summary["invalid_skipped"] += 1
                continue

            member_score = self._extract_representative_par2(payload)
            parent_id_raw = payload.get("parent_id")

            if not self._is_nonempty_str(member_id):
                summary["invalid_skipped"] += 1
                continue
            if member_id != folder_algo_id:
                err = (
                    "[MutationExperiencePool.update][ERROR] "
                    f"ID mismatch at {json_path}. Folder expects '{folder_algo_id}' but payload id is '{member_id}'."
                )
                print(err)
                summary["errors"].append(err)
                summary["invalid_skipped"] += 1
                continue
            if not self._is_nonempty_str(member_desc):
                summary["invalid_skipped"] += 1
                continue
            if not self._is_nonempty_str(member_step):
                summary["invalid_skipped"] += 1
                continue
            if member_score is None:
                print(
                    "[MutationExperiencePool.update][ERROR] "
                    f"Invalid par2 score found for member at {json_path}: {member_score}"
                )
                summary["invalid_skipped"] += 1
                continue

            member_step_str = member_step.strip()

            parent_ids: List[str] = []
            if isinstance(parent_id_raw, list):
                parent_ids = [x for x in parent_id_raw if isinstance(x, str) and x.strip()]

            # Per spec: member parent_id must be a list of length exactly one.
            if not isinstance(parent_id_raw, list):
                err = (
                    "[MutationExperiencePool.update][ERROR] Invalid parent_id type. "
                    f"Expected list with exactly 1 parent_id for member id '{member_id}' at {json_path}. "
                    f"parent_id={parent_id_raw!r}"
                )
                print(err)
                summary["errors"].append(err)
                summary["parent_cardinality_errors"] += 1
                summary["invalid_skipped"] += 1
                continue

            if len(parent_ids) != 1:
                err = (
                    "[MutationExperiencePool.update][ERROR] Invalid parent cardinality. "
                    f"Expected exactly 1 parent_id, got {len(parent_ids)} for member id '{member_id}' "
                    f"at {json_path}. parent_id={parent_id_raw!r}"
                )
                print(err)
                summary["errors"].append(err)
                summary["parent_cardinality_errors"] += 1
                summary["invalid_skipped"] += 1
                continue

            parent_id = parent_ids[0]
            leader = leader_map.get(parent_id)
            if leader is None:
                summary["missing_leader_skipped"] += 1
                continue

            leader_score = float(leader["score"])
            if leader_score == 0:
                summary["invalid_skipped"] += 1
                continue

            summary["members_loaded"] += 1
            member_score_f = float(member_score)

            if member_score_f == leader_score:
                summary["neutral_skipped"] += 1
                continue

            if member_score_f < leader_score:
                relative_change = (leader_score - member_score_f) / abs(leader_score)
                good_candidates.append(
                    {
                        "pair_key": f"{leader['id']}__{member_id}",
                        "leader_algorithm_id": leader["id"],
                        "leader_code_id": leader["code_id"],
                        "leader_function_name": leader["function_name"],
                        "leader_algorithm_description": leader["description"],
                        "leader_code": leader["code"],
                        "leader_par2": leader_score,
                        "member_algorithm_id": member_id,
                        "member_code_id": member_code_id,
                        "member_function_name": (
                            member_function_name if isinstance(member_function_name, str) else ""
                        ),
                        "member_code": member_code,
                        "member_algorithm_description": member_desc,
                        "step": member_step_str,
                        "member_par2": member_score_f,
                        "leader_raw_par2": leader.get("raw_par2"),
                        "member_raw_par2": payload.get("raw_par2_score"),
                        "relative_change": relative_change,
                    }
                )
            else:
                relative_change = (member_score_f - leader_score) / abs(leader_score)
                bad_candidates.append(
                    {
                        "pair_key": f"{leader['id']}__{member_id}",
                        "leader_algorithm_id": leader["id"],
                        "leader_code_id": leader["code_id"],
                        "leader_function_name": leader["function_name"],
                        "leader_algorithm_description": leader["description"],
                        "leader_code": leader["code"],
                        "leader_par2": leader_score,
                        "member_algorithm_id": member_id,
                        "member_code_id": member_code_id,
                        "member_function_name": (
                            member_function_name if isinstance(member_function_name, str) else ""
                        ),
                        "member_code": member_code,
                        "member_algorithm_description": member_desc,
                        "step": member_step_str,
                        "member_par2": member_score_f,
                        "leader_raw_par2": leader.get("raw_par2"),
                        "member_raw_par2": payload.get("raw_par2_score"),
                        "relative_change": relative_change,
                    }
                )

        good_candidates.sort(
            key=lambda x: (-x["relative_change"], x["member_algorithm_id"], x["leader_algorithm_id"])
        )
        bad_candidates.sort(
            key=lambda x: (-x["relative_change"], x["member_algorithm_id"], x["leader_algorithm_id"])
        )

        summary["good_candidates"] = len(good_candidates)
        summary["bad_candidates"] = len(bad_candidates)

        selected_good = good_candidates[: max(0, int(top_k_good))]
        selected_bad = bad_candidates[: max(0, int(top_k_bad))]
        summary["selected_good"] = len(selected_good)
        summary["selected_bad"] = len(selected_bad)

        if debug:
            print("\n[DEBUG] Selected GOOD candidates:")
            for cand in selected_good:
                print(f" - {cand['pair_key']}: leader_id={cand['leader_algorithm_id']}, member_id={cand['member_algorithm_id']}, step={cand['step']}, rel_change={cand['relative_change']:.4f}")
            print("\n[DEBUG] Selected BAD candidates:")
            for cand in selected_bad:
                print(f" - {cand['pair_key']}: leader_id={cand['leader_algorithm_id']}, member_id={cand['member_algorithm_id']}, step={cand['step']}, rel_change={cand['relative_change']:.4f}")
            print()

        print(
            f"[MutationExperiencePool.update] generation stage: "
            f"selected_good={len(selected_good)}, selected_bad={len(selected_bad)}"
        )

        good_analysis_map = self._generate_mutation_batch_analyses(
            candidates=selected_good,
            outcome=OutcomeLabel.GOOD,
            batch_size=5,
            debug=debug,
        )
        bad_analysis_map = self._generate_mutation_batch_analyses(
            candidates=selected_bad,
            outcome=OutcomeLabel.BAD,
            batch_size=5,
            debug=debug,
        )

        def _persist_selected(
            candidates: List[Dict[str, Any]],
            outcome: OutcomeLabel,
            analysis_map: Dict[str, str],
        ) -> None:
            for cand in candidates:
                pair_key = cand["pair_key"]
                generated_analysis = analysis_map.get(pair_key)
                
                if not generated_analysis:
                    continue  # do not store pairs without analysis

                record = MutationExperienceRecord(
                    leader_algorithm_description=cand["leader_algorithm_description"],
                    member_algorithm_description=cand["member_algorithm_description"],
                    step=cand["step"],
                    analysis=generated_analysis,
                    leader_algorithm_id=cand["leader_algorithm_id"],
                    member_algorithm_id=cand["member_algorithm_id"],
                    leader_raw_par2=cand.get("leader_raw_par2"),
                    member_raw_par2=cand.get("member_raw_par2"),
                )

                receipt = self.persist(record=record, outcome=outcome)

                if outcome == OutcomeLabel.GOOD:
                    summary["persisted_good"] += 1
                else:
                    summary["persisted_bad"] += 1

                if receipt.created:
                    summary["persisted_created"] += 1
                else:
                    summary["persisted_deduped"] += 1

        _persist_selected(selected_good, OutcomeLabel.GOOD, good_analysis_map)
        _persist_selected(selected_bad, OutcomeLabel.BAD, bad_analysis_map)

        print("\n[MutationExperiencePool] Update Summary:")
        print(f"  - Leaders processsed: {summary['leaders_loaded']} "
              f"({summary['invalid_skipped']} total skipped)")
        print(f"  - Members processed: {summary['members_loaded']} "
              f"({summary['neutral_skipped']} neutral, {summary['missing_code_skipped']} missing code)")
        print(f"  - Candidates found: {summary['good_candidates']} GOOD, {summary['bad_candidates']} BAD")
        print(f"  - Candidates selected for generation: {summary['selected_good']} GOOD, {summary['selected_bad']} BAD")
        print(f"  - Experiences actually persisted: {summary['persisted_good']} GOOD, {summary['persisted_bad']} BAD")
        print(f"  - Persistence details: {summary['persisted_created']} created, {summary['persisted_deduped']} deduped")

        return summary


class CombinationExperiencePool(BaseExperiencePool):
    """Combination pool storing good and bad crossover experiences."""

    pool_name: PoolName = "combination"
    allowed_outcomes = (OutcomeLabel.GOOD, OutcomeLabel.BAD)
    _PROMPT_DIR = Path(__file__).resolve().parents[2] / "data" / "prompts"
    _PROMPT_IMPROVED_FILE = _PROMPT_DIR / "combination_analysis_improved.txt"
    _PROMPT_DEGRADED_FILE = _PROMPT_DIR / "combination_analysis_degraded.txt"

    def validate_record(self, record: ExperienceRecord) -> None:
        """Validate combination experience payload type.

        Args:
            record: Candidate record payload.

        Returns:
            None

        Raises:
            TypeError: If record is not `CombinationExperienceRecord`.
        """

        if not isinstance(record, CombinationExperienceRecord):
            raise TypeError("CombinationExperiencePool expects CombinationExperienceRecord")

    def to_index_text(self, record: ExperienceRecord) -> str:
        """Build index text from combination record.

        Args:
            record: Validated `CombinationExperienceRecord`.

        Returns:
            str: Search text with parent algorithm descriptions only.

        Notes:
            Per design, the combination pool embedding is computed from:
            "Parent Algorithm 1: ...\nParent Algorithm 2: ..."
            and excludes offspring/analysis text.
        """

        assert isinstance(record, CombinationExperienceRecord)
        return (
            f"Parent Algorithm 1: {record.parent_alg1_description}\n"
            f"Parent Algorithm 2: {record.parent_alg2_description}"
        )

    def retrieve(
        self,
        query_text: str | List[str],
        top_k: int,
        outcome: OutcomeQuery = None,
        balanced: bool = False,
        verbose: bool = True,
    ) -> List[RetrievedExperience]:
        """Retrieve from combination pool using one query or potential leaders.

        Special behavior for this pool:
        - If `query_text` is a list, treat it as potential leader descriptions.
        - Build all unique unordered pairs (`N choose 2`) from those leaders.
        - Format each pair as:
          "Parent Algorithm 1: <desc1>\nParent Algorithm 2: <desc2>"
        - Retrieve top-k results independently for each pair query.
        - Merge all candidates and keep unique records by `record_id`.
        - For duplicates, keep the instance with the highest similarity score.
        - Return global top-k by score after deduplication.

        Args:
            query_text: Single query string, or list of potential leader
                descriptions used to build pair queries.
            top_k: Maximum number of final unique records to return.
            outcome: Optional outcome filter.
            balanced: Forwarded to base retrieval when searching multiple
                outcome partitions.

        Returns:
            list[RetrievedExperience]: Ranked unique retrieval results.
        """

        if top_k <= 0:
            if verbose:
                print(
                    f"[{self.__class__.__name__}.retrieve] pool='{self.pool_name}' skipped: top_k={top_k}"
                )
            return []

        if isinstance(query_text, str):
            query_texts = [query_text.strip()] if query_text.strip() else []
        elif isinstance(query_text, list):
            leaders = [q.strip() for q in query_text if isinstance(q, str) and q.strip()]

            # Build unique pair queries (order does not matter): N choose 2.
            # Use dict-from-keys for stable dedup while preserving pair generation order.
            pair_queries = [
                (
                    f"Parent Algorithm 1: {p1}\n"
                    f"Parent Algorithm 2: {p2}"
                )
                for p1, p2 in combinations(leaders, 2)
            ]
            query_texts = list(dict.fromkeys(pair_queries))
        else:
            raise TypeError(
                "CombinationExperiencePool.retrieve expects query_text as str or list[str]"
            )

        if not query_texts:
            if verbose:
                print(f"[{self.__class__.__name__}.retrieve] pool='{self.pool_name}' no valid queries")
            return []

        if verbose:
            outcome_str = outcome.value if outcome is not None else "ALL"
            print(
                f"[{self.__class__.__name__}.retrieve] pool='{self.pool_name}' "
                f"start: sub_queries={len(query_texts)}, top_k={top_k}, outcome={outcome_str}, balanced={balanced}"
            )

        best_by_record_id: Dict[str, RetrievedExperience] = {}

        for one_query in query_texts:
            hits = super().retrieve(
                query_text=one_query,
                top_k=top_k,
                outcome=outcome,
                balanced=balanced,
                verbose=False,
            )

            for hit in hits:
                prev = best_by_record_id.get(hit.record_id)
                if prev is None or hit.score > prev.score:
                    best_by_record_id[hit.record_id] = hit

        merged = list(best_by_record_id.values())
        merged.sort(key=lambda x: (x.score, x.record_id), reverse=True)
        result = merged[:top_k]
        if verbose:
            print(
                f"[{self.__class__.__name__}.retrieve] pool='{self.pool_name}' done: "
                f"unique_candidates={len(merged)}, returned={len(result)}"
            )
        return result

    def _dict_to_record(self, payload_dict: dict) -> ExperienceRecord:
        """Deserialize dictionary into `CombinationExperienceRecord`.

        Args:
            payload_dict: Persisted payload dictionary.

        Returns:
            ExperienceRecord: `CombinationExperienceRecord` instance.
        """

        normalized = dict(payload_dict)
        normalized.setdefault("parent_alg1_id", None)
        normalized.setdefault("parent_alg2_id", None)
        normalized.setdefault("new_algorithm_id", None)
        return CombinationExperienceRecord(**normalized)

    @staticmethod
    def _build_code_diff(
        parent_code: str,
        member_code: str,
        parent_label: str,
        member_label: str,
        max_chars: int = 16000,
    ) -> str:
        """Build unified diff text from parent/member code strings."""

        lines1 = parent_code.splitlines(keepends=True)
        lines2 = member_code.splitlines(keepends=True)
        diff_iter = difflib.unified_diff(
            lines1,
            lines2,
            fromfile=parent_label,
            tofile=member_label,
        )
        diff_text = "".join(diff_iter)
        # if len(diff_text) > max_chars:
        #     return diff_text[:max_chars] + "\n... [diff truncated]"
        return diff_text

    def _load_analysis_prompt_template(self, outcome: OutcomeLabel) -> str:
        """Load prompt template text from data/prompts with safe fallback."""

        prompt_path = (
            self._PROMPT_IMPROVED_FILE if outcome == OutcomeLabel.GOOD else self._PROMPT_DEGRADED_FILE
        )

        try:
            text = prompt_path.read_text(encoding="utf-8")
            if text.strip():
                return text
        except Exception:
            pass

        return (
            "You are analyzing SAT-solver crossover triplets.\n"
            "Return a valid JSON object wrapped in ```json ... ``` with exactly {{EXPECTED_COUNT}} analyses.\n"
            "Use key 'pairs' with items of shape: {\"pair_key\": \"...\", \"analysis\": \"...\"}.\n"
            "\n"
            "Batch data:\n"
            "{{BATCH_CONTENT}}\n"
        )

    def _build_analysis_prompt_for_triplet(
        self,
        candidate: Dict[str, Any],
        outcome: OutcomeLabel,
    ) -> tuple[str, str]:
        """Build one LLM prompt for a single triplet candidate."""

        prompt_template = self._load_analysis_prompt_template(outcome=outcome)

        prompt_key = "pair_1"
        diff_parent1_to_member = self._build_code_diff(
            parent_code=candidate["parent1_code"],
            member_code=candidate["member_code"],
            parent_label="parent_1_algorithm",
            member_label="child_algorithm",
        )
        diff_parent2_to_member = self._build_code_diff(
            parent_code=candidate["parent2_code"],
            member_code=candidate["member_code"],
            parent_label="parent_2_algorithm",
            member_label="child_algorithm",
        )

        batch_content = "\n\n".join(
            [
                "PAIR 1",
                f"pair_key: {prompt_key}",
                f"parent1_par2: {candidate['parent1_par2']}",
                f"parent2_par2: {candidate['parent2_par2']}",
                f"child_par2: {candidate['member_par2']}",
                "parent1_description:",
                candidate["parent1_description"],
                "parent2_description:",
                candidate["parent2_description"],
                "child_description:",
                candidate["member_description"],
                "code_diff_parent1_to_child:",
                diff_parent1_to_member if diff_parent1_to_member.strip() else "[no diff]",
                "code_diff_parent2_to_child:",
                diff_parent2_to_member if diff_parent2_to_member.strip() else "[no diff]",
            ]
        )

        prompt = (
            prompt_template.replace("{{EXPECTED_COUNT}}", "1")
            .replace("{{BATCH_CONTENT}}", batch_content)
        )
        return prompt, prompt_key

    def _parse_triplet_analysis_response(
        self,
        response_text: str,
        expected_key: str,
    ) -> Optional[str]:
        """Parse model JSON response and return analysis for one expected key."""

        json_text = self._extract_json_from_fenced_block(response_text) or response_text
        parsed = self._extract_json_object_from_text(json_text)
        if parsed is None:
            return None

        pairs = parsed.get("pairs")
        if not isinstance(pairs, list):
            return None

        for item in pairs:
            if not isinstance(item, dict):
                continue
            pair_key = item.get("pair_key")
            analysis = item.get("analysis")
            if (
                isinstance(pair_key, str)
                and pair_key == expected_key
                and isinstance(analysis, str)
                and analysis.strip()
            ):
                return analysis.strip()
        return None

    def _generate_triplet_analysis(
        self,
        candidate: Dict[str, Any],
        outcome: OutcomeLabel,
        debug: bool = False,
        max_attempts: int = 5,
    ) -> Optional[str]:
        """Generate one analysis for one selected triplet candidate."""

        prompt, expected_key = self._build_analysis_prompt_for_triplet(
            candidate=candidate,
            outcome=outcome,
        )

        if debug:
            print(f"\n[DEBUG] === COMBINATION PROMPT ({outcome.value}) ===")
            print(prompt)
            print("[DEBUG] =========================================\n")

        for attempt in range(1, max_attempts + 1):
            try:
                response_text = get_llm_response(
                    prompt=prompt,
                    system_message=(
                        "You are an expert SAT solver engineer. "
                        "Return valid JSON wrapped in ```json ... ``` only."
                    ),
                    model="gpt-5.4-2026-03-05",
                    temperature=0.7,
                )

                if debug:
                    print(
                        f"\n[DEBUG] === COMBINATION RAW RESPONSE "
                        f"({outcome.value}, attempt={attempt}) ==="
                    )
                    print(response_text)
                    print("[DEBUG] =======================================================\n")

                parsed_analysis = self._parse_triplet_analysis_response(
                    response_text=response_text,
                    expected_key=expected_key,
                )

                if debug:
                    print(
                        f"\n[DEBUG] === COMBINATION PARSED ANALYSIS "
                        f"({outcome.value}, attempt={attempt}) ==="
                    )
                    print(parsed_analysis)
                    print("[DEBUG] ========================================================\n")

                if parsed_analysis:
                    return parsed_analysis

            except Exception as exc:  # noqa: BLE001
                if debug:
                    print(
                        "[DEBUG] Combination analysis generation failed "
                        f"for triplet='{candidate['triplet_key']}', outcome='{outcome.value}', "
                        f"attempt={attempt}: {type(exc).__name__}: {exc}"
                    )

        return None

    def _get_code_string_from_aws(
        self,
        code_id: str,
        cache: Dict[str, Optional[str]],
    ) -> Optional[str]:
        """Fetch code string by code_id from AWS-backed DB."""

        if code_id in cache:
            return cache[code_id]

        try:
            result = get_code_result(code_id)
        except Exception:
            cache[code_id] = None
            return None

        code = getattr(result, "code", None) if result is not None else None
        if not isinstance(code, str) or not code.strip():
            cache[code_id] = None
            return None

        cache[code_id] = code
        return code

    def _load_parent_algorithms(
        self,
        parent_source_dir: Path,
        summary: Dict[str, Any],
        code_cache: Dict[str, Optional[str]],
    ) -> Dict[str, Dict[str, Any]]:
        """Load parent algorithms from leaders/members snapshot.

        Expected layout:
        - `<parent_source_dir>/leaders/algorithm_<id>/<id>.json`
        - `<parent_source_dir>/members/algorithm_<id>/<id>.json`

        Leaders are loaded first; members only fill missing IDs.

        Args:
            parent_source_dir: Root directory containing `leaders/` and `members/`.
            summary: Mutable update summary dictionary.

        Returns:
            dict[str, dict[str, Any]]: Parent metadata keyed by algorithm ID.
        """

        leaders_dir = parent_source_dir / "leaders"
        members_dir = parent_source_dir / "members"

        if not leaders_dir.exists() or not leaders_dir.is_dir():
            raise ValueError(f"Missing leaders directory: {leaders_dir}")
        if not members_dir.exists() or not members_dir.is_dir():
            raise ValueError(f"Missing members directory: {members_dir}")

        parent_map: Dict[str, Dict[str, Any]] = {}

        def _load_one_partition(partition_dir: Path, partition_name: str) -> None:
            for algo_dir, folder_algo_id in self._iter_algorithm_dirs(partition_dir):
                summary["parents_seen"] += 1

                json_path = algo_dir / f"{folder_algo_id}.json"
                if not json_path.exists():
                    summary["invalid_skipped"] += 1
                    continue

                payload = self._safe_load_json(json_path)
                if payload is None:
                    summary["invalid_skipped"] += 1
                    continue

                algo_id = payload.get("id")
                description = payload.get("description")
                code_id = self._extract_singleton_str_from_fields(payload, ("code_id_list",))

                if code_id is None:
                    summary["invalid_skipped"] += 1
                    continue

                code = self._get_code_string_from_aws(code_id=code_id, cache=code_cache)
                if code is None:
                    summary["missing_code_skipped"] += 1
                    continue

                if not self._is_valid_score_list(payload, "raw_par2_score"):
                    summary["invalid_skipped"] += 1
                    continue

                score = self._extract_representative_par2(payload)

                if not self._is_nonempty_str(algo_id):
                    summary["invalid_skipped"] += 1
                    continue

                if algo_id != folder_algo_id:
                    err = (
                        "[CombinationExperiencePool.update][ERROR] "
                        f"ID mismatch at {json_path}. Folder expects '{folder_algo_id}' but payload id is '{algo_id}'."
                    )
                    print(err)
                    summary["errors"].append(err)
                    summary["invalid_skipped"] += 1
                    continue

                if not self._is_nonempty_str(description):
                    err = (
                        "[CombinationExperiencePool.update][ERROR] "
                        f"Empty description for parent id '{algo_id}' at {json_path}."
                    )
                    print(err)
                    summary["errors"].append(err)
                    summary["invalid_skipped"] += 1
                    continue

                if score is None:
                    err = (
                        "[CombinationExperiencePool.update][ERROR] "
                        f"Invalid par2 score found for parent at {json_path}: {score}"
                    )
                    print(err)
                    summary["errors"].append(err)
                    summary["invalid_skipped"] += 1
                    continue

                if algo_id in parent_map:
                    # Keep first-seen record (leaders are loaded before members).
                    continue

                parent_map[algo_id] = {
                    "id": algo_id,
                    "code_id": code_id,
                    "code": code,
                    "description": description,
                    "score": score,
                    "json_path": str(json_path),
                    "partition": partition_name,
                }
                summary["parents_loaded"] += 1

        _load_one_partition(leaders_dir, "leaders")
        _load_one_partition(members_dir, "members")
        return parent_map

    def update(
        self,
        combined_dir: str | Path,
        parent_source_dir: str | Path,
        threshold: float = 0.10,
        top_k_good: int = 5,
        top_k_bad: int = 5,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """Parse combination artifacts and persist good/bad combination experiences.

        Expected layouts:
        - Combined offspring directory:
          `<combined_dir>/leaders/algorithm_<id>/<id>.json`
        - Parent source directory (same schema as mutation input):
          `<parent_source_dir>/leaders/algorithm_<id>/<id>.json`
          `<parent_source_dir>/members/algorithm_<id>/<id>.json`

                For each offspring member:
                - Track valid members with:
                    - singleton code-id list field (`code_id_list`)
                    - `parent_id` list with exactly 2 entries
                    - valid `raw_par2_score` list (5 finite numbers)
                    - non-empty `description`
                - Resolve both parents from `parent_source_dir` and validate them with
                    the same required fields except `parent_id` cardinality.
                - GOOD triplet: member PAR2 is smaller than both parents.
                - BAD triplet: member PAR2 is larger than both parents.
                - Rank GOOD by max relative percentage decrease against the two parents.
                - Rank BAD by max relative percentage increase against the two parents.
                - Select top-K from each side and generate one LLM analysis per triplet.

        Args:
            combined_dir: Directory containing combined algorithms under `leaders/`.
            parent_source_dir: Directory containing historical parent algorithms
                under `leaders/` and `members/`.
                        threshold: Backward-compatible argument kept for API stability.
                                Not used in current ranking/classification.
                        top_k_good: Number of top-ranked GOOD triplets selected.
                        top_k_bad: Number of top-ranked BAD triplets selected.
                        debug: Print prompt/response and selected candidates when `True`.

        Returns:
            dict[str, Any]: Update summary counters and diagnostics.

        Raises:
            ValueError: If required directories are missing.
        """

        combined_root = Path(combined_dir)
        parent_root = Path(parent_source_dir)
        print(
            f"[CombinationExperiencePool.update] start: combined_dir='{combined_root}', "
            f"parent_source_dir='{parent_root}', top_k_good={max(0, int(top_k_good))}, "
            f"top_k_bad={max(0, int(top_k_bad))}"
        )

        combined_members_dir = combined_root / "leaders"
        if not combined_members_dir.exists() or not combined_members_dir.is_dir():
            raise ValueError(f"Missing combined leaders directory: {combined_members_dir}")

        summary: Dict[str, Any] = {
            "combined_dir": str(combined_root),
            "parent_source_dir": str(parent_root),
            "threshold": threshold,
            "top_k_good": max(0, int(top_k_good)),
            "top_k_bad": max(0, int(top_k_bad)),
            "parents_seen": 0,
            "parents_loaded": 0,
            "combined_seen": 0,
            "combined_loaded": 0,
            "good_candidates": 0,
            "bad_candidates": 0,
            "selected_good": 0,
            "selected_bad": 0,
            "generated_good": 0,
            "generated_bad": 0,
            "analysis_failed": 0,
            "persisted_good": 0,
            "persisted_bad": 0,
            "persisted_created": 0,
            "persisted_deduped": 0,
            "neutral_skipped": 0,
            "invalid_skipped": 0,
            "missing_code_skipped": 0,
            "missing_parent_skipped": 0,
            "parent_cardinality_errors": 0,
            "errors": [],
        }

        code_cache: Dict[str, Optional[str]] = {}
        parent_map = self._load_parent_algorithms(
            parent_source_dir=parent_root,
            summary=summary,
            code_cache=code_cache,
        )

        good_candidates: List[Dict[str, Any]] = []
        bad_candidates: List[Dict[str, Any]] = []

        for algo_dir, folder_algo_id in self._iter_algorithm_dirs(combined_members_dir):
            summary["combined_seen"] += 1

            json_path = algo_dir / f"{folder_algo_id}.json"
            if not json_path.exists():
                summary["invalid_skipped"] += 1
                continue

            payload = self._safe_load_json(json_path)
            if payload is None:
                summary["invalid_skipped"] += 1
                continue

            offspring_id = payload.get("id")
            offspring_desc = payload.get("description")
            offspring_code_id = self._extract_singleton_str_from_fields(payload, ("code_id_list",))

            if offspring_code_id is None:
                print(f"[CombinationExperiencePool.update][ERROR] Missing code_id_list for offspring at {json_path}")
                summary["invalid_skipped"] += 1
                continue

            offspring_code = self._get_code_string_from_aws(code_id=offspring_code_id, cache=code_cache)
            if offspring_code is None:
                print(f"[CombinationExperiencePool.update][ERROR] Missing code for offspring code_id '{offspring_code_id}' at {json_path}")
                summary["missing_code_skipped"] += 1
                continue

            if not self._is_valid_score_list(payload, "raw_par2_score"):
                print(f"[CombinationExperiencePool.update][ERROR] Invalid or missing raw_par2_score list for offspring at {json_path}")
                summary["invalid_skipped"] += 1
                continue

            offspring_score = self._extract_representative_par2(payload)
            parent_id_raw = payload.get("parent_id")

            if not self._is_nonempty_str(offspring_id):
                summary["invalid_skipped"] += 1
                continue

            if offspring_id != folder_algo_id:
                err = (
                    "[CombinationExperiencePool.update][ERROR] "
                    f"ID mismatch at {json_path}. Folder expects '{folder_algo_id}' but payload id is '{offspring_id}'."
                )
                print(err)
                summary["errors"].append(err)
                summary["invalid_skipped"] += 1
                continue

            if not self._is_nonempty_str(offspring_desc):
                err = (
                    "[CombinationExperiencePool.update][ERROR] "
                    f"Empty description for offspring at {json_path}"
                )
                print(err)
                summary["errors"].append(err)
                summary["invalid_skipped"] += 1
                continue

            if offspring_score is None:
                err = (
                    "[CombinationExperiencePool.update][ERROR] "
                    f"Invalid par2 score found for combined algorithm at {json_path}: {offspring_score}"
                )
                print(err)
                summary["errors"].append(err)
                summary["invalid_skipped"] += 1
                continue

            if not isinstance(parent_id_raw, list):
                err = (
                    "[CombinationExperiencePool.update][ERROR] Invalid parent_id type. "
                    f"Expected list with exactly 2 parent_id entries for combined id '{offspring_id}' "
                    f"at {json_path}. parent_id={parent_id_raw!r}"
                )
                print(err)
                summary["errors"].append(err)
                summary["parent_cardinality_errors"] += 1
                summary["invalid_skipped"] += 1
                continue

            parent_ids = [x for x in parent_id_raw if isinstance(x, str) and x.strip()]
            if len(parent_ids) != 2:
                err = (
                    "[CombinationExperiencePool.update][ERROR] Invalid parent cardinality. "
                    f"Expected exactly 2 parent_id entries, got {len(parent_ids)} for combined id '{offspring_id}' "
                    f"at {json_path}. parent_id={parent_id_raw!r}"
                )
                print(err)
                summary["errors"].append(err)
                summary["parent_cardinality_errors"] += 1
                summary["invalid_skipped"] += 1
                continue

            parent1 = parent_map.get(parent_ids[0])
            parent2 = parent_map.get(parent_ids[1])
            if parent1 is None or parent2 is None:
                summary["missing_parent_skipped"] += 1
                continue

            parent1_score = float(parent1["score"])
            parent2_score = float(parent2["score"])
            child_score = float(offspring_score)

            summary["combined_loaded"] += 1

            better_than_both = child_score < parent1_score and child_score < parent2_score
            worse_than_both = child_score > parent1_score and child_score > parent2_score

            improvement_vs_parent1 = self._relative_change(parent1_score, child_score, improve=True)
            improvement_vs_parent2 = self._relative_change(parent2_score, child_score, improve=True)
            degradation_vs_parent1 = self._relative_change(parent1_score, child_score, improve=False)
            degradation_vs_parent2 = self._relative_change(parent2_score, child_score, improve=False)

            if (
                improvement_vs_parent1 is None
                or improvement_vs_parent2 is None
                or degradation_vs_parent1 is None
                or degradation_vs_parent2 is None
            ):
                summary["invalid_skipped"] += 1
                continue

            if not better_than_both and not worse_than_both:
                summary["neutral_skipped"] += 1
                continue

            triplet = {
                "triplet_key": f"{parent1['id']}__{parent2['id']}__{offspring_id}",
                "member_algorithm_id": offspring_id,
                "member_code_id": offspring_code_id,
                "member_code": offspring_code,
                "member_description": offspring_desc,
                "member_par2": child_score,
                "parent1_algorithm_id": parent1["id"],
                "parent1_code_id": parent1["code_id"],
                "parent1_code": parent1["code"],
                "parent1_description": parent1["description"],
                "parent1_par2": parent1_score,
                "parent2_algorithm_id": parent2["id"],
                "parent2_code_id": parent2["code_id"],
                "parent2_code": parent2["code"],
                "parent2_description": parent2["description"],
                "parent2_par2": parent2_score,
            }

            if better_than_both:
                triplet["relative_change_parent1"] = float(improvement_vs_parent1)
                triplet["relative_change_parent2"] = float(improvement_vs_parent2)
                triplet["relative_change"] = max(float(improvement_vs_parent1), float(improvement_vs_parent2))
                good_candidates.append(triplet)
            else:
                triplet["relative_change_parent1"] = float(degradation_vs_parent1)
                triplet["relative_change_parent2"] = float(degradation_vs_parent2)
                triplet["relative_change"] = max(float(degradation_vs_parent1), float(degradation_vs_parent2))
                bad_candidates.append(triplet)

        good_candidates.sort(
            key=lambda x: (-x["relative_change"], x["member_algorithm_id"], x["triplet_key"])
        )
        bad_candidates.sort(
            key=lambda x: (-x["relative_change"], x["member_algorithm_id"], x["triplet_key"])
        )

        summary["good_candidates"] = len(good_candidates)
        summary["bad_candidates"] = len(bad_candidates)

        selected_good = good_candidates[: max(0, int(top_k_good))]
        selected_bad = bad_candidates[: max(0, int(top_k_bad))]
        summary["selected_good"] = len(selected_good)
        summary["selected_bad"] = len(selected_bad)

        print(
            f"[CombinationExperiencePool.update] generation stage: "
            f"selected_good={len(selected_good)}, selected_bad={len(selected_bad)}"
        )

        if debug:
            print("\n[DEBUG] Selected GOOD triplets:")
            for cand in selected_good:
                print(
                    f" - {cand['triplet_key']}: "
                    f"member={cand['member_algorithm_id']}, rel_change={cand['relative_change']:.4f}"
                )
            print("\n[DEBUG] Selected BAD triplets:")
            for cand in selected_bad:
                print(
                    f" - {cand['triplet_key']}: "
                    f"member={cand['member_algorithm_id']}, rel_change={cand['relative_change']:.4f}"
                )
            print()

        def _persist_selected(candidates: List[Dict[str, Any]], outcome: OutcomeLabel) -> None:
            for cand in candidates:
                generated_analysis = self._generate_triplet_analysis(
                    candidate=cand,
                    outcome=outcome,
                    debug=debug,
                )

                if not generated_analysis:
                    summary["analysis_failed"] += 1
                    continue

                if outcome == OutcomeLabel.GOOD:
                    summary["generated_good"] += 1
                else:
                    summary["generated_bad"] += 1

                record = CombinationExperienceRecord(
                    parent_alg1_description=cand["parent1_description"],
                    parent_alg2_description=cand["parent2_description"],
                    new_algorithm_description=cand["member_description"],
                    analysis=generated_analysis,
                    parent_alg1_id=cand["parent1_algorithm_id"],
                    parent_alg2_id=cand["parent2_algorithm_id"],
                    new_algorithm_id=cand["member_algorithm_id"],
                )

                receipt = self.persist(record=record, outcome=outcome)
                if outcome == OutcomeLabel.GOOD:
                    summary["persisted_good"] += 1
                else:
                    summary["persisted_bad"] += 1

                if receipt.created:
                    summary["persisted_created"] += 1
                else:
                    summary["persisted_deduped"] += 1

        _persist_selected(selected_good, OutcomeLabel.GOOD)
        _persist_selected(selected_bad, OutcomeLabel.BAD)

        print("\n[CombinationExperiencePool] Update Summary:")
        print(
            f"  - Parents loaded: {summary['parents_loaded']} "
            f"({summary['invalid_skipped']} total skipped, {summary['missing_code_skipped']} missing code)"
        )
        print(
            f"  - Members processed: {summary['combined_loaded']} "
            f"({summary['neutral_skipped']} neutral, {summary['missing_parent_skipped']} missing parent)"
        )
        print(
            f"  - Triplet candidates: {summary['good_candidates']} GOOD, {summary['bad_candidates']} BAD"
        )
        print(
            f"  - Selected for generation: {summary['selected_good']} GOOD, {summary['selected_bad']} BAD"
        )
        print(
            f"  - Analyses generated: {summary['generated_good']} GOOD, {summary['generated_bad']} BAD "
            f"({summary['analysis_failed']} failed)"
        )
        print(
            f"  - Persisted experiences: {summary['persisted_good']} GOOD, {summary['persisted_bad']} BAD"
        )
        print(
            f"  - Persistence details: {summary['persisted_created']} created, "
            f"{summary['persisted_deduped']} deduped"
        )

        return summary
