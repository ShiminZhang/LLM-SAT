"""Streaming parallel pipeline for mutant generation + build + SLURM submission.

Replaces the sequential flow of:
  1. Generate all algorithm variants (API calls)
  2. Generate all code (API calls)
  3. Build all solvers
  4. Submit one SLURM batch

With a streaming pipeline where each algorithm flows through
algo_gen -> code_gen -> build -> submit as soon as it's ready,
overlapping work across all stages.

Usage:
    from llmsat.pipelines.parallel_orchestrator import run_streaming_mutants
    run_streaming_mutants(leaders, variant_prompt_template, code_prompt_template,
                          generation_tag, ...)
"""

import asyncio
import json
import os
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from llmsat.llmsat import (
    CHATGPT_DATA_GENERATION_TABLE,
    AlgorithmResult,
    AlgorithmStatus,
    CodeResult,
    CodeStatus,
    Role,
    NOT_INITIALIZED,
    get_id,
    get_logger,
    SAT2025_BENCHMARK_PATH,
)
from llmsat.utils.gemini_helper import get_response_from_gemini
from llmsat.utils.aws import (
    get_algorithm_result,
    get_ids_from_router_table,
    update_algorithm_result,
    update_code_result,
    update_router_table,
    append_code_id,
)
from llmsat.utils.utils import atomic_append
from llmsat.utils.paths import (
    get_solver_dir,
    get_solver_result_dir,
    get_generation_output_dir,
)
from llmsat.pipelines.gemini_data_generation import (
    count_steps,
    extract_step_text,
    generate_code_prompt,
    parse_algorithm_response,
    parse_code_response,
    _save_timing_log,
)
from llmsat.pipelines.evaluation import (
    EvaluationPipeline,
    QUICK_EVAL_TIMEOUT_SECONDS,
    QUICK_EVAL_WALL_TIME,
    QUICK_EVAL_PAR2_PENALTY,
    QUICK_EVAL_BENCHMARK_LIST,
)
from llmsat.config import DEFAULT_MODEL, EXPERIENCE_POOL_DATA_ROOT

try:
    from experience_pool import ExperiencePoolManager as _ExperiencePoolManager
    _EXPERIENCE_POOL_AVAILABLE = True
except ImportError:
    _ExperiencePoolManager = None
    _EXPERIENCE_POOL_AVAILABLE = False

logger = get_logger(__name__)

# --- Concurrency constants ---
MAX_API_CONCURRENT = 10  # Max in-flight Gemini API calls (shared across stages 1 & 2)
MAX_BUILD_CONCURRENT = 3  # Max concurrent ./configure && make on login node
MAX_DB_CONCURRENT = 5     # Max concurrent DB connections

_SENTINEL = object()  # Signals end of queue

NERSC_SLURM_THRESHOLD = 128  # solvers × CNFs before NERSC batch submission


class _SlurmTaskAccumulator:
    """
    Thread-safe accumulator for SLURM solver evaluation tasks (NERSC only).

    Batches built solver tasks and submits them via slurm_run_evaluate_batch
    once accumulated task count (solvers × CNFs) reaches slurm_threshold.
    Remaining tasks are submitted unconditionally by flush_all().
    """

    def __init__(
        self,
        pipeline: Any,
        cnf_files: List[str],
        slurm_threshold: int,
        benchmark_path: str,
    ) -> None:
        self.pipeline = pipeline
        self.cnf_files = cnf_files
        self.slurm_threshold = slurm_threshold
        self.benchmark_path = benchmark_path
        self._solver_tasks: List[Tuple[str, str, str]] = []
        self._lock = threading.Lock()
        self._job_ids: List[int] = []
        self._tasks_per_solver: int = len(cnf_files)

    def add_solver(self, solver_path: str, result_dir: str, code_id: str) -> List[int]:
        """Add a built solver; submits a SLURM batch when the threshold is reached.

        Returns:
            List of newly submitted SLURM job IDs (empty if no submission happened).
        """
        submitted_ids: List[int] = []
        with self._lock:
            self._solver_tasks.append((solver_path, result_dir, code_id))
            if self._tasks_per_solver > 0:
                if len(self._solver_tasks) * self._tasks_per_solver >= self.slurm_threshold:
                    to_submit = list(self._solver_tasks)
                    self._solver_tasks = []
                    submitted_ids = self._submit_batch(to_submit)
        return submitted_ids

    def _submit_batch(self, solver_tasks: List[Tuple[str, str, str]]) -> List[int]:
        ids = self.pipeline.slurm_run_evaluate_batch(
            solver_tasks=solver_tasks,
            benchmark_path=self.benchmark_path,
            cnf_files=self.cnf_files,
        )
        self._job_ids.extend(ids)
        logger.info(
            f"[Accumulator] Submitted {len(solver_tasks)} solver(s) "
            f"({len(solver_tasks) * self._tasks_per_solver} tasks) → SLURM IDs: {ids}"
        )
        return ids

    def flush_all(self) -> List[int]:
        """Submit any remaining solvers unconditionally. Returns all SLURM job IDs."""
        with self._lock:
            if self._solver_tasks:
                logger.info(
                    f"[Accumulator] Flushing {len(self._solver_tasks)} remaining solver(s)"
                )
                self._submit_batch(self._solver_tasks)
                self._solver_tasks = []
        return list(self._job_ids)

    def get_job_ids(self) -> List[int]:
        """Return a copy of all SLURM IDs submitted through this accumulator."""
        with self._lock:
            return list(self._job_ids)


class ParallelPipeline:
    """Three-stage streaming pipeline: algo gen -> code gen -> build + submit."""

    def __init__(
        self,
        generation_tag: str,
        model: str = DEFAULT_MODEL,
        quick_eval: bool = True,
        nersc: bool = False,
    ):
        self.generation_tag = generation_tag
        self.model = model
        self.quick_eval = quick_eval
        self.nersc = nersc

        # Queues between stages
        self.code_queue: asyncio.Queue = asyncio.Queue()
        self.build_queue: asyncio.Queue = asyncio.Queue()

        # Concurrency controls
        self.api_semaphore = asyncio.Semaphore(MAX_API_CONCURRENT)
        self.build_semaphore = asyncio.Semaphore(MAX_BUILD_CONCURRENT)

        # Thread pools for blocking calls
        self.api_pool = ThreadPoolExecutor(max_workers=MAX_API_CONCURRENT, thread_name_prefix="api")
        self.build_pool = ThreadPoolExecutor(max_workers=MAX_BUILD_CONCURRENT, thread_name_prefix="build")
        self.db_pool = ThreadPoolExecutor(max_workers=MAX_DB_CONCURRENT, thread_name_prefix="db")

        # Evaluation pipeline (reuse existing build_solver / slurm_run_evaluate)
        if nersc:
            from llmsat.pipelines.evaluation_nersc import (
                EvaluationPipeline as NerscEvaluationPipeline,
                QUICK_EVAL_TIMEOUT_SECONDS as NERSC_QUICK_EVAL_TIMEOUT_SECONDS,
                QUICK_EVAL_WALL_TIME as NERSC_QUICK_EVAL_WALL_TIME,
                QUICK_EVAL_PAR2_PENALTY as NERSC_QUICK_EVAL_PAR2_PENALTY,
                QUICK_EVAL_BENCHMARK_LIST as NERSC_QUICK_EVAL_BENCHMARK_LIST,
            )
            self.eval_pipeline = NerscEvaluationPipeline(generation_tag=generation_tag)
            if quick_eval:
                self.eval_pipeline.timeout = NERSC_QUICK_EVAL_TIMEOUT_SECONDS
                self.eval_pipeline.wall_time = NERSC_QUICK_EVAL_WALL_TIME
                self.eval_pipeline.par2_penalty = NERSC_QUICK_EVAL_PAR2_PENALTY
                benchmark_list = NERSC_QUICK_EVAL_BENCHMARK_LIST
        else:
            self.eval_pipeline = EvaluationPipeline(generation_tag=generation_tag)
            if quick_eval:
                self.eval_pipeline.timeout = QUICK_EVAL_TIMEOUT_SECONDS
                self.eval_pipeline.wall_time = QUICK_EVAL_WALL_TIME
                self.eval_pipeline.par2_penalty = QUICK_EVAL_PAR2_PENALTY
                benchmark_list = QUICK_EVAL_BENCHMARK_LIST
        if quick_eval:
            if os.path.exists(benchmark_list):
                with open(benchmark_list) as f:
                    self.eval_pipeline.cnf_files = [line.strip() for line in f if line.strip()]
                logger.info(f"Quick-eval mode: {len(self.eval_pipeline.cnf_files)} CNFs, "
                            f"{self.eval_pipeline.timeout}s timeout")
            else:
                logger.warning(f"Quick-eval benchmark list not found: {benchmark_list}")

        # NERSC accumulator: batch solvers until threshold before SLURM submit
        if nersc:
            benchmark_path = getattr(self.eval_pipeline, 'benchmark_path', None) or SAT2025_BENCHMARK_PATH
            self.accumulator: Optional[_SlurmTaskAccumulator] = _SlurmTaskAccumulator(
                pipeline=self.eval_pipeline,
                cnf_files=self.eval_pipeline.cnf_files,
                slurm_threshold=NERSC_SLURM_THRESHOLD,
                benchmark_path=benchmark_path,
            )
        else:
            self.accumulator = None

        # Job tracking
        self.output_dir = get_generation_output_dir(generation_tag)
        os.makedirs(self.output_dir, exist_ok=True)
        self.job_ids_path = os.path.join(self.output_dir, "submitted_job_ids.jsonl")

        # Counters
        self.leaders_generated = 0
        self.algos_generated = 0
        self.codes_generated = 0
        self.builds_succeeded = 0
        self.builds_failed = 0
        self.slurm_submitted = 0

        # Timing
        self._timing_lists: Dict[str, List[float]] = {}
        self._timing_lock = threading.Lock()
        self._stage_spans: Dict[str, List[float]] = {}  # stage -> [earliest_start, latest_end]
        self._pipeline_start: Optional[float] = None

        self.system_message = os.environ.get(
            "LLMSAT_SYSTEM_MESSAGE",
            "You are an AI researcher specialising in SAT solver heuristics.",
        )

        # Experience pool (optional — degrades gracefully if unavailable)
        self.exp_pool_manager = None
        self._mutation_pool_disabled = (
            os.environ.get("MUTATION_POOL", "1").strip() == "0"
        )
        if self._mutation_pool_disabled:
            logger.info(
                "[parallel] MUTATION_POOL=0 — mutation experience pool disabled by env var"
            )
        elif _EXPERIENCE_POOL_AVAILABLE and EXPERIENCE_POOL_DATA_ROOT is not None:
            try:
                self.exp_pool_manager = _ExperiencePoolManager(
                    data_root=EXPERIENCE_POOL_DATA_ROOT
                )
                logger.info(
                    f"[parallel] Experience pool initialized at: {EXPERIENCE_POOL_DATA_ROOT}"
                )
            except Exception as e:
                logger.warning(
                    f"[parallel] Experience pool init failed (pool disabled): {e}"
                )
                self.exp_pool_manager = None

    # ------------------------------------------------------------------ #
    # Timing helpers
    # ------------------------------------------------------------------ #

    def _record_item_time(self, stage: str, start: float, end: float):
        elapsed = end - start
        with self._timing_lock:
            self._timing_lists.setdefault(stage, []).append(elapsed)
            span = self._stage_spans.get(stage)
            if span is None:
                self._stage_spans[stage] = [start, end]
            else:
                if start < span[0]: span[0] = start
                if end > span[1]: span[1] = end

    def _build_timing_dict(self, mode: str, eval_wait_s: float = 0.0) -> Dict[str, Any]:
        pipeline_end = time.time()
        stages = {}
        for stage, times in self._timing_lists.items():
            span = self._stage_spans.get(stage, [0, 0])
            stages[stage] = {
                "n": len(times),
                "avg_s": round(sum(times) / len(times), 2),
                "min_s": round(min(times), 2),
                "max_s": round(max(times), 2),
                "total_s": round(sum(times), 2),
                "wall_s": round(span[1] - span[0], 2),
            }
        pipeline_wall = round(pipeline_end - self._pipeline_start, 2) if self._pipeline_start else 0.0
        return {
            "generation_tag": self.generation_tag,
            "timestamp": datetime.now().isoformat(),
            "script": "parallel_orchestrator",
            "mode": mode,
            "stages": stages,
            "pipeline_wall_s": pipeline_wall,
            "eval_wait_s": round(eval_wait_s, 2),
            "total_s": round(pipeline_wall + eval_wait_s, 2),
            "counts": {
                "leaders": self.leaders_generated, "algos": self.algos_generated,
                "codes": self.codes_generated, "builds_ok": self.builds_succeeded,
                "builds_failed": self.builds_failed, "slurm_submitted": self.slurm_submitted,
            },
        }

    # ------------------------------------------------------------------ #
    # Stage 1: Algorithm (variant) generation
    # ------------------------------------------------------------------ #

    def _build_experience_section(self, leader_algorithm: str, target_step: int) -> str:
        """Query mutation pool and return a formatted section for prompt injection.

        Returns "None" when MUTATION_POOL=0 (explicit env-var opt-out, makes the
        ablation visible in saved prompts). Returns "" when the pool is unavailable
        for other reasons (no manager, no hits, errors) — so the prompt degrades
        cleanly to zero-shot in those cases.
        """
        if self._mutation_pool_disabled:
            return "None"
        if self.exp_pool_manager is None:
            return ""

        step_text = extract_step_text(leader_algorithm, target_step)
        query = (
            f"Leader Algorithm Description: {leader_algorithm}\n"
            f"Mutation Step: {step_text}"
        )

        try:
            logger.info(
                f"[exp_pool] Querying mutation pool for step {target_step} "
                f"(leader_len={len(leader_algorithm)})"
            )
            res = self.exp_pool_manager.search_experience_pool(
                pool_name="mutation",
                query_text=query,
                retrieve_good_k=3,
                retrieve_bad_k=3,
                sample_good_k=0,
                sample_bad_k=0,
            )

            good_hits = res.good.unique
            bad_hits = res.bad.unique

            logger.info(
                f"[exp_pool] Retrieved {len(good_hits)} good, "
                f"{len(bad_hits)} bad mutation examples (step {target_step})"
            )

            if not good_hits and not bad_hits:
                return ""

            lines = [
                "### Past Mutation Experience",
                "",
                "The following examples come from previous iterations of this evolutionary "
                "search. Use them to guide your mutation: learn from the structure of good "
                "mutations and avoid the failure modes of bad ones.",
            ]

            if good_hits:
                lines += ["", "#### Good Mutations (what worked)", ""]
                for i, hit in enumerate(good_hits, start=1):
                    rec = hit.payload  # MutationExperienceRecord
                    lines += [
                        f"**Example {i}**",
                        f"- Mutation step: {rec.step}",
                        f"- Resulting variant: {rec.member_algorithm_description}",
                        f"- Why it improved: {rec.analysis}",
                        "",
                    ]

            if bad_hits:
                lines += ["#### Bad Mutations (what to avoid)", ""]
                for i, hit in enumerate(bad_hits, start=1):
                    rec = hit.payload  # MutationExperienceRecord
                    lines += [
                        f"**Example {i}**",
                        f"- Mutation step: {rec.step}",
                        f"- Resulting variant: {rec.member_algorithm_description}",
                        f"- Why it failed: {rec.analysis}",
                        "",
                    ]

            return "\n".join(lines)

        except Exception as e:
            logger.warning(f"[exp_pool] Experience pool query failed (non-fatal): {e}")
            return ""

    async def _generate_one_variant(
        self,
        leader_id: str,
        leader_result: AlgorithmResult,
        target_step: int,
        variant_prompt_template: str,
        variant_idx: int,
        total_variants: int,
    ):
        """Generate a single mutant variant and push to code_queue."""
        leader_algorithm = leader_result.description
        experience_section = self._build_experience_section(leader_algorithm, target_step)
        prompt = variant_prompt_template.replace("{leader_algorithm}", leader_algorithm)
        prompt = prompt.replace("{target_step_num}", str(target_step))
        prompt = prompt.replace("{experience_pool_section}", experience_section)

        loop = asyncio.get_event_loop()
        try:
            _t0 = time.time()
            async with self.api_semaphore:
                raw_text = await loop.run_in_executor(
                    self.api_pool,
                    lambda: get_response_from_gemini(
                        prompt,
                        system_message=self.system_message,
                        model=self.model,
                    ),
                )
            self._record_item_time("algo_gen", _t0, time.time())

            member_desc, _, member_reason = parse_algorithm_response({"text": raw_text})
            member_id = get_id(member_desc)

            algorithm = AlgorithmResult(
                id=member_id,
                function_name=leader_result.function_name,
                description=member_desc,
                role=Role.MEMBER,
                status=AlgorithmStatus.Generated,
                last_updated=datetime.now(),
                code_id_list=[],
                parent_id=[leader_id],
                parent_algorithm_description=[leader_algorithm],
                analysis=member_reason,
                prompt=variant_prompt_template,
                mutation_step=extract_step_text(leader_algorithm, target_step),
            )

            # Record mutation step to disk so evaluation can patch it after DB round-trip
            try:
                map_path = os.path.join(
                    get_generation_output_dir(self.generation_tag),
                    "mutation_step_map.jsonl",
                )
                atomic_append(map_path, json.dumps({member_id: extract_step_text(leader_algorithm, target_step)}) + "\n")
            except Exception as e:
                logger.warning(f"[parallel] Failed to record mutation step for {member_id}: {e}")

            # DB writes
            await loop.run_in_executor(
                self.db_pool,
                lambda: update_router_table(
                    CHATGPT_DATA_GENERATION_TABLE, member_id, self.generation_tag
                ),
            )
            await loop.run_in_executor(
                self.db_pool,
                lambda: update_algorithm_result(algorithm),
            )

            self.algos_generated += 1
            logger.info(
                f"[parallel] Generated variant {self.algos_generated} "
                f"(leader {leader_id[:8]}..., step {target_step})"
            )

            # Push to code generation stage
            await self.code_queue.put(algorithm)

        except Exception as e:
            logger.error(
                f"[parallel] Failed to generate variant for leader {leader_id[:8]}... "
                f"step {target_step}: {e}"
            )

    async def _generate_all_variants(
        self,
        leaders: Dict[str, AlgorithmResult],
        variant_prompt_template: str,
        m_variants_per_leader: int,
        existing_members_by_leader: Dict[str, int],
    ):
        """Launch all variant generation tasks, then signal completion."""
        tasks = []

        for leader_id, leader_result in leaders.items():
            existing_count = existing_members_by_leader.get(leader_id, 0)
            if existing_count >= m_variants_per_leader:
                logger.info(f"Leader {leader_id[:8]}... already has {existing_count} members, skipping")
                continue

            leader_algorithm = leader_result.description
            num_steps = count_steps(leader_algorithm)
            if num_steps == 0:
                logger.warning(f"Leader {leader_id[:8]}... has no step markers, skipping")
                continue

            needed = m_variants_per_leader - existing_count
            if num_steps < m_variants_per_leader:
                step_assignments = [
                    (i + 1) if i < num_steps else num_steps
                    for i in range(m_variants_per_leader)
                ]
            else:
                start_step = num_steps - m_variants_per_leader + 1
                step_assignments = [start_step + i for i in range(m_variants_per_leader)]

            step_assignments = step_assignments[existing_count:]

            for j, target_step in enumerate(step_assignments):
                tasks.append(
                    self._generate_one_variant(
                        leader_id=leader_id,
                        leader_result=leader_result,
                        target_step=target_step,
                        variant_prompt_template=variant_prompt_template,
                        variant_idx=existing_count + j,
                        total_variants=m_variants_per_leader,
                    )
                )

        logger.info(f"[parallel] Launching {len(tasks)} variant generation tasks")

        # Run all concurrently; semaphore limits actual in-flight API calls
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log any exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"[parallel] Variant task {i} raised: {result}")

        # Signal code_gen workers that no more algorithms are coming
        await self.code_queue.put(_SENTINEL)
        logger.info(f"[parallel] Algorithm generation complete: {self.algos_generated} generated")

    # ------------------------------------------------------------------ #
    # Stage 0 (init mode): Leader generation
    # ------------------------------------------------------------------ #

    async def _generate_one_leader(
        self,
        leader_idx: int,
        n_leaders: int,
        designer_prompt: str,
        variant_prompt_template: str,
        m_variants_per_leader: int,
    ):
        """Generate one leader, push to code_queue, then generate all its variants.

        Each leader task is self-contained: generate leader → push leader to
        code_queue → spawn variant tasks → wait for variants. This means
        variant generation starts as soon as the leader is ready, overlapping
        with other leaders still being generated.
        """
        temperature = 0.5 + (1.0 - 0.5) * leader_idx / max(n_leaders - 1, 1)
        loop = asyncio.get_event_loop()

        try:
            _t0 = time.time()
            async with self.api_semaphore:
                raw_text = await loop.run_in_executor(
                    self.api_pool,
                    lambda t=temperature: get_response_from_gemini(
                        designer_prompt,
                        system_message=self.system_message,
                        model=self.model,
                        temperature=t,
                    ),
                )
            self._record_item_time("algo_gen", _t0, time.time())

            description, target_function, reason = parse_algorithm_response({"text": raw_text})
            leader_id = get_id(description)
            fn = target_function or "kissat_restarting"

            leader_result = AlgorithmResult(
                id=leader_id,
                function_name=fn,
                description=description,
                role=Role.LEADER,
                status=AlgorithmStatus.Generated,
                last_updated=datetime.now(),
                code_id_list=[],
                parent_id=None,
                analysis=reason,
                prompt=designer_prompt,
            )

            # DB writes
            await loop.run_in_executor(
                self.db_pool,
                lambda: update_router_table(
                    CHATGPT_DATA_GENERATION_TABLE, leader_id, self.generation_tag
                ),
            )
            await loop.run_in_executor(
                self.db_pool,
                lambda: update_algorithm_result(leader_result),
            )

            self.leaders_generated += 1
            logger.info(
                f"[parallel] Generated leader {self.leaders_generated}/{n_leaders} "
                f"({leader_id[:8]}..., temp={temperature:.2f})"
            )

            # Push leader to code_queue — leaders need code + build + eval too
            await self.code_queue.put(leader_result)

            # Generate variants for this leader
            num_steps = count_steps(description)
            if num_steps == 0:
                logger.warning(
                    f"Leader {leader_id[:8]}... has no step markers, skipping variants"
                )
                return leader_result

            # Step assignment logic (same as _generate_all_variants / _generate_team_data_sync)
            if num_steps < m_variants_per_leader:
                step_assignments = [
                    (i + 1) if i < num_steps else num_steps
                    for i in range(m_variants_per_leader)
                ]
            else:
                start_step = num_steps - m_variants_per_leader + 1
                step_assignments = [start_step + i for i in range(m_variants_per_leader)]

            variant_tasks = [
                self._generate_one_variant(
                    leader_id=leader_id,
                    leader_result=leader_result,
                    target_step=target_step,
                    variant_prompt_template=variant_prompt_template,
                    variant_idx=j,
                    total_variants=m_variants_per_leader,
                )
                for j, target_step in enumerate(step_assignments)
            ]

            results = await asyncio.gather(*variant_tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(
                        f"[parallel] Variant {i} for leader {leader_id[:8]}... raised: {result}"
                    )

            return leader_result

        except Exception as e:
            logger.error(f"[parallel] Failed to generate leader {leader_idx}: {e}")
            return None

    # ------------------------------------------------------------------ #
    # Stage 2: Code generation
    # ------------------------------------------------------------------ #

    async def _code_gen_worker(self, worker_id: int, code_prompt_template: str):
        """Consume algorithms from code_queue, generate code, push to build_queue."""
        loop = asyncio.get_event_loop()

        while True:
            item = await self.code_queue.get()
            if item is _SENTINEL:
                # Propagate sentinel so other workers also stop
                await self.code_queue.put(_SENTINEL)
                break

            algorithm: AlgorithmResult = item
            _t0 = time.time()

            try:
                code_prompt = generate_code_prompt(
                    code_prompt_template,
                    algorithm.description,
                    algorithm.function_name,
                )

                async with self.api_semaphore:
                    raw_text = await loop.run_in_executor(
                        self.api_pool,
                        lambda p=code_prompt: get_response_from_gemini(
                            p,
                            system_message=self.system_message,
                            model=self.model,
                        ),
                    )
                self._record_item_time("code_gen", _t0, time.time())

                code_str = parse_code_response({"text": raw_text})
                code_id = get_id(code_str)

                code_result = CodeResult(
                    id=code_id,
                    algorithm_id=algorithm.id,
                    code=code_str,
                    status=CodeStatus.Generated,
                    par2=None,
                    last_updated=datetime.now(),
                    build_success=NOT_INITIALIZED,
                )

                # DB writes
                await loop.run_in_executor(
                    self.db_pool,
                    lambda: update_code_result(code_result),
                )

                # Update algorithm with code_id
                await loop.run_in_executor(
                    self.db_pool,
                    lambda: append_code_id(algorithm.id, code_id),
                )
                # Also update in-memory for downstream use
                algorithm.code_id_list = [code_id]
                algorithm.status = AlgorithmStatus.CodeGenerated

                await loop.run_in_executor(
                    self.db_pool,
                    lambda: update_algorithm_result(algorithm),
                )

                self.codes_generated += 1
                logger.info(
                    f"[parallel] Code generated {self.codes_generated} "
                    f"(algo {algorithm.id[:8]}...)"
                )

                # Push to build stage
                await self.build_queue.put((algorithm, code_result))

            except Exception as e:
                logger.error(
                    f"[parallel] Code gen failed for algo {algorithm.id[:8]}...: {e}"
                )

        logger.info(f"[parallel] Code gen worker {worker_id} finished")

    # ------------------------------------------------------------------ #
    # Stage 3: Build + SLURM submit
    # ------------------------------------------------------------------ #

    async def _build_submit_worker(self, worker_id: int):
        """Consume (algorithm, code_result) from build_queue, build solver, submit SLURM."""
        loop = asyncio.get_event_loop()

        while True:
            item = await self.build_queue.get()
            if item is _SENTINEL:
                await self.build_queue.put(_SENTINEL)
                break

            algorithm, code_result = item
            _t0 = time.time()

            try:
                # Build solver (blocking: ./configure && make)
                async with self.build_semaphore:
                    solver_path = await loop.run_in_executor(
                        self.build_pool,
                        lambda: self.eval_pipeline.build_solver(code_result),
                    )

                self._record_item_time("build", _t0, time.time())

                if solver_path is None:
                    self.builds_failed += 1
                    logger.warning(
                        f"[parallel] Build failed for code {code_result.id[:8]}..."
                    )
                    continue

                self.builds_succeeded += 1

                # Determine result directory
                result_dir = get_solver_result_dir(
                    algorithm.id,
                    code_result.id,
                    generation_tag=self.generation_tag,
                    role=algorithm.role,
                    mkdir=True,
                )

                # Submit SLURM job(s)
                benchmark_path = getattr(self.eval_pipeline, 'benchmark_path', None) or SAT2025_BENCHMARK_PATH
                cnf_files = self.eval_pipeline.cnf_files
                _t_submit = time.time()

                if self.nersc and self.accumulator is not None:
                    # NERSC: accumulate solvers and submit in large batches
                    submitted_ids = await loop.run_in_executor(
                        self.build_pool,
                        lambda sp=solver_path, rd=result_dir, cid=code_result.id:
                            self.accumulator.add_solver(sp, rd, cid),
                    )
                    if submitted_ids:
                        self._record_job_ids(submitted_ids, algorithm.id, code_result.id)
                    self.slurm_submitted += 1
                    logger.info(
                        f"[parallel] Accumulated solver {self.slurm_submitted} "
                        f"(code {code_result.id[:8]}...) for NERSC batch"
                    )
                else:
                    job_ids = await loop.run_in_executor(
                        self.build_pool,
                        lambda: self.eval_pipeline.slurm_run_evaluate(
                            solver_path=solver_path,
                            benchmark_path=benchmark_path,
                            result_dir=result_dir,
                            cnf_files=cnf_files,
                        ),
                    )

                    if job_ids:
                        self._record_job_ids(job_ids, algorithm.id, code_result.id)
                        self.slurm_submitted += 1
                        logger.info(
                            f"[parallel] SLURM submitted {self.slurm_submitted} "
                            f"(code {code_result.id[:8]}..., jobs: {job_ids})"
                        )

                self._record_item_time("slurm_submit", _t_submit, time.time())

            except Exception as e:
                logger.error(
                    f"[parallel] Build/submit failed for code {code_result.id[:8]}...: {e}"
                )

        logger.info(f"[parallel] Build/submit worker {worker_id} finished")

    def _record_job_ids(self, job_ids: List[int], algorithm_id: str, code_id: str):
        """Atomically append job IDs to the JSONL tracking file."""
        record = json.dumps({
            "job_ids": job_ids,
            "algorithm_id": algorithm_id,
            "code_id": code_id,
        })
        atomic_append(self.job_ids_path, record + "\n")

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    async def _run(
        self,
        leaders: Dict[str, AlgorithmResult],
        variant_prompt_template: str,
        code_prompt_template: str,
        m_variants_per_leader: int,
        existing_members_by_leader: Dict[str, int],
    ):
        """Run the three-stage streaming pipeline."""
        self._pipeline_start = time.time()

        # Start code gen workers (consume from code_queue, produce to build_queue)
        num_code_workers = min(MAX_API_CONCURRENT, 5)
        code_workers = [
            asyncio.create_task(self._code_gen_worker(i, code_prompt_template))
            for i in range(num_code_workers)
        ]

        # Start build+submit workers (consume from build_queue)
        num_build_workers = MAX_BUILD_CONCURRENT
        build_workers = [
            asyncio.create_task(self._build_submit_worker(i))
            for i in range(num_build_workers)
        ]

        # Stage 1: generate all variants (produces to code_queue)
        await self._generate_all_variants(
            leaders, variant_prompt_template, m_variants_per_leader,
            existing_members_by_leader,
        )

        # Wait for code gen workers to finish
        await asyncio.gather(*code_workers)

        # Signal build workers that no more work is coming
        await self.build_queue.put(_SENTINEL)

        # Wait for build+submit workers to finish
        await asyncio.gather(*build_workers)

        # Also write a consolidated job_ids.json for backward compat with polling loop
        self._write_consolidated_job_ids()

        logger.info(
            f"[parallel] Pipeline complete: "
            f"{self.algos_generated} algos, {self.codes_generated} codes, "
            f"{self.builds_succeeded} builds OK, {self.builds_failed} builds failed, "
            f"{self.slurm_submitted} SLURM arrays submitted"
        )
        # Timing is saved by the caller (run_streaming_mutants) after eval wait.

    def _write_consolidated_job_ids(self):
        """Write a consolidated submitted_job_ids.json for backward compat with run_loop_a.sh."""
        all_job_ids = []
        jsonl_path = self.job_ids_path
        if os.path.exists(jsonl_path):
            with open(jsonl_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        all_job_ids.extend(record.get("job_ids", []))
                    except json.JSONDecodeError:
                        continue

        # In NERSC mode, some IDs may only live in the accumulator until final flush.
        if self.nersc and self.accumulator is not None:
            all_job_ids.extend(self.accumulator.get_job_ids())

        # Keep stable order while removing duplicates.
        all_job_ids = list(dict.fromkeys(all_job_ids))

        consolidated_path = os.path.join(self.output_dir, "submitted_job_ids.json")
        with open(consolidated_path, "w") as f:
            json.dump({"job_ids": all_job_ids}, f)
        logger.info(f"Wrote {len(all_job_ids)} job IDs to {consolidated_path}")

    async def _run_init(
        self,
        designer_prompt: str,
        variant_prompt_template: str,
        code_prompt_template: str,
        n_leaders: int,
        m_variants_per_leader: int,
    ):
        """Run init mode: leaders + variants + code + build + submit, all streaming.

        Each leader task generates the leader, pushes it to code_queue, then
        spawns and awaits all its variant tasks. This means variant generation
        for leader N overlaps with leader generation for leader N+1, and code
        gen / build / submit overlap with everything via the queues.
        """
        self._pipeline_start = time.time()

        # Start code gen workers (consume from code_queue, produce to build_queue)
        num_code_workers = min(MAX_API_CONCURRENT, 5)
        code_workers = [
            asyncio.create_task(self._code_gen_worker(i, code_prompt_template))
            for i in range(num_code_workers)
        ]

        # Start build+submit workers (consume from build_queue)
        num_build_workers = MAX_BUILD_CONCURRENT
        build_workers = [
            asyncio.create_task(self._build_submit_worker(i))
            for i in range(num_build_workers)
        ]

        # Launch all leader tasks concurrently. Each leader task:
        #   1. Generates the leader (API call)
        #   2. Pushes leader to code_queue
        #   3. Spawns + awaits all variant tasks for that leader
        # The api_semaphore limits total in-flight API calls across all of these.
        leader_tasks = [
            self._generate_one_leader(
                leader_idx=i,
                n_leaders=n_leaders,
                designer_prompt=designer_prompt,
                variant_prompt_template=variant_prompt_template,
                m_variants_per_leader=m_variants_per_leader,
            )
            for i in range(n_leaders)
        ]

        results = await asyncio.gather(*leader_tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"[parallel] Leader task {i} raised: {result}")

        # All leaders and their variants are done — signal code gen workers
        await self.code_queue.put(_SENTINEL)
        logger.info(
            f"[parallel] Init generation complete: "
            f"{self.leaders_generated} leaders, {self.algos_generated} variants"
        )

        # Wait for code gen workers to drain
        await asyncio.gather(*code_workers)

        # Signal build workers
        await self.build_queue.put(_SENTINEL)

        # Wait for build+submit workers to drain
        await asyncio.gather(*build_workers)

        self._write_consolidated_job_ids()

        logger.info(
            f"[parallel] Init pipeline complete: "
            f"{self.leaders_generated} leaders, {self.algos_generated} variants, "
            f"{self.codes_generated} codes, "
            f"{self.builds_succeeded} builds OK, {self.builds_failed} builds failed, "
            f"{self.slurm_submitted} SLURM arrays submitted"
        )


def run_streaming_mutants(
    leaders: Dict[str, AlgorithmResult],
    variant_prompt_template: str,
    code_prompt_template: str,
    generation_tag: str,
    m_variants_per_leader: int = 3,
    model: str = DEFAULT_MODEL,
    quick_eval: bool = True,
    nersc: bool = False,
):
    """
    Streaming parallel mutant generation + code gen + build + SLURM submit.

    Drop-in replacement for _generate_mutants_sync() that also handles
    build and SLURM submission (replaces the separate evaluation.py --run_all step).

    Args:
        leaders: Dict of leader_id -> AlgorithmResult
        variant_prompt_template: Template for variant generation prompt
        code_prompt_template: Template for code generation prompt
        generation_tag: Output generation tag
        m_variants_per_leader: Number of variants per leader
        model: Gemini model name
        quick_eval: Use quick-eval mode (50 CNFs, 600s timeout)

    Returns:
        List of generated member algorithm IDs.
    """
    # Build map of existing members per leader in this generation tag
    all_tag_ids = get_ids_from_router_table(CHATGPT_DATA_GENERATION_TABLE, generation_tag)
    existing_members_by_leader: Dict[str, int] = {}
    for aid in all_tag_ids:
        ar = get_algorithm_result(aid)
        if ar and ar.role == Role.MEMBER:
            parent = ar.parent_id[0] if isinstance(ar.parent_id, list) else ar.parent_id
            existing_members_by_leader[parent] = existing_members_by_leader.get(parent, 0) + 1

    pipeline = ParallelPipeline(
        generation_tag=generation_tag,
        model=model,
        quick_eval=quick_eval,
        nersc=nersc,
    )

    asyncio.run(pipeline._run(
        leaders=leaders,
        variant_prompt_template=variant_prompt_template,
        code_prompt_template=code_prompt_template,
        m_variants_per_leader=m_variants_per_leader,
        existing_members_by_leader=existing_members_by_leader,
    ))

    if nersc and pipeline.accumulator is not None:
        flushed_ids = pipeline.accumulator.flush_all()
        logger.info(f"[parallel] NERSC accumulator flushed: {len(flushed_ids)} SLURM job IDs")
        pipeline._write_consolidated_job_ids()

    logger.info(
        f"[parallel] Streaming pipeline done: "
        f"{pipeline.algos_generated} variants generated, "
        f"{pipeline.slurm_submitted} SLURM arrays submitted"
    )

    # Save generation timing (eval_wait added later by caller via save_eval_timing)
    _save_timing_log(pipeline._build_timing_dict("mutants"), pipeline.output_dir)

    return []  # member_ids not tracked individually (consistent with batch mode)


def run_streaming_init(
    designer_prompt: str,
    variant_prompt_template: str,
    code_prompt_template: str,
    generation_tag: str,
    n_leaders: int = 5,
    m_variants_per_leader: int = 3,
    model: str = DEFAULT_MODEL,
    quick_eval: bool = True,
    nersc: bool = False
):
    """
    Streaming parallel init: leader gen + variant gen + code gen + build + SLURM submit.

    Drop-in replacement for generate_team_data() that also handles build and
    SLURM submission. Leaders and their variants are generated concurrently,
    and each algorithm flows through code gen -> build -> submit as soon as
    it's ready.

    Args:
        designer_prompt: The designer prompt text for leader generation
        variant_prompt_template: Template for variant generation prompt
        code_prompt_template: Template for code generation prompt
        generation_tag: Output generation tag
        n_leaders: Number of leaders to generate
        m_variants_per_leader: Number of variants per leader
        model: Gemini model name
        quick_eval: Use quick-eval mode (50 CNFs, 600s timeout)
    """
    pipeline = ParallelPipeline(
        generation_tag=generation_tag,
        model=model,
        quick_eval=quick_eval,
        nersc=nersc
    )

    asyncio.run(pipeline._run_init(
        designer_prompt=designer_prompt,
        variant_prompt_template=variant_prompt_template,
        code_prompt_template=code_prompt_template,
        n_leaders=n_leaders,
        m_variants_per_leader=m_variants_per_leader,
    ))

    if nersc and pipeline.accumulator is not None:
        flushed_ids = pipeline.accumulator.flush_all()
        logger.info(f"[parallel] NERSC accumulator flushed: {len(flushed_ids)} SLURM job IDs")
        pipeline._write_consolidated_job_ids()

    logger.info(
        f"[parallel] Init streaming pipeline done: "
        f"{pipeline.leaders_generated} leaders, {pipeline.algos_generated} variants, "
        f"{pipeline.slurm_submitted} SLURM arrays submitted"
    )

    # Save generation timing (eval_wait added later by caller via save_eval_timing)
    _save_timing_log(pipeline._build_timing_dict("init"), pipeline.output_dir)
