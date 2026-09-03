#!/usr/bin/env python3
"""Run ShinkaEvolve on a configured LLM-SAT single-function experiment."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from shinka.core import EvolutionConfig, ShinkaEvolveRunner
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(
    os.environ.get("SAT_REPO_ROOT", EXPERIMENT_DIR.parents[1])
).resolve()
RESULTS_DIR = Path(
    os.environ.get(
        "SHINKA_RESULTS_DIR",
        EXPERIMENT_DIR / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
).resolve()

TARGET_FUNCTION = os.environ.get("SHINKA_TARGET", "kissat_decide_phase")
BENCHMARK_FAMILY = os.environ.get(
    "SHINKA_BENCHMARK_FAMILY", "cryptography-ascon"
)
TARGET_CONFIGS = {
    "kissat_decide_phase": {
        "initial": "initial.c",
        "default_generations": 101,
        "prompt": f"""
You are evolving exactly one function in the Kissat SAT solver:

  int kissat_decide_phase(kissat *solver, unsigned idx)

The function selects the positive or negative decision phase for variable idx
and must always return exactly -1 or 1. Improve solver runtime on the supplied
{BENCHMARK_FAMILY} benchmark family. Only edit code between the EVOLVE-BLOCK markers.

The candidate is inserted directly into the existing src/decide.c. Keep the
function name, return type, and parameters unchanged. Do not add includes,
helper functions, new struct fields, external state, or calls to APIs that are
not already available in the surrounding Kissat source. You may use existing
solver fields, Kissat macros, target phases, saved phases, solver statistics,
idx, arithmetic, and local variables. The implementation must remain valid C,
deterministic, and safe for every variable index. Avoid division by zero,
invalid shifts, overflow-dependent behavior, and returning zero.
""",
    },
    "kissat_restarting": {
        "initial": "initial_restart.c",
        "default_generations": 101,
        "prompt": f"""
You are evolving exactly one restart-policy function in the Kissat SAT solver:

  bool kissat_restarting(kissat *solver)

The function decides whether search should restart at the current point and
must return a valid bool. Improve solver runtime on the supplied {BENCHMARK_FAMILY}
benchmark family. Only edit code between the EVOLVE-BLOCK markers.

The candidate is inserted directly into the existing src/restart.c. Keep the
function name, return type, and parameter unchanged. Do not add includes,
helper functions, new struct fields, external state, or calls to APIs that are
not already available in the surrounding Kissat source. Preserve the basic
safety preconditions: do not restart when restarts are disabled, at decision
level zero, or before the configured conflict limit. You may use existing
solver fields, Kissat macros, glue averages, restart limits, reluctant state,
statistics, options, arithmetic, and local variables. The implementation must
remain valid C, deterministic, and safe. Avoid division by zero, invalid
shifts, overflow-dependent behavior, and non-finite arithmetic.
""",
    },
}
if TARGET_FUNCTION not in TARGET_CONFIGS:
    supported = ", ".join(sorted(TARGET_CONFIGS))
    raise ValueError(f"Unsupported SHINKA_TARGET={TARGET_FUNCTION!r}; choose: {supported}")
TARGET_CONFIG = TARGET_CONFIGS[TARGET_FUNCTION]

# Shinka counts the fixed initial baseline as generation 0. A 100-offspring
# budget therefore requires 101 generations/evaluations including that baseline.
NUM_GENERATIONS = int(
    os.environ.get("SHINKA_NUM_GENERATIONS", str(TARGET_CONFIG["default_generations"]))
)
OFFSPRING_BUDGET = int(
    os.environ.get("SHINKA_OFFSPRING_BUDGET", str(NUM_GENERATIONS - 1))
)
if NUM_GENERATIONS != OFFSPRING_BUDGET + 1:
    raise ValueError(
        "Shinka includes generation-zero baseline in SHINKA_NUM_GENERATIONS; "
        "expected SHINKA_NUM_GENERATIONS == SHINKA_OFFSPRING_BUDGET + 1"
    )
MAX_EVALUATIONS = int(os.environ.get("SHINKA_MAX_EVALUATIONS", "4"))
MAX_PROPOSALS = int(os.environ.get("SHINKA_MAX_PROPOSALS", "4"))
MODEL = os.environ.get("SHINKA_MODEL", "gpt-5.6-luna")
REASONING_EFFORT = os.environ.get("SHINKA_REASONING_EFFORT", "medium")


TASK_PROMPT = (TARGET_CONFIG["prompt"] + """
Fitness is based only on measured CPU runtime with PAR2 timeout penalties.
Compilation and runtime diagnostics from failed candidates are supplied as
feedback; use them to produce a compilable successor.
""").strip()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stage_config = {
        "target_function": TARGET_FUNCTION,
        "num_generations_including_baseline": NUM_GENERATIONS,
        "offspring_budget": OFFSPRING_BUDGET,
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "base_solver": os.environ.get("OE_BASE_SOLVER"),
        "target_source": os.environ.get("OE_TARGET_SOURCE"),
        "parent_stage_candidate_hash": os.environ.get("SHINKA_PARENT_STAGE_HASH"),
        "benchmark_family": os.environ.get("SHINKA_BENCHMARK_FAMILY"),
        "benchmark_instances": int(os.environ["SHINKA_BENCHMARK_INSTANCES"])
        if os.environ.get("SHINKA_BENCHMARK_INSTANCES")
        else None,
        "benchmark_dir": os.environ.get("OE_BENCHMARK_DIR"),
        "benchmark_list": os.environ.get("OE_BENCHMARK_LIST"),
        "max_parallel_evaluations": MAX_EVALUATIONS,
        "max_parallel_proposals": MAX_PROPOSALS,
        "solver_timeout_seconds": int(os.environ.get("OE_TIMEOUT", "1200")),
        "par2_penalty": float(os.environ.get("OE_PAR2_PENALTY", "2400")),
        "candidate_job_walltime": os.environ.get("OE_WALL_TIME", "10:00:00"),
        "candidate_job_cpus": int(os.environ.get("OE_SLURM_CPUS", "8")),
        "candidate_job_constraint": os.environ.get("OE_SLURM_CONSTRAINT", ""),
    }
    (RESULTS_DIR / "stage_config.json").write_text(
        json.dumps(stage_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    job_config = LocalJobConfig(
        eval_program_path=str(EXPERIMENT_DIR / "evaluate.py"),
        time="12:00:00",
        eval_verbose=True,
        numeric_threads_per_job=1,
    )
    db_config = DatabaseConfig(
        db_path=str(RESULTS_DIR / "evolution_db.sqlite"),
        num_islands=5,
        archive_size=30,
        elite_selection_ratio=0.1,
        num_archive_inspirations=1,
        num_top_k_inspirations=1,
        migration_interval=10,
        migration_rate=0.1,
        island_elitism=True,
        exploitation_ratio=0.2,
    )
    evo_config = EvolutionConfig(
        task_sys_msg=TASK_PROMPT,
        patch_types=["diff"],
        patch_type_probs=[1.0],
        num_generations=NUM_GENERATIONS,
        max_patch_resamples=2,
        max_patch_attempts=3,
        job_type="local",
        language="c",
        llm_models=[MODEL],
        llm_dynamic_selection="fixed",
        # Shinka classifies gpt-5.6-luna as a reasoning model. An explicit
        # effort avoids the invalid empty-string reasoning payload produced by
        # Shinka's generic default; medium matches the API's normal default.
        llm_kwargs={"max_tokens": 16384, "reasoning_efforts": [REASONING_EFFORT]},
        meta_rec_interval=None,
        embedding_model=None,
        init_program_path=str(EXPERIMENT_DIR / str(TARGET_CONFIG["initial"])),
        results_dir=str(RESULTS_DIR),
        max_novelty_attempts=1,
        use_text_feedback=True,
        evolve_prompts=False,
    )

    print(
        "Shinka LLM-SAT comparison: "
        f"target={TARGET_FUNCTION}, generations={NUM_GENERATIONS} "
        f"(baseline + {OFFSPRING_BUDGET} offspring), "
        f"model={MODEL}, reasoning={REASONING_EFFORT}, "
        f"max_evaluations={MAX_EVALUATIONS}, results={RESULTS_DIR}"
    )
    runner = ShinkaEvolveRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        max_evaluation_jobs=MAX_EVALUATIONS,
        max_proposal_jobs=MAX_PROPOSALS,
        max_db_workers=1,
        verbose=True,
    )
    runner.run()


if __name__ == "__main__":
    main()
