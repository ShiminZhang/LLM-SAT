#!/usr/bin/env python3
"""
Evaluate any single solver (baseline or generated) on SAT competition benchmarks.

Delegates to EvaluationPipeline.slurm_run_evaluate() so that SLURM settings,
timeout handling, and CPU-time measurement stay consistent across all evals.

Usage:
  # Full eval (400 CNFs, 5000s timeout, PAR-2 penalty 10000)
  python scripts/evaluate_solver.py solvers/base
  python scripts/evaluate_solver.py solvers/phase_decide_iter0/leaders/algorithm_.../code_.../

  # Quick eval (50 CNFs, 600s timeout, PAR-2 penalty 1200)
  python scripts/evaluate_solver.py solvers/base --quick-eval

  # Collect results after SLURM jobs finish
  python scripts/evaluate_solver.py solvers/base --collect
  python scripts/evaluate_solver.py solvers/base --collect --quick-eval
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llmsat.llmsat import SAT2025_BENCHMARK_PATH, setup_logging, get_logger
from llmsat.utils.results import instance_key
from llmsat.pipelines.evaluation import (
    EvaluationPipeline,
    SLURM_TIMEOUT_SECONDS,
    QUICK_EVAL_BENCHMARK_LIST,
    QUICK_EVAL_PAR2_PENALTY,
    QUICK_EVAL_TIMEOUT_SECONDS,
    QUICK_EVAL_WALL_TIME,
)

setup_logging()
logger = get_logger(__name__)


def find_solver_binary(solver_path: str) -> str:
    """Locate the kissat binary inside a solver directory."""
    for candidate in ("kissat", "build/kissat"):
        path = os.path.join(solver_path, candidate)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    sys.exit(
        f"ERROR: No executable kissat binary found in {solver_path}\n"
        f"  Looked for: {solver_path}/kissat, {solver_path}/build/kissat"
    )


def result_dir_for(solver_path: str, quick: bool) -> str:
    """Canonical result directory: <solver>/result_quick or <solver>/result."""
    suffix = "result_quick" if quick else "result"
    return os.path.join(solver_path, suffix)


def collect(pipeline: EvaluationPipeline, result_dir: str) -> None:
    """Parse .solving.log files and report PAR-2."""
    if not os.path.isdir(result_dir):
        sys.exit(f"ERROR: Result directory not found: {result_dir}")

    solving_times = {}
    for fname in sorted(os.listdir(result_dir)):
        if not fname.endswith(".solving.log"):
            continue
        instance = instance_key(fname)
        t = pipeline.parse_solving_time(os.path.join(result_dir, fname))
        if t is not None:
            solving_times[instance] = t

    if not solving_times:
        sys.exit(f"ERROR: No .solving.log files found in {result_dir}")

    total = len(solving_times)
    penalty = pipeline.par2_penalty
    solved = sum(1 for t in solving_times.values() if t < penalty)
    par2 = sum(solving_times.values()) / total

    print(f"\nResults from {result_dir}")
    print(f"  Instances: {total}")
    print(f"  Solved:    {solved}/{total}")
    print(f"  Timeouts:  {total - solved}/{total}")
    print(f"  PAR-2:     {par2:.2f}")

    if solved:
        solved_times = [t for t in solving_times.values() if t < penalty]
        print(f"  Avg time (solved): {sum(solved_times) / len(solved_times):.2f}s")

    out_path = os.path.join(result_dir, "solving_times.json")
    with open(out_path, "w") as f:
        json.dump(solving_times, f, indent=2)
    print(f"\nWrote {total} entries to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a single solver on SAT competition benchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "solver_path",
        help="Path to solver directory (e.g. solvers/base, solvers/.../code_.../)",
    )
    parser.add_argument(
        "--quick-eval",
        action="store_true",
        help="Quick evaluation: 50 CNFs, 600s timeout, 1200 PAR-2 penalty",
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Collect results from completed SLURM jobs instead of submitting",
    )
    parser.add_argument(
        "--result-dir",
        default=None,
        help="Override result directory (default: <solver>/result or result_quick)",
    )
    args = parser.parse_args()

    solver_path = args.solver_path.rstrip("/")
    if not os.path.isdir(solver_path):
        sys.exit(f"ERROR: Solver directory not found: {solver_path}")

    rdir = args.result_dir or result_dir_for(solver_path, args.quick_eval)

    pipeline = EvaluationPipeline()
    cnf_files = None

    if args.quick_eval:
        if not os.path.exists(QUICK_EVAL_BENCHMARK_LIST):
            sys.exit(
                f"ERROR: Quick-eval benchmark list not found: {QUICK_EVAL_BENCHMARK_LIST}\n"
                f"  Generate it first: python scripts/generate_benchmark_subset.py"
            )
        with open(QUICK_EVAL_BENCHMARK_LIST) as f:
            cnf_files = [line.strip() for line in f if line.strip()]
        pipeline.timeout = QUICK_EVAL_TIMEOUT_SECONDS
        pipeline.wall_time = QUICK_EVAL_WALL_TIME
        pipeline.par2_penalty = QUICK_EVAL_PAR2_PENALTY
        logger.info(f"Quick-eval mode: {len(cnf_files)} CNFs, {QUICK_EVAL_TIMEOUT_SECONDS}s timeout")
    else:
        logger.info(f"Full-eval mode: {SLURM_TIMEOUT_SECONDS}s timeout")

    if args.collect:
        collect(pipeline, rdir)
        return

    find_solver_binary(solver_path)

    slurm_ids = pipeline.slurm_run_evaluate(
        solver_path=solver_path,
        benchmark_path=SAT2025_BENCHMARK_PATH,
        result_dir=rdir,
        cnf_files=cnf_files,
    )

    if slurm_ids:
        quick_flag = " --quick-eval" if args.quick_eval else ""
        print(f"\nAfter jobs finish, run:")
        print(f"  python scripts/evaluate_solver.py {solver_path}{quick_flag} --collect")


if __name__ == "__main__":
    main()
