#!/usr/bin/env python
"""Evaluate the base (unmodified) kissat solver to get baseline PAR2."""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llmsat.llmsat import (
    BASE_SOLVER_PATH,
    SAT2025_BENCHMARK_PATH,
    setup_logging,
    get_logger,
)
from llmsat.pipelines.evaluation import EvaluationPipeline, PAR2_PENALTY

setup_logging()
logger = get_logger(__name__)

BASELINE_RESULT_DIR = "solvers/baseline/result"
BASELINE_SOLVING_TIMES_PATH = "data/results/baseline/baseline_solving_times.json"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate baseline kissat solver")
    parser.add_argument("--collect", action="store_true",
                        help="Collect results from completed SLURM jobs")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Print SLURM commands without submitting")
    args = parser.parse_args()

    pipeline = EvaluationPipeline()

    if args.collect:
        logger.info(f"Collecting baseline results from {BASELINE_RESULT_DIR}")
        solving_times = {}
        if os.path.isdir(BASELINE_RESULT_DIR):
            for file in sorted(os.listdir(BASELINE_RESULT_DIR)):
                if file.endswith(".solving.log"):
                    instance_name = file.split(".")[0]
                    t = pipeline.parse_solving_time(f"{BASELINE_RESULT_DIR}/{file}")
                    if t is not None:
                        solving_times[instance_name] = t

        total = len(solving_times)
        timeouts = sum(1 for t in solving_times.values() if t >= PAR2_PENALTY)
        par2 = sum(solving_times.values()) / total if total > 0 else None

        print(f"\nBaseline Results ({total}/400 instances)")
        print(f"  PAR2:     {par2:.2f}" if par2 else "  PAR2:     N/A")
        print(f"  Timeouts: {timeouts}/{total}")
        if solving_times:
            solved = {k: v for k, v in solving_times.items() if v < PAR2_PENALTY}
            if solved:
                print(f"  Solved:   {len(solved)}/{total}")
                print(f"  Avg solve time (solved only): {sum(solved.values())/len(solved):.2f}s")

        os.makedirs(os.path.dirname(BASELINE_SOLVING_TIMES_PATH), exist_ok=True)
        with open(BASELINE_SOLVING_TIMES_PATH, "w") as f:
            json.dump(solving_times, f)
        logger.info(f"Wrote solving times to {BASELINE_SOLVING_TIMES_PATH}")

        if par2 is not None:
            print(f"\n{'='*50}")
            print("Add this to your path_config.yaml for PAR2 normalization:")
            print(f"  baseline_par2: {par2:.2f}")
            print(f"{'='*50}")
        return

    # Submit evaluation jobs
    slurm_ids = pipeline.slurm_run_evaluate(
        solver_path=BASE_SOLVER_PATH,
        benchmark_path=SAT2025_BENCHMARK_PATH,
        result_dir=BASELINE_RESULT_DIR,
        dry_run=args.dry_run,
    )

    if slurm_ids:
        print(f"\nSubmitted SLURM job array: {slurm_ids}")
        print(f"Results will be written to: {BASELINE_RESULT_DIR}")
        print(f"\nAfter jobs finish, run:")
        print(f"  PYTHONPATH=src python scripts/evaluate_baseline.py --collect")


if __name__ == "__main__":
    main()
