#!/usr/bin/env python3
"""
Evaluate the baseline Kissat solver (solvers/base) on benchmarks.
This gives you a PAR2 score to compare against LLM-generated variants.
"""

import os
import sys
import json
import argparse
from typing import Dict, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llmsat.llmsat import (
    BASE_SOLVER_PATH,
    SAT2025_BENCHMARK_PATH,
    setup_logging,
    get_logger
)
from llmsat.utils.utils import wrap_command_to_slurm

setup_logging()
logger = get_logger(__name__)


def parse_solving_time(log_file: str) -> Optional[float]:
    """Parse solving time from a .solving.log file."""
    if not os.path.exists(log_file):
        return None

    with open(log_file, 'r') as f:
        content = f.read()

    # Look for "process-time" line (same as evaluation.py uses)
    for line in content.split('\n'):
        if 'process-time' in line:
            try:
                # Format is usually: "c process-time: X.XX seconds"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'process-time:' and i + 1 < len(parts):
                        time_str = parts[i + 1]
                        return float(time_str)
            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse process-time from line: {line} ({e})")

    # If no process-time found, this might be an incomplete/timeout
    logger.warning(f"No process-time found in log (incomplete run): {log_file}")
    return None


def submit_evaluation_jobs(solver_binary: str, benchmark_path: str, result_dir: str, dry_run: bool = False, max_jobs: int = 200):
    """Submit SLURM jobs to evaluate the solver on all benchmarks."""

    os.makedirs(result_dir, exist_ok=True)

    if not os.path.exists(solver_binary):
        logger.error(f"Solver binary not found: {solver_binary}")
        logger.error(f"Did you run ./setup_kissat_solver.sh?")
        return []

    if not os.path.isdir(benchmark_path):
        logger.error(f"Benchmark directory not found: {benchmark_path}")
        return []

    benchmark_files = [f for f in os.listdir(benchmark_path) if f.endswith('.cnf')]

    if not benchmark_files:
        logger.error(f"No .cnf files found in {benchmark_path}")
        return []

    logger.info(f"Found {len(benchmark_files)} benchmark instances")
    logger.info(f"Solver: {solver_binary}")
    logger.info(f"Results will be saved to: {result_dir}")
    logger.info(f"Max jobs per run: {max_jobs}")

    slurm_ids = []
    skipped = 0
    jobs_submitted = 0
    jobs_failed = 0

    for benchmark_file in benchmark_files:
        log_file = f"{result_dir}/{benchmark_file}.solving.log"

        # Skip if already evaluated
        if os.path.exists(log_file):
            skipped += 1
            continue

        # Check if we've hit the job limit
        if jobs_submitted >= max_jobs:
            logger.warning(f"Reached maximum job limit ({max_jobs}), stopping submission")
            logger.warning(f"Re-run this command to submit remaining {len(benchmark_files) - skipped - jobs_submitted} jobs")
            break

        command = f"{solver_binary} {benchmark_path}/{benchmark_file} > {log_file}"
        slurm_log = f"{result_dir}/{benchmark_file}.slurm.log"

        slurm_cmd = wrap_command_to_slurm(
            command,
            output_file=slurm_log,
            job_name=f"baseline_{benchmark_file[:30]}",  # Truncate long names
            time="00:30:00"  # 30 minutes timeout
        )

        if dry_run:
            print(f"Would submit: {slurm_cmd}")
            jobs_submitted += 1
        else:
            logger.info(f"Submitting job for {benchmark_file}")
            try:
                slurm_output = os.popen(slurm_cmd).read().strip()

                # Check if sbatch failed
                if not slurm_output or "error" in slurm_output.lower():
                    logger.error(f"Failed to submit job for {benchmark_file}: {slurm_output}")
                    jobs_failed += 1
                    # If we hit a job limit error, stop trying
                    if "QOSMaxSubmitJobPerUserLimit" in slurm_output or "job submit limit" in slurm_output.lower():
                        logger.error("Hit SLURM job submission limit. Stopping further submissions.")
                        logger.info(f"Successfully submitted {jobs_submitted} jobs before hitting limit")
                        break
                    continue

                # Parse job ID
                slurm_id = int(slurm_output.split()[-1])
                slurm_ids.append(slurm_id)
                jobs_submitted += 1
                logger.info(f"  Job ID: {slurm_id}")

            except (ValueError, IndexError) as e:
                logger.error(f"Failed to parse SLURM job ID from output: '{slurm_output}' - {e}")
                jobs_failed += 1
                continue

    if skipped > 0:
        logger.info(f"Skipped {skipped} already-evaluated instances")

    if dry_run:
        print(f"\nDry run complete. Would submit {jobs_submitted} jobs.")
    else:
        logger.info(f"Job submission summary: {jobs_submitted} submitted, {skipped} skipped, {jobs_failed} failed")
        logger.info(f"Monitor with: squeue -u $USER")
        if jobs_submitted < len(benchmark_files) - skipped:
            remaining = len(benchmark_files) - skipped - jobs_submitted
            logger.warning(f"Still have {remaining} benchmarks to evaluate. Re-run to submit more.")

    return slurm_ids


def collect_results(result_dir: str, output_file: str = None):
    """Collect results from all .solving.log files and compute PAR2 score."""

    if not os.path.isdir(result_dir):
        logger.error(f"Result directory not found: {result_dir}")
        return

    solving_times: Dict[str, float] = {}
    timeouts_or_errors = []

    log_files = [f for f in os.listdir(result_dir) if f.endswith('.solving.log')]

    if not log_files:
        logger.warning(f"No .solving.log files found in {result_dir}")
        return

    logger.info(f"Collecting results from {len(log_files)} log files...")

    for log_file in log_files:
        instance_name = log_file.replace('.solving.log', '')
        log_path = os.path.join(result_dir, log_file)

        solving_time = parse_solving_time(log_path)

        if solving_time is not None:
            solving_times[instance_name] = solving_time
            if solving_time >= 5000:
                timeouts_or_errors.append(instance_name)
        else:
            # Incomplete/timeout - assign penalty
            solving_times[instance_name] = 5000.0
            timeouts_or_errors.append(instance_name)

    # Compute PAR2 score
    if solving_times:
        par2 = sum(solving_times.values()) / len(solving_times)

        print(f"\n{'='*80}")
        print(f"Baseline Solver Results")
        print(f"{'='*80}")
        print(f"Completed instances: {len(solving_times)}")
        print(f"Timeouts/Errors: {len(timeouts_or_errors)}")
        print(f"PAR2 Score: {par2:.2f}")
        print(f"{'='*80}\n")

        if timeouts_or_errors:
            logger.warning(f"Found {len(timeouts_or_errors)} instances that timed out or had errors")
            logger.info(f"First few problematic instances: {', '.join(timeouts_or_errors[:5])}")

        # Save results to JSON
        if output_file is None:
            output_file = os.path.join(result_dir, "baseline_solving_times.json")

        with open(output_file, 'w') as f:
            json.dump(solving_times, f, indent=2)

        logger.info(f"Saved solving times to: {output_file}")

        return par2
    else:
        logger.warning("No solving times collected")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline Kissat solver on benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit evaluation jobs for baseline solver
  python scripts/evaluate_baseline_solver.py --submit

  # Dry run to see what would be submitted
  python scripts/evaluate_baseline_solver.py --submit --dry-run

  # Collect results and compute PAR2 score
  python scripts/evaluate_baseline_solver.py --collect

  # Use custom paths
  python scripts/evaluate_baseline_solver.py --submit --solver /path/to/kissat --benchmarks /path/to/cnf/
        """
    )
    parser.add_argument("--submit", action="store_true", help="Submit SLURM jobs for evaluation")
    parser.add_argument("--collect", action="store_true", help="Collect results and compute PAR2")
    parser.add_argument("--solver", type=str, default=None,
                       help=f"Path to solver binary (default: {BASE_SOLVER_PATH}/kissat)")
    parser.add_argument("--benchmarks", type=str, default=SAT2025_BENCHMARK_PATH,
                       help=f"Path to benchmark directory (default: {SAT2025_BENCHMARK_PATH})")
    parser.add_argument("--result-dir", type=str, default="data/results/baseline",
                       help="Directory to store results (default: data/results/baseline)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file for solving times (default: <result-dir>/baseline_solving_times.json)")
    parser.add_argument("--max-jobs", type=int, default=200,
                       help="Maximum number of jobs to submit in one run (default: 200)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run - show what would be done")

    args = parser.parse_args()

    # Set solver binary path
    if args.solver is None:
        solver_binary = f"{BASE_SOLVER_PATH}/kissat"
    else:
        solver_binary = args.solver

    if args.submit:
        submit_evaluation_jobs(
            solver_binary=solver_binary,
            benchmark_path=args.benchmarks,
            result_dir=args.result_dir,
            dry_run=args.dry_run,
            max_jobs=args.max_jobs
        )
    elif args.collect:
        collect_results(
            result_dir=args.result_dir,
            output_file=args.output
        )
    else:
        parser.print_help()
        print("\n⚠️  No action specified. Use --submit or --collect")


if __name__ == "__main__":
    main()
