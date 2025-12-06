#!/usr/bin/env python3
"""
Evaluate a single solver on all benchmarks with configurable timeout/penalty.

Usage:
  python scripts/evaluate_single_solver.py --solver-path solvers/algorithm_XXX/code_YYY --timeout 300 --penalty 600
  python scripts/evaluate_single_solver.py --solver-path solvers/algorithm_XXX/code_YYY --submit  # Submit SLURM jobs
  python scripts/evaluate_single_solver.py --solver-path solvers/algorithm_XXX/code_YYY --collect  # Collect results
"""

import os
import sys
import json
import argparse
import glob
import re
from typing import Dict, List, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llmsat.llmsat import (
    SAT2025_BENCHMARK_PATH,
    setup_logging,
    get_logger
)
from llmsat.utils.utils import wrap_command_to_slurm_array

setup_logging()
logger = get_logger(__name__)

# Default settings (10min timeout, 20min penalty)
DEFAULT_TIMEOUT = 600
DEFAULT_PENALTY = 1200 


def parse_solving_time(log_file: str, timeout: int, penalty: int) -> Optional[float]:
    """Parse solving time from a .solving.log file."""
    if not os.path.exists(log_file):
        return None

    try:
        with open(log_file, 'r') as f:
            content = f.read()
    except Exception as e:
        logger.warning(f"Failed to read log file {log_file}: {e}")
        return penalty

    if not content.strip():
        logger.warning(f"Empty log file (likely timeout or crash): {log_file}")
        return penalty

    # Check for timeout marker
    if "TIMEOUT" in content:
        return penalty

    # Look for "process-time" line
    for line in content.split('\n'):
        if 'process-time' in line:
            match = re.search(r'(\d+\.?\d*)\s+seconds', line)
            if match:
                time = float(match.group(1))
                # If time exceeds timeout, apply penalty
                if time >= timeout:
                    return penalty
                return time

    # Check for error indicators
    for line in content.split('\n'):
        if 'error' in line.lower() or 'CANCELLED' in line or 'TIME LIMIT' in line:
            return penalty

    logger.warning(f"No process-time found in log (incomplete run): {log_file}")
    return penalty


def submit_evaluation(solver_path: str, benchmark_path: str, result_dir: str,
                      timeout: int, penalty: int, max_jobs: int = 400,
                      dry_run: bool = False, force: bool = False, cnf_file: str = None) -> List[int]:
    """Submit SLURM job array for solver evaluation."""
    
    # Clear existing results if force
    if force and os.path.isdir(result_dir):
        existing_logs = glob.glob(f"{result_dir}/*.solving.log")
        if existing_logs:
            logger.info(f"Force mode: removing {len(existing_logs)} existing .solving.log files")
            for log_file in existing_logs:
                os.remove(log_file)
    
    os.makedirs(result_dir, exist_ok=True)

    solver_binary = f"{solver_path}/build/kissat"
    if not os.path.exists(solver_binary):
        # Try alternative location
        solver_binary = f"{solver_path}/kissat"
        if not os.path.exists(solver_binary):
            logger.error(f"Solver binary not found at {solver_path}/build/kissat or {solver_path}/kissat")
            return []

    if not os.path.isdir(benchmark_path):
        logger.error(f"Benchmark directory not found: {benchmark_path}")
        return []

    # Collect CNF files to evaluate
    if cnf_file:
        # Read from file
        with open(cnf_file, 'r') as f:
            cnf_files = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(cnf_files)} CNF files from {cnf_file}")
    else:
        # Use all CNF files from benchmark directory
        cnf_files = sorted([f for f in os.listdir(benchmark_path) if f.endswith(".cnf")])
        logger.info(f"Found {len(cnf_files)} CNF files in {benchmark_path}")

    # Filter out already-evaluated instances (unless force mode)
    if not force:
        original_count = len(cnf_files)
        cnf_files = [f for f in cnf_files if not os.path.exists(f"{result_dir}/{f}.solving.log")]
        jobs_skipped = original_count - len(cnf_files)
    else:
        jobs_skipped = 0

    if not cnf_files:
        logger.info(f"All {jobs_skipped} benchmarks already evaluated, nothing to submit")
        return []

    logger.info(f"Found {len(cnf_files)} benchmarks to evaluate ({jobs_skipped} already done)")
    logger.info(f"Solver: {solver_binary}")
    logger.info(f"Results will be saved to: {result_dir}")

    if len(cnf_files) > max_jobs:
        logger.warning(f"Limiting evaluation to {max_jobs} benchmarks")
        cnf_files = cnf_files[:max_jobs]

    # Write CNF file list
    cnf_list_path = f"{result_dir}/cnf_file_list.txt"
    with open(cnf_list_path, "w") as f:
        for cnf_file in cnf_files:
            f.write(f"{cnf_file}\n")
    logger.info(f"Wrote {len(cnf_files)} CNF files to {cnf_list_path}")

    # Create wrapper script with timeout
    script_path = f"{result_dir}/run_solver_array.sh"
    abs_solver = os.path.abspath(solver_binary)
    abs_benchmark = os.path.abspath(benchmark_path)
    abs_result = os.path.abspath(result_dir)
    abs_cnf_list = os.path.abspath(cnf_list_path)

    script_content = f"""#!/bin/bash
# SLURM job array script for solver evaluation with timeout

CNF_LIST="{abs_cnf_list}"
SOLVER="{abs_solver}"
BENCHMARK_PATH="{abs_benchmark}"
RESULT_DIR="{abs_result}"
TIMEOUT={timeout}

# Get the CNF file for this array task (0-indexed)
CNF_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CNF_LIST")

if [ -z "$CNF_FILE" ]; then
    echo "ERROR: No CNF file found for array task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Running solver on $CNF_FILE (array task $SLURM_ARRAY_TASK_ID)"
echo "Timeout: ${{TIMEOUT}}s"
timeout ${{TIMEOUT}}s "$SOLVER" "$BENCHMARK_PATH/$CNF_FILE" > "$RESULT_DIR/$CNF_FILE.solving.log" 2>&1
EXIT_CODE=$?
if [ $EXIT_CODE -eq 124 ]; then
    echo "TIMEOUT after ${{TIMEOUT}}s" >> "$RESULT_DIR/$CNF_FILE.solving.log"
fi
echo "Solver finished with exit code $EXIT_CODE"
exit $EXIT_CODE
"""
    with open(script_path, "w") as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)
    logger.info(f"Created job array script at {script_path}")

    if dry_run:
        print(f"\nDry run complete.")
        print(f"Would submit job array with {len(cnf_files)} tasks")
        print(f"Timeout: {timeout}s, Penalty: {penalty}s")
        print(f"Script: {script_path}")
        return []

    # Calculate SLURM wall time (timeout + 2 minute buffer)
    wall_time_seconds = timeout + 120
    slurm_time = f"{wall_time_seconds // 3600:02d}:{(wall_time_seconds % 3600) // 60:02d}:{wall_time_seconds % 60:02d}"

    # Submit job array
    array_range = f"0-{len(cnf_files) - 1}"
    slurm_cmd = wrap_command_to_slurm_array(
        script_path=script_path,
        array_range=array_range,
        mem="8G",
        time=slurm_time,
        job_name="single_solver_eval",
        output_file=f"{abs_result}/slurm_array_%a.log",
        max_concurrent=100,
    )
    logger.info(f"Settings: timeout={timeout}s, penalty={penalty}s, slurm_time={slurm_time}")
    logger.info(f"Submitting job array with command: {slurm_cmd}")

    try:
        slurm_output = os.popen(slurm_cmd).read().strip()
        if not slurm_output or "error" in slurm_output.lower():
            logger.error(f"Failed to submit job array: {slurm_output}")
            return []

        slurm_id = int(slurm_output.split()[-1])
        logger.info(f"Submitted job array {slurm_id} with {len(cnf_files)} tasks")
        logger.info(f"Monitor with: squeue -u $USER")
        return [slurm_id]

    except (ValueError, IndexError) as e:
        logger.error(f"Failed to parse SLURM job ID: {e}")
        return []


def collect_results(result_dir: str, timeout: int, penalty: int, output_file: str = None) -> Optional[float]:
    """Collect results and compute PAR2 score."""
    if not os.path.isdir(result_dir):
        logger.error(f"Result directory not found: {result_dir}")
        return None

    solving_times: Dict[str, float] = {}
    timeouts_or_errors = []

    log_files = [f for f in os.listdir(result_dir) if f.endswith('.solving.log')]

    if not log_files:
        logger.warning(f"No .solving.log files found in {result_dir}")
        return None

    logger.info(f"Collecting results from {len(log_files)} log files...")
    logger.info(f"Using timeout={timeout}s, penalty={penalty}s")

    for log_file in log_files:
        instance_name = log_file.replace('.solving.log', '')
        log_path = os.path.join(result_dir, log_file)

        solving_time = parse_solving_time(log_path, timeout, penalty)

        if solving_time is not None:
            solving_times[instance_name] = solving_time
            if solving_time >= penalty:
                timeouts_or_errors.append(instance_name)
        else:
            solving_times[instance_name] = float(penalty)
            timeouts_or_errors.append(instance_name)

    if solving_times:
        par2 = sum(solving_times.values()) / len(solving_times)

        print(f"\n{'='*80}")
        print(f"Single Solver Evaluation Results")
        print(f"{'='*80}")
        print(f"Result directory: {result_dir}")
        print(f"Completed instances: {len(solving_times)}")
        print(f"Timeouts/Errors: {len(timeouts_or_errors)} (penalty: {penalty}s)")
        print(f"PAR2 Score: {par2:.2f}")
        print(f"{'='*80}\n")

        if output_file is None:
            output_file = os.path.join(result_dir, "solving_times.json")

        with open(output_file, 'w') as f:
            json.dump(solving_times, f, indent=2)
        logger.info(f"Saved solving times to: {output_file}")

        return par2
    else:
        logger.warning("No solving times collected")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a single solver with configurable timeout/penalty",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit evaluation for a specific solver (5min timeout, 10min penalty)
  python scripts/evaluate_single_solver.py --solver-path solvers/algorithm_4dbc.../code_8190... --submit
  
  # Force re-evaluation (clear existing results)
  python scripts/evaluate_single_solver.py --solver-path solvers/algorithm_4dbc.../code_8190... --submit --force

  # Dry run to see what would be submitted
  python scripts/evaluate_single_solver.py --solver-path solvers/algorithm_4dbc.../code_8190... --submit --dry-run

  # Collect results after jobs complete
  python scripts/evaluate_single_solver.py --solver-path solvers/algorithm_4dbc.../code_8190... --collect

  # Custom timeout/penalty
  python scripts/evaluate_single_solver.py --solver-path solvers/algorithm_XXX/code_YYY --submit --timeout 600 --penalty 1200
        """
    )
    parser.add_argument("--solver-path", type=str, required=True,
                       help="Path to solver directory (e.g., solvers/algorithm_XXX/code_YYY)")
    parser.add_argument("--submit", action="store_true", help="Submit SLURM jobs for evaluation")
    parser.add_argument("--collect", action="store_true", help="Collect results and compute PAR2")
    parser.add_argument("--benchmarks", type=str, default=SAT2025_BENCHMARK_PATH,
                       help=f"Path to benchmark directory (default: {SAT2025_BENCHMARK_PATH})")
    parser.add_argument("--cnf-file", type=str, default=None,
                       help="File with list of CNF files (one per line) to evaluate (optional)")
    parser.add_argument("--result-dir", type=str, default=None,
                       help="Directory to store results (default: <solver-path>/result)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                       help=f"Timeout in seconds per CNF instance (default: {DEFAULT_TIMEOUT})")
    parser.add_argument("--penalty", type=int, default=DEFAULT_PENALTY,
                       help=f"Penalty for timeout/error in seconds (default: {DEFAULT_PENALTY})")
    parser.add_argument("--max-jobs", type=int, default=400,
                       help="Maximum number of jobs to submit (default: 400)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run - show what would be done")
    parser.add_argument("--force", action="store_true", help="Force re-evaluation (clear existing results)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file for solving times")

    args = parser.parse_args()

    # Default result directory
    if args.result_dir is None:
        args.result_dir = os.path.join(args.solver_path, "result_short")

    if args.submit:
        submit_evaluation(
            solver_path=args.solver_path,
            benchmark_path=args.benchmarks,
            result_dir=args.result_dir,
            timeout=args.timeout,
            penalty=args.penalty,
            max_jobs=args.max_jobs,
            dry_run=args.dry_run,
            force=args.force,
            cnf_file=args.cnf_file
        )
    elif args.collect:
        collect_results(
            result_dir=args.result_dir,
            timeout=args.timeout,
            penalty=args.penalty,
            output_file=args.output
        )
    else:
        parser.print_help()
        print("\n⚠️  No action specified. Use --submit or --collect")


if __name__ == "__main__":
    main()