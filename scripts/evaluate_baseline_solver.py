#!/usr/bin/env python3
"""
Evaluate the baseline Kissat solver (solvers/base) on benchmarks.
This gives you a PAR2 score to compare against LLM-generated variants.

Uses the same settings as LLM evaluation:
- 30 minute timeout per instance
- 5000s penalty for timeouts/errors
- Job array submission to avoid hitting SLURM QOS limits
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llmsat.llmsat import (
    BASE_SOLVER_PATH,
    SAT2025_BENCHMARK_PATH,
    setup_logging,
    get_logger
)
from llmsat.utils.utils import wrap_command_to_slurm, wrap_command_to_slurm_array

setup_logging()
logger = get_logger(__name__)

# Default settings (can be overridden via command line)
# For short evaluation (matching LLM eval): --timeout 600 --penalty 1200
# For standard SAT competition: --timeout 5000 --penalty 10000
DEFAULT_TIMEOUT_SECONDS = 600  # 10 minutes (matches LLM sequential eval)
DEFAULT_PENALTY_SECONDS = 1200  # 20 minutes penalty (matches LLM sequential eval)

# Global variables that will be set from command line args
TIMEOUT_SECONDS = DEFAULT_TIMEOUT_SECONDS
PENALTY_SECONDS = DEFAULT_PENALTY_SECONDS


def parse_solving_time(log_file: str) -> Optional[float]:
    """Parse solving time from a .solving.log file."""
    if not os.path.exists(log_file):
        return None

    with open(log_file, 'r') as f:
        content = f.read()

    # Look for "process-time" line
    # Format can be:
    #   "c process-time:                         2m 58s             177.71 seconds"
    #   "c process-time:                                              0.00 seconds"
    for line in content.split('\n'):
        if 'process-time' in line:
            try:
                parts = line.split()
                # Look for "seconds" and get the number before it
                for i, part in enumerate(parts):
                    if part == 'seconds' and i > 0:
                        time_str = parts[i - 1]
                        return float(time_str)
            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse process-time from line: {line} ({e})")

    # If no process-time found, this might be an incomplete/timeout
    logger.warning(f"No process-time found in log (incomplete run): {log_file}")
    return None


def submit_evaluation_jobs(solver_binary: str, benchmark_path: str, result_dir: str, dry_run: bool = False, max_jobs: int = 200, timeout: int = None, penalty: int = None, force: bool = False, cnf_file: str = None) -> List[int]:
    """
    Submit solver evaluation using a SLURM job array (counts as 1 job toward QOS limit).
    
    Creates a CNF file list and a wrapper script, then submits a single job array
    that runs the solver on all benchmarks. Matches the approach used for LLM evaluation.
    
    Args:
        solver_binary: Path to the solver binary
        benchmark_path: Path to benchmark CNF files
        result_dir: Directory to store results
        dry_run: If True, don't actually submit jobs
        max_jobs: Maximum number of jobs to submit
        timeout: Timeout in seconds per instance (default: TIMEOUT_SECONDS)
        penalty: Penalty in seconds for timeout/error (default: PENALTY_SECONDS)
        force: If True, re-evaluate all benchmarks (clear existing results)
        cnf_file: Path to file containing CNF filenames to evaluate (one per line)
    """
    if timeout is None:
        timeout = TIMEOUT_SECONDS
    if penalty is None:
        penalty = PENALTY_SECONDS
    
    # Clear existing results if force is True
    if force and os.path.isdir(result_dir):
        import glob
        existing_logs = glob.glob(f"{result_dir}/*.solving.log")
        if existing_logs:
            logger.info(f"Force mode: removing {len(existing_logs)} existing .solving.log files")
            for log_file in existing_logs:
                os.remove(log_file)
        
    os.makedirs(result_dir, exist_ok=True)

    if not os.path.exists(solver_binary):
        logger.error(f"Solver binary not found: {solver_binary}")
        logger.error(f"Did you run ./setup_kissat_solver.sh?")
        return []

    if not os.path.isdir(benchmark_path):
        logger.error(f"Benchmark directory not found: {benchmark_path}")
        return []

    # Collect CNF files to evaluate
    cnf_files = []
    jobs_skipped = 0
    
    if cnf_file:
        # Load CNF filenames from file
        if not os.path.exists(cnf_file):
            logger.error(f"CNF file list not found: {cnf_file}")
            return []
        with open(cnf_file, 'r') as f:
            all_cnfs = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(all_cnfs)} CNF filenames from {cnf_file}")
        
        # Filter to those that exist and haven't been evaluated
        for cnf in all_cnfs:
            cnf_path = os.path.join(benchmark_path, cnf)
            if not os.path.exists(cnf_path):
                logger.warning(f"CNF file not found: {cnf_path}")
                continue
            if os.path.exists(f"{result_dir}/{cnf}.solving.log"):
                jobs_skipped += 1
                continue
            cnf_files.append(cnf)
    else:
        # Scan benchmark directory
        for benchmark_file in sorted(os.listdir(benchmark_path)):
            if benchmark_file.endswith(".cnf"):
                if os.path.exists(f"{result_dir}/{benchmark_file}.solving.log"):
                    jobs_skipped += 1
                    continue
                cnf_files.append(benchmark_file)

    if not cnf_files:
        logger.info(f"All {jobs_skipped} benchmarks already evaluated, nothing to submit")
        return []

    logger.info(f"Found {len(cnf_files)} benchmarks to evaluate ({jobs_skipped} already done)")
    logger.info(f"Solver: {solver_binary}")
    logger.info(f"Results will be saved to: {result_dir}")

    # Limit to max_jobs if needed
    if len(cnf_files) > max_jobs:
        logger.warning(f"Limiting evaluation to {max_jobs} benchmarks (out of {len(cnf_files)} remaining)")
        cnf_files = cnf_files[:max_jobs]

    # Write CNF file list for the job array
    cnf_list_path = f"{result_dir}/cnf_file_list.txt"
    with open(cnf_list_path, "w") as f:
        for cnf_file in cnf_files:
            f.write(f"{cnf_file}\n")
    logger.info(f"Wrote {len(cnf_files)} CNF files to {cnf_list_path}")

    # Create wrapper script for job array
    script_path = f"{result_dir}/run_baseline_array.sh"
    # Use absolute paths
    abs_solver = os.path.abspath(solver_binary)
    abs_benchmark = os.path.abspath(benchmark_path)
    abs_result = os.path.abspath(result_dir)
    abs_cnf_list = os.path.abspath(cnf_list_path)
    
    script_content = f"""#!/bin/bash
# SLURM job array script for baseline solver evaluation
# Each array task reads its assigned CNF file from the list

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

echo "Running baseline solver on $CNF_FILE (array task $SLURM_ARRAY_TASK_ID)"
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
    
    # Submit job array (0-indexed, so 0 to N-1)
    array_range = f"0-{len(cnf_files) - 1}"
    slurm_cmd = wrap_command_to_slurm_array(
        script_path=script_path,
        array_range=array_range,
        mem="8G",
        time=slurm_time,
        job_name="baseline_array",
        output_file=f"{abs_result}/slurm_array_%a.log",
        max_concurrent=100,  # Limit concurrent tasks
    )
    logger.info(f"Settings: timeout={timeout}s, penalty={penalty}s, slurm_time={slurm_time}")
    logger.info(f"Submitting job array with command: {slurm_cmd}")

    try:
        slurm_output = os.popen(slurm_cmd).read().strip()

        if not slurm_output or "error" in slurm_output.lower():
            logger.error(f"Failed to submit job array: {slurm_output}")
            return []

        # Parse the job array ID (e.g., "Submitted batch job 12345")
        slurm_id = int(slurm_output.split()[-1])
        logger.info(f"Submitted job array {slurm_id} with {len(cnf_files)} tasks ({jobs_skipped} skipped)")
        logger.info(f"Monitor with: squeue -u $USER")
        logger.info(f"After completion, run: python scripts/evaluate_baseline_solver.py --collect")

        return [slurm_id]

    except (ValueError, IndexError) as e:
        logger.error(f"Failed to parse SLURM job ID from output: '{slurm_output}' - {e}")
        return []


def collect_results(result_dir: str, output_file: str = None, penalty: int = None):
    """
    Collect results from all .solving.log files and compute PAR2 score.
    
    Args:
        result_dir: Directory containing .solving.log files
        output_file: Output JSON file path (default: <result_dir>/baseline_solving_times.json)
        penalty: Penalty in seconds for timeout/error (default: PENALTY_SECONDS)
    """
    if penalty is None:
        penalty = PENALTY_SECONDS
        
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
    logger.info(f"Using penalty: {penalty}s for timeouts/errors")

    for log_file in log_files:
        instance_name = log_file.replace('.solving.log', '')
        log_path = os.path.join(result_dir, log_file)

        solving_time = parse_solving_time(log_path)

        if solving_time is not None:
            solving_times[instance_name] = solving_time
            if solving_time >= penalty:
                timeouts_or_errors.append(instance_name)
        else:
            # Incomplete/timeout - assign penalty (matches LLM evaluation)
            solving_times[instance_name] = float(penalty)
            timeouts_or_errors.append(instance_name)

    # Compute PAR2 score
    if solving_times:
        par2 = sum(solving_times.values()) / len(solving_times)

        print(f"\n{'='*80}")
        print(f"Baseline Solver Results")
        print(f"{'='*80}")
        print(f"Completed instances: {len(solving_times)}")
        print(f"Timeouts/Errors: {len(timeouts_or_errors)} (penalty: {penalty}s)")
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
  # Submit evaluation jobs on 100-CNF subset (default: 600s timeout, 1200s penalty)
  python scripts/evaluate_baseline_solver.py --submit --cnf-file data/cnf_subset_100.txt
  
  # Submit evaluation on all benchmarks
  python scripts/evaluate_baseline_solver.py --submit
  
  # Submit with standard SAT competition settings (5000s/10000s)
  python scripts/evaluate_baseline_solver.py --submit --timeout 5000 --penalty 10000

  # Dry run to see what would be submitted
  python scripts/evaluate_baseline_solver.py --submit --dry-run --cnf-file data/cnf_subset_100.txt

  # Collect results and compute PAR2 score (uses default penalty)
  python scripts/evaluate_baseline_solver.py --collect
  
  # Collect results with custom penalty
  python scripts/evaluate_baseline_solver.py --collect --penalty 1200

  # Use custom paths
  python scripts/evaluate_baseline_solver.py --submit --solver /path/to/kissat --benchmarks /path/to/cnf/
        """
    )
    parser.add_argument("--submit", action="store_true", help="Submit SLURM jobs for evaluation")
    parser.add_argument("--collect", action="store_true", help="Collect results and compute PAR2")
    parser.add_argument("--solver", type=str, default=None,
                       help=f"Path to solver binary (default: {BASE_SOLVER_PATH}/build/kissat)")
    parser.add_argument("--benchmarks", type=str, default=SAT2025_BENCHMARK_PATH,
                       help=f"Path to benchmark directory (default: {SAT2025_BENCHMARK_PATH})")
    parser.add_argument("--result-dir", type=str, default="data/results/baseline",
                       help="Directory to store results (default: data/results/baseline)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file for solving times (default: <result-dir>/baseline_solving_times.json)")
    parser.add_argument("--max-jobs", type=int, default=400,
                       help="Maximum number of jobs to submit in one run (default: 400)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS,
                       help=f"Timeout in seconds per CNF instance (default: {DEFAULT_TIMEOUT_SECONDS})")
    parser.add_argument("--penalty", type=int, default=DEFAULT_PENALTY_SECONDS,
                       help=f"Penalty for timeout/error in seconds (default: {DEFAULT_PENALTY_SECONDS})")
    parser.add_argument("--dry-run", action="store_true", help="Dry run - show what would be done")
    parser.add_argument("--force", action="store_true", help="Force re-evaluation of all benchmarks (clears existing results)")
    parser.add_argument("--cnf-file", type=str, default=None,
                       help="Path to file containing CNF filenames to evaluate (one per line). If not provided, scans benchmark directory.")

    args = parser.parse_args()
    
    # Update global settings from command line args
    global TIMEOUT_SECONDS, PENALTY_SECONDS
    TIMEOUT_SECONDS = args.timeout
    PENALTY_SECONDS = args.penalty

    # Set solver binary path - use build/kissat which outputs statistics including process-time
    if args.solver is None:
        solver_binary = f"{BASE_SOLVER_PATH}/build/kissat"
    else:
        solver_binary = args.solver

    if args.submit:
        submit_evaluation_jobs(
            solver_binary=solver_binary,
            benchmark_path=args.benchmarks,
            result_dir=args.result_dir,
            dry_run=args.dry_run,
            max_jobs=args.max_jobs,
            timeout=args.timeout,
            penalty=args.penalty,
            force=args.force,
            cnf_file=args.cnf_file
        )
    elif args.collect:
        collect_results(
            result_dir=args.result_dir,
            output_file=args.output,
            penalty=args.penalty
        )
    else:
        parser.print_help()
        print("\n⚠️  No action specified. Use --submit or --collect")


if __name__ == "__main__":
    main()

