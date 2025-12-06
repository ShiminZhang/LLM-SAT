#!/usr/bin/env python3
"""
Submit ALL solver evaluations as parallel SLURM job arrays.
This maximizes cluster utilization by submitting multiple arrays at once.

Unlike evaluate_sequential.py which waits between algorithms, this script
submits all job arrays immediately and then waits for all to complete.
"""

import argparse
import os
import sys
import subprocess
import time
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llmsat.llmsat import (
    BASE_SOLVER_PATH,
    SAT2025_BENCHMARK_PATH,
    setup_logging,
    get_logger,
    CodeStatus,
)
from llmsat.utils.aws import (
    get_ids_from_router_table,
    get_algorithm_result,
    get_code_result,
    update_code_result,
)
from llmsat.llmsat import CHATGPT_DATA_GENERATION_TABLE

setup_logging()
logger = get_logger(__name__)

# Configuration
MAX_ARRAY_SIZE = 1000  # SLURM limit is 1001, use 1000 for safety
MAX_CONCURRENT_PER_ARRAY = 100  # Tasks running concurrently per array
DEFAULT_TIMEOUT = 600 # seconds
DEFAULT_PENALTY = 1200 # seconds


@dataclass
class SolverTask:
    algorithm_id: str
    code_id: str
    solver_path: str
    result_dir: str


def get_all_solver_tasks(generation_tag: str, skip_evaluated: bool = True, force: bool = False) -> List[SolverTask]:
    """Get all solver tasks for a generation tag."""
    algorithm_ids = get_ids_from_router_table(CHATGPT_DATA_GENERATION_TABLE, generation_tag)

    tasks = []
    skipped_no_binary = 0
    skipped_evaluated = 0
    skipped_build_failed = 0

    for algorithm_id in algorithm_ids:
        algorithm = get_algorithm_result(algorithm_id)
        if not algorithm:
            continue

        for code_id in (algorithm.code_id_list or []):
            code_result = get_code_result(code_id)
            if not code_result:
                continue

            # Skip already evaluated (unless force=True)
            if not force and skip_evaluated and code_result.status == CodeStatus.Evaluated and code_result.par2 is not None:
                skipped_evaluated += 1
                continue

            # Skip build failures
            if code_result.status == CodeStatus.BuildFailed:
                skipped_build_failed += 1
                continue

            # Check if binary exists
            solver_path = f"solvers/algorithm_{algorithm_id}/code_{code_id}/build/kissat"
            if not os.path.exists(solver_path):
                skipped_no_binary += 1
                continue

            result_dir = f"solvers/algorithm_{algorithm_id}/result/code_{code_id}"

            tasks.append(SolverTask(
                algorithm_id=algorithm_id,
                code_id=code_id,
                solver_path=os.path.abspath(solver_path),
                result_dir=os.path.abspath(result_dir),
            ))

    logger.info(f"Found {len(tasks)} solver tasks")
    logger.info(f"Skipped: {skipped_evaluated} evaluated, {skipped_build_failed} build failed, {skipped_no_binary} no binary")

    return tasks


def get_cnf_files(benchmark_path: str, cnf_file: str = None) -> List[str]:
    """Get list of CNF files."""
    if cnf_file:
        with open(cnf_file, 'r') as f:
            cnfs = [line.strip() for line in f if line.strip()]
        return [c for c in cnfs if os.path.exists(os.path.join(benchmark_path, c))]
    else:
        return sorted([f for f in os.listdir(benchmark_path) if f.endswith('.cnf')])


def create_task_chunks(
    solver_tasks: List[SolverTask],
    cnf_files: List[str],
    chunk_size: int = MAX_ARRAY_SIZE
) -> List[List[Tuple[SolverTask, str]]]:
    """Split all (solver, cnf) pairs into chunks for job arrays."""

    # Create all (solver, cnf) pairs
    all_pairs = []
    for solver in solver_tasks:
        for cnf in cnf_files:
            all_pairs.append((solver, cnf))

    # Split into chunks
    chunks = []
    for i in range(0, len(all_pairs), chunk_size):
        chunks.append(all_pairs[i:i + chunk_size])

    return chunks


def create_array_script(
    chunk_id: int,
    pairs: List[Tuple[SolverTask, str]],
    benchmark_path: str,
    work_dir: str,
    timeout: int,
) -> Tuple[str, int]:
    """Create a job array script for a chunk of tasks."""

    chunk_dir = os.path.join(work_dir, f"chunk_{chunk_id:03d}")
    os.makedirs(chunk_dir, exist_ok=True)
    os.makedirs(os.path.join(chunk_dir, "logs"), exist_ok=True)

    # Write task list: solver_path\tresult_dir\tcnf_file
    task_list_path = os.path.join(chunk_dir, "tasks.txt")
    with open(task_list_path, 'w') as f:
        for solver, cnf in pairs:
            f.write(f"{solver.solver_path}\t{solver.result_dir}\t{cnf}\n")

    # Create wrapper script
    script_path = os.path.join(chunk_dir, "run.sh")
    script_content = f"""#!/bin/bash
# Job array script for chunk {chunk_id}
# Tasks: {len(pairs)}

TASK_FILE="{task_list_path}"
BENCHMARK_PATH="{os.path.abspath(benchmark_path)}"
TIMEOUT={timeout}

# Get task info for this array index
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_FILE")
SOLVER_PATH=$(echo "$LINE" | cut -f1)
RESULT_DIR=$(echo "$LINE" | cut -f2)
CNF_FILE=$(echo "$LINE" | cut -f3)

CNF_PATH="${{BENCHMARK_PATH}}/${{CNF_FILE}}"
OUTPUT_FILE="${{RESULT_DIR}}/${{CNF_FILE}}.solving.log"

# Create result directory
mkdir -p "$RESULT_DIR"

# Skip if already done
if [ -f "$OUTPUT_FILE" ]; then
    echo "Already done: $OUTPUT_FILE"
    exit 0
fi

# Run solver
echo "Task $SLURM_ARRAY_TASK_ID: $SOLVER_PATH on $CNF_FILE"
timeout ${{TIMEOUT}}s "$SOLVER_PATH" "$CNF_PATH" > "$OUTPUT_FILE" 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "TIMEOUT after ${{TIMEOUT}}s" >> "$OUTPUT_FILE"
fi

echo "Exit code: $EXIT_CODE"
"""

    with open(script_path, 'w') as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)

    return script_path, len(pairs)


def submit_array(
    script_path: str,
    num_tasks: int,
    chunk_id: int,
    work_dir: str,
    timeout: int,
    max_concurrent: int,
    dry_run: bool = False,
) -> int:
    """Submit a job array and return job ID."""

    chunk_dir = os.path.join(work_dir, f"chunk_{chunk_id:03d}")

    # Calculate wall time (timeout + 2 min buffer)
    wall_minutes = (timeout + 120) // 60
    wall_time = f"00:{wall_minutes:02d}:00"

    array_spec = f"0-{num_tasks - 1}%{max_concurrent}"

    cmd = f"""sbatch \
        --job-name=eval_{chunk_id:03d} \
        --array={array_spec} \
        --time={wall_time} \
        --mem=4G \
        --cpus-per-task=1 \
        --account=def-vganesh \
        --output={chunk_dir}/logs/%a.log \
        --error={chunk_dir}/logs/%a.err \
        {script_path}"""

    if dry_run:
        logger.info(f"[DRY RUN] Would submit: {cmd}")
        return -1

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Failed to submit chunk {chunk_id}: {result.stderr}")
        return -1

    # Parse job ID
    job_id = int(result.stdout.strip().split()[-1])
    return job_id


def wait_for_jobs(job_ids: List[int], check_interval: int = 60):
    """Wait for all jobs to complete."""

    job_ids = [j for j in job_ids if j > 0]
    if not job_ids:
        return

    logger.info(f"Waiting for {len(job_ids)} job arrays to complete...")

    while True:
        # Check how many are still running
        job_str = ",".join(str(j) for j in job_ids)
        result = subprocess.run(
            f"squeue -j {job_str} -h 2>/dev/null | wc -l",
            shell=True, capture_output=True, text=True
        )

        remaining = int(result.stdout.strip()) if result.stdout.strip() else 0

        if remaining == 0:
            logger.info("All jobs completed!")
            return

        logger.info(f"{remaining} tasks still running, checking again in {check_interval}s...")
        time.sleep(check_interval)


def collect_results(solver_tasks: List[SolverTask], cnf_files: List[str], penalty: int):
    """Collect results and update database."""

    logger.info("Collecting results...")

    for solver in solver_tasks:
        total_time = 0.0
        solved = 0
        timeouts = 0

        for cnf in cnf_files:
            log_file = os.path.join(solver.result_dir, f"{cnf}.solving.log")

            if not os.path.exists(log_file):
                total_time += penalty
                timeouts += 1
                continue

            with open(log_file, 'r') as f:
                content = f.read()

            if "TIMEOUT" in content:
                total_time += penalty
                timeouts += 1
                continue

            # Parse process-time
            found_time = False
            for line in content.split('\n'):
                if 'process-time' in line.lower():
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'seconds' and i > 0:
                            try:
                                total_time += float(parts[i-1])
                                solved += 1
                                found_time = True
                                break
                            except ValueError:
                                pass
                    if found_time:
                        break

            if not found_time:
                total_time += penalty
                timeouts += 1

        par2 = total_time / len(cnf_files) if cnf_files else 0

        logger.info(f"Code {solver.code_id[:16]}: PAR2={par2:.2f}, Solved={solved}/{len(cnf_files)}, Timeout={timeouts}")

        # Update database
        code_result = get_code_result(solver.code_id)
        if code_result:
            code_result.par2 = par2
            code_result.status = CodeStatus.Evaluated
            update_code_result(code_result)


def main():
    parser = argparse.ArgumentParser(description="Submit parallel SLURM job arrays for solver evaluation")
    parser.add_argument("--tag", "-t", required=True, help="Generation tag")
    parser.add_argument("--benchmark-path", "-b", default=SAT2025_BENCHMARK_PATH, help="Benchmark directory")
    parser.add_argument("--cnf-file", help="File with CNF list (one per line)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help=f"Timeout per CNF (default: {DEFAULT_TIMEOUT}s)")
    parser.add_argument("--penalty", type=int, default=DEFAULT_PENALTY, help=f"Penalty for timeout (default: {DEFAULT_PENALTY}s)")
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT_PER_ARRAY, help=f"Max concurrent tasks per array (default: {MAX_CONCURRENT_PER_ARRAY})")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Don't submit, just show what would be done")
    parser.add_argument("--no-wait", action="store_true", help="Submit and exit without waiting")
    parser.add_argument("--collect-only", action="store_true", help="Only collect results, don't submit")
    parser.add_argument("--force", action="store_true", help="Force re-collection of already evaluated codes")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Number of arrays to submit at once (default: 1 to avoid QOS limits)")
    parser.add_argument("--chunk-size", type=int, default=400,
                       help="Tasks per array (default: 400 to stay under QOS 500 job limit)")

    args = parser.parse_args()

    work_dir = f"data/results/parallel_eval/{args.tag}"
    os.makedirs(work_dir, exist_ok=True)

    # Get tasks and CNFs
    solver_tasks = get_all_solver_tasks(args.tag, skip_evaluated=True, force=args.force)
    cnf_files = get_cnf_files(args.benchmark_path, args.cnf_file)

    if not solver_tasks:
        logger.error("No solver tasks found!")
        sys.exit(1)

    if not cnf_files:
        logger.error("No CNF files found!")
        sys.exit(1)

    total_tasks = len(solver_tasks) * len(cnf_files)
    logger.info(f"Total: {len(solver_tasks)} solvers × {len(cnf_files)} CNFs = {total_tasks} tasks")

    if args.collect_only:
        collect_results(solver_tasks, cnf_files, args.penalty)
        return

    # Create chunks (use smaller chunk size to stay under QOS limit)
    chunk_size = args.chunk_size if hasattr(args, 'chunk_size') else MAX_ARRAY_SIZE
    chunks = create_task_chunks(solver_tasks, cnf_files, chunk_size)
    logger.info(f"Split into {len(chunks)} job arrays (max {chunk_size} tasks each)")

    # Estimate time
    rounds_per_chunk = (chunk_size + args.max_concurrent - 1) // args.max_concurrent
    batch_size = args.batch_size if hasattr(args, 'batch_size') else 1
    num_batches = (len(chunks) + batch_size - 1) // batch_size
    # Assume ~50% timeout rate, average task time is ~timeout/2
    avg_time_per_task = args.timeout * 0.5
    estimated_time_per_chunk = rounds_per_chunk * avg_time_per_task / 60  # minutes
    estimated_total = estimated_time_per_chunk * num_batches / batch_size
    logger.info(f"Estimated time: ~{estimated_total:.0f} minutes ({num_batches} batches of {batch_size} arrays, {args.max_concurrent} concurrent per array)")

    # Submit chunks in batches to avoid QOS limits
    all_job_ids = []

    for batch_start in range(0, len(chunks), batch_size):
        batch_end = min(batch_start + batch_size, len(chunks))
        batch_chunks = chunks[batch_start:batch_end]
        batch_num = batch_start // batch_size + 1

        logger.info(f"Submitting batch {batch_num}/{num_batches} (chunks {batch_start}-{batch_end-1})")

        batch_job_ids = []
        for i, chunk in enumerate(batch_chunks):
            chunk_id = batch_start + i
            script_path, num_tasks = create_array_script(
                chunk_id=chunk_id,
                pairs=chunk,
                benchmark_path=args.benchmark_path,
                work_dir=work_dir,
                timeout=args.timeout,
            )

            job_id = submit_array(
                script_path=script_path,
                num_tasks=num_tasks,
                chunk_id=chunk_id,
                work_dir=work_dir,
                timeout=args.timeout,
                max_concurrent=args.max_concurrent,
                dry_run=args.dry_run,
            )

            if job_id > 0:
                batch_job_ids.append(job_id)
                all_job_ids.append(job_id)
                logger.info(f"Submitted chunk {chunk_id}/{len(chunks)-1}: job {job_id} with {num_tasks} tasks")

        if args.dry_run:
            continue

        # Wait for this batch to complete before submitting next batch
        if batch_end < len(chunks) and batch_job_ids:
            logger.info(f"Waiting for batch {batch_num} to complete before submitting next batch...")
            wait_for_jobs(batch_job_ids, check_interval=30)

    if args.dry_run:
        logger.info(f"[DRY RUN] Would have submitted {len(chunks)} job arrays in {num_batches} batches")
        return

    logger.info(f"Submitted {len(all_job_ids)} job arrays total")

    # Save job IDs
    with open(os.path.join(work_dir, "job_ids.json"), 'w') as f:
        json.dump({"job_ids": all_job_ids, "tag": args.tag, "timeout": args.timeout}, f)

    if args.no_wait:
        logger.info("Exiting without waiting (--no-wait). Run with --collect-only later.")
        return

    # Wait for any remaining jobs and collect results
    wait_for_jobs(all_job_ids)
    collect_results(solver_tasks, cnf_files, args.penalty)

    logger.info("Done!")


if __name__ == "__main__":
    main()