#!/usr/bin/env python3
"""
Sequential evaluation of LLM-generated solvers, algorithm by algorithm.
Designed to stay well under SLURM QOS job limits.

This script:
1. Builds solvers from generated code (if not already built)
2. Evaluates one algorithm at a time (all 10 codes)
3. Waits for completion before moving to the next
4. Uses job arrays to minimize job count OR direct evaluation (--direct flag)
5. Handles CNF instances efficiently
"""

import argparse
import os
import sys
import subprocess
import time
import json
import shutil
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llmsat.llmsat import (
    BASE_SOLVER_PATH,
    SAT2025_BENCHMARK_PATH,
    setup_logging,
    get_logger,
    CodeResult,
    CodeStatus,
    AlgorithmResult,
    AlgorithmStatus,
)
from llmsat.utils.aws import (
    get_ids_from_router_table,
    get_algorithm_result,
    get_code_result,
    update_code_result,
    update_algorithm_result,
)
from llmsat.llmsat import CHATGPT_DATA_GENERATION_TABLE
from llmsat.utils.paths import get_solver_dir, get_algorithm_dir

setup_logging()
logger = get_logger(__name__)

# Evaluation settings - for DPO data generation (short runs)
TIMEOUT_SECONDS = 600  # 10 minutes
PENALTY_SECONDS = 1200  # 20 minutes (2x timeout)
SLURM_TIME = "00:15:00"  # 15 minutes wall time (with buffer)
MAX_CONCURRENT_TASKS = 50  # Limit concurrent tasks per array
DIRECT_WORKERS = 8  # Number of parallel workers for direct evaluation


def run_solver_on_cnf(args: Tuple[str, str, str, str, int]) -> Tuple[str, str, Optional[float]]:
    """
    Run a single solver on a single CNF file.
    
    Args:
        args: Tuple of (solver_path, cnf_path, result_dir, cnf_filename, timeout)
    
    Returns:
        Tuple of (solver_path, cnf_filename, solving_time or None if timeout/error)
    """
    solver_path, cnf_path, result_dir, cnf_filename, timeout = args
    
    os.makedirs(result_dir, exist_ok=True)
    output_file = os.path.join(result_dir, f"{cnf_filename}.solving.log")
    
    # Skip if already done
    if os.path.exists(output_file):
        # Parse existing result
        try:
            with open(output_file, 'r') as f:
                content = f.read()
            if "TIMEOUT" in content:
                return (solver_path, cnf_filename, None)
            for line in content.split('\n'):
                if 'process-time' in line.lower():
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'seconds' and i > 0:
                            return (solver_path, cnf_filename, float(parts[i-1]))
        except:
            pass
        return (solver_path, cnf_filename, None)
    
    try:
        # Run solver with timeout
        result = subprocess.run(
            [solver_path, cnf_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        # Write output to log file
        with open(output_file, 'w') as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)
        
        # Parse solving time from output
        for line in result.stdout.split('\n'):
            if 'process-time' in line.lower():
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'seconds' and i > 0:
                        try:
                            return (solver_path, cnf_filename, float(parts[i-1]))
                        except ValueError:
                            pass
        
        return (solver_path, cnf_filename, None)
        
    except subprocess.TimeoutExpired:
        # Write timeout marker
        with open(output_file, 'w') as f:
            f.write(f"TIMEOUT after {timeout}s\n")
        return (solver_path, cnf_filename, None)
        
    except Exception as e:
        # Write error
        with open(output_file, 'w') as f:
            f.write(f"ERROR: {str(e)}\n")
        return (solver_path, cnf_filename, None)


def evaluate_solvers_direct(
    solvers: List['SolverInfo'],
    cnf_files: List[str],
    benchmark_path: str,
    timeout: int,
    num_workers: int = DIRECT_WORKERS
) -> Dict[str, Dict]:
    """
    Evaluate all solvers on all CNFs using direct multiprocessing.
    
    Returns dict mapping code_id -> results dict
    """
    # Build task list
    tasks = []
    for solver in solvers:
        for cnf in cnf_files:
            cnf_path = os.path.join(benchmark_path, cnf)
            tasks.append((solver.solver_path, cnf_path, solver.result_dir, cnf, timeout))
    
    total_tasks = len(tasks)
    logger.info(f"Running {total_tasks} evaluation tasks with {num_workers} workers...")
    
    # Run with multiprocessing
    completed = 0
    start_time = time.time()
    
    with Pool(num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(run_solver_on_cnf, tasks)):
            completed += 1
            if completed % 100 == 0 or completed == total_tasks:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total_tasks - completed) / rate if rate > 0 else 0
                logger.info(f"Progress: {completed}/{total_tasks} ({100*completed/total_tasks:.1f}%), "
                           f"Rate: {rate:.1f}/s, ETA: {eta/60:.1f} min")
    
    logger.info(f"Completed all {total_tasks} tasks in {(time.time()-start_time)/60:.1f} minutes")
    
    # Collect results per solver
    results = {}
    for solver in solvers:
        solver_results = collect_results_for_solver(solver, cnf_files)
        results[solver.code_id] = solver_results
    
    return results


def filter_code(code: str) -> Optional[str]:
    """Extract and normalize the kissat_restarting function from generated code."""
    def normalize_escaped_whitespace(text: str) -> str:
        text = text.replace('\\n', '\n')
        text = text.replace('\\t', '\t')
        text = text.replace('\\r', '\r')
        text = text.replace('\\"', '"')
        text = text.replace("\\'", "'")
        text = text.replace('\\\\', '\\')
        text = text.replace("\r\n", "\n")
        return text

    def extract_function(text: str, func_name: str) -> Optional[str]:
        header_regex = rf"(?:static\s+)?(?:inline\s+)?bool\s+{func_name}\b[^\{{]*\{{"
        m = re.search(header_regex, text, flags=re.DOTALL)
        if m:
            start = m.start()
            open_brace = m.end() - 1
            brace = 0
            i = open_brace
            while i < len(text):
                c = text[i]
                if c == "{":
                    brace += 1
                elif c == "}":
                    brace -= 1
                    if brace == 0:
                        end = i + 1
                        return text[start:end]
                i += 1
            return None
        name_idx = text.find(func_name)
        if name_idx == -1:
            return None
        after_name = text[name_idx + len(func_name):]
        brace_idx = after_name.find("{")
        if brace_idx == -1:
            return None
        header_prefix = f"bool {func_name}"
        header_suffix = after_name[: brace_idx + 1]
        header = f"{header_prefix}{header_suffix}"
        rest = after_name[brace_idx + 1:]
        brace = 1
        i = 0
        while i < len(rest):
            c = rest[i]
            if c == "{":
                brace += 1
            elif c == "}":
                brace -= 1
                if brace == 0:
                    body_end = i + 1
                    return header + rest[:body_end]
            i += 1
        return None

    # Extract content within <code>...</code> if present
    if "<code>" in code:
        code = code.split("<code>")[1].split("</code>")[0]

    code = normalize_escaped_whitespace(code)

    if "kissat_restarting" not in code:
        logger.error("No kissat_restarting function found in code")
        return None

    extracted = extract_function(code, "kissat_restarting")
    if not extracted:
        logger.error("Failed to parse kissat_restarting function body")
        return None
    return extracted


def build_solver(code_result: CodeResult) -> Optional[str]:
    """Build a solver from code_result. Returns solver path if successful, None otherwise."""
    logger.info(f"Building solver for code {code_result.id[:16]}...")
    
    code = filter_code(code_result.code)
    if code is None:
        logger.error("Failed to extract kissat_restarting function from code")
        return None
    
    # Create solver directory
    new_solver_path = get_solver_dir(code_result.algorithm_id, code_result.id)
    logger.info(f"Building solver at {new_solver_path}")
    
    if os.path.exists(new_solver_path):
        shutil.rmtree(new_solver_path)
    shutil.copytree(BASE_SOLVER_PATH, new_solver_path)
    
    # Replace code in restart.c
    restart_file = f"{new_solver_path}/src/restart.c"
    with open(restart_file, "r") as f:
        lines = f.readlines()
    
    # Find LLMSAT markers
    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if start_idx is None and (line.startswith("//LLMSAT start") or stripped.startswith("// LLMSAT: start")):
            start_idx = i
            continue
        if start_idx is not None and (line.startswith("//LLMSAT end") or stripped.startswith("// LLMSAT: end")):
            end_idx = i
            break
    
    if start_idx is None or end_idx is None:
        logger.error("Could not find LLMSAT markers in restart.c")
        return None
    
    # Write modified file
    with open(restart_file, "w") as f:
        f.writelines(lines[:start_idx + 1])
        f.write(code.rstrip() + "\n")
        f.writelines(lines[end_idx:])
    
    # Compile solver
    try:
        configure_proc = subprocess.run(
            ["./configure"],
            cwd=new_solver_path,
            capture_output=True,
            text=True,
        )
        
        if configure_proc.returncode != 0:
            logger.error(f"Configure failed: {configure_proc.stderr}")
            return None
        
        make_proc = subprocess.run(
            ["make", "-j4"],
            cwd=new_solver_path,
            capture_output=True,
            text=True,
        )
        
        if make_proc.returncode != 0:
            logger.error(f"Make failed: {make_proc.stderr}")
            # Save build log for debugging
            algorithm_dir = get_algorithm_dir(code_result.algorithm_id)
            os.makedirs(algorithm_dir, exist_ok=True)
            failed_log_path = f"{algorithm_dir}/code_{code_result.id}.build_failed.log"
            with open(failed_log_path, "w") as f:
                f.write(f"=== configure stdout ===\n{configure_proc.stdout}\n")
                f.write(f"=== configure stderr ===\n{configure_proc.stderr}\n")
                f.write(f"=== make stdout ===\n{make_proc.stdout}\n")
                f.write(f"=== make stderr ===\n{make_proc.stderr}\n")
            return None
        
        logger.info(f"Build successful for code {code_result.id[:16]}")
        return new_solver_path
        
    except Exception as e:
        logger.error(f"Build exception: {e}")
        return None


@dataclass
class SolverInfo:
    """Information about a solver to evaluate."""
    algorithm_id: str
    code_id: str
    solver_path: str
    result_dir: str
    

def get_solvers_for_tag(generation_tag: str, skip_build: bool = False) -> Dict[str, List[SolverInfo]]:
    """Get all solvers grouped by algorithm for a generation tag.
    
    Args:
        generation_tag: The generation tag to get solvers for
        skip_build: If True, skip solvers that aren't built. If False, build them.
    """
    algorithm_ids = get_ids_from_router_table(CHATGPT_DATA_GENERATION_TABLE, generation_tag)
    
    solvers_by_algorithm = {}
    build_success = 0
    build_failed = 0
    already_built = 0
    already_evaluated = 0
    
    for algorithm_id in algorithm_ids:
        algorithm = get_algorithm_result(algorithm_id)
        if not algorithm:
            logger.warning(f"Algorithm {algorithm_id} not found, skipping")
            continue
            
        solvers = []
        for code_id in algorithm.code_id_list:
            code_result = get_code_result(code_id)
            if not code_result:
                logger.warning(f"Code {code_id} not found, skipping")
                continue
            
            # Check if already evaluated
            if code_result.status == CodeStatus.Evaluated and code_result.par2 is not None:
                logger.info(f"Code {code_id[:16]} already evaluated (PAR2: {code_result.par2}), skipping")
                already_evaluated += 1
                continue
            
            # Check if build already failed
            if code_result.status == CodeStatus.BuildFailed:
                logger.info(f"Code {code_id[:16]} previously failed to build, skipping")
                build_failed += 1
                continue
            
            # Path mapping: solvers use algorithm_<id>/code_<id>/ structure
            solver_path = f"solvers/algorithm_{algorithm_id}/code_{code_id}/build/kissat"
            result_dir = f"solvers/algorithm_{algorithm_id}/result/code_{code_id}"
            
            if not os.path.exists(solver_path):
                if skip_build:
                    logger.warning(f"Solver binary not found at {solver_path}, skipping (--skip-build)")
                    continue
                
                # Try to build the solver
                logger.info(f"Solver not found, attempting to build...")
                built_path = build_solver(code_result)
                
                if built_path is None:
                    # Build failed - update status
                    code_result.status = CodeStatus.BuildFailed
                    update_code_result(code_result)
                    build_failed += 1
                    continue
                
                build_success += 1
            else:
                already_built += 1
                
            solvers.append(SolverInfo(
                algorithm_id=algorithm_id,
                code_id=code_id,
                solver_path=os.path.abspath(solver_path),
                result_dir=os.path.abspath(result_dir),
            ))
        
        if solvers:
            solvers_by_algorithm[algorithm_id] = solvers
    
    logger.info(f"Build summary: {build_success} newly built, {already_built} already built, "
                f"{build_failed} failed, {already_evaluated} already evaluated")
    
    return solvers_by_algorithm


def get_cnf_files(benchmark_path: str, cnf_file: str = None) -> List[str]:
    """Get CNF files from the benchmark directory or from a CNF list file."""
    if cnf_file:
        # Load CNF filenames from file
        if not os.path.exists(cnf_file):
            raise FileNotFoundError(f"CNF file list not found: {cnf_file}")
        with open(cnf_file, 'r') as f:
            cnf_files = [line.strip() for line in f if line.strip()]
        # Validate that files exist
        missing = [cnf for cnf in cnf_files if not os.path.exists(os.path.join(benchmark_path, cnf))]
        if missing:
            logger.warning(f"{len(missing)} CNF files from list not found in {benchmark_path}")
        return [cnf for cnf in cnf_files if os.path.exists(os.path.join(benchmark_path, cnf))]
    else:
        # Scan directory
        cnf_files = []
        for f in sorted(os.listdir(benchmark_path)):
            if f.endswith(".cnf"):
                cnf_files.append(f)
        return cnf_files


def create_mega_job_array(
    solvers: List[SolverInfo],
    cnf_files: List[str],
    benchmark_path: str,
    work_dir: str,
) -> str:
    """
    Create a single job array that evaluates all solvers on all CNFs.
    
    Task ID mapping: task_id = solver_index * num_cnfs + cnf_index
    """
    os.makedirs(work_dir, exist_ok=True)
    
    num_solvers = len(solvers)
    num_cnfs = len(cnf_files)
    total_tasks = num_solvers * num_cnfs
    
    # Write solver list
    solver_list_path = os.path.join(work_dir, "solver_list.txt")
    with open(solver_list_path, "w") as f:
        for solver in solvers:
            f.write(f"{solver.solver_path}\t{solver.result_dir}\n")
    
    # Write CNF list
    cnf_list_path = os.path.join(work_dir, "cnf_list.txt")
    with open(cnf_list_path, "w") as f:
        for cnf in cnf_files:
            f.write(f"{cnf}\n")
    
    # Write metadata
    metadata_path = os.path.join(work_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump({
            "num_solvers": num_solvers,
            "num_cnfs": num_cnfs,
            "total_tasks": total_tasks,
            "timeout": TIMEOUT_SECONDS,
            "penalty": PENALTY_SECONDS,
            "benchmark_path": benchmark_path,
            "solvers": [{"algorithm_id": s.algorithm_id, "code_id": s.code_id} for s in solvers],
        }, f, indent=2)
    
    # Create wrapper script
    wrapper_script = os.path.join(work_dir, "run_evaluation.sh")
    with open(wrapper_script, "w") as f:
        f.write(f"""#!/bin/bash
# Auto-generated evaluation script
# Task ID = solver_index * num_cnfs + cnf_index

TASK_ID=$SLURM_ARRAY_TASK_ID
NUM_CNFS={num_cnfs}
BENCHMARK_PATH="{os.path.abspath(benchmark_path)}"
TIMEOUT={TIMEOUT_SECONDS}

# Calculate solver and CNF indices
SOLVER_IDX=$((TASK_ID / NUM_CNFS))
CNF_IDX=$((TASK_ID % NUM_CNFS))

# Read solver info (path and result dir)
SOLVER_LINE=$(sed -n "$((SOLVER_IDX + 1))p" "{solver_list_path}")
SOLVER_PATH=$(echo "$SOLVER_LINE" | cut -f1)
RESULT_DIR=$(echo "$SOLVER_LINE" | cut -f2)

# Read CNF filename
CNF_FILE=$(sed -n "$((CNF_IDX + 1))p" "{cnf_list_path}")
CNF_PATH="${{BENCHMARK_PATH}}/${{CNF_FILE}}"

# Create result directory
mkdir -p "$RESULT_DIR"

# Output file
OUTPUT_FILE="${{RESULT_DIR}}/${{CNF_FILE}}.solving.log"

# Skip if already done
if [ -f "$OUTPUT_FILE" ]; then
    echo "Already completed: $OUTPUT_FILE"
    exit 0
fi

# Run solver with timeout
echo "Running: $SOLVER_PATH on $CNF_FILE"
echo "Task $TASK_ID: Solver $SOLVER_IDX, CNF $CNF_IDX"
timeout ${{TIMEOUT}}s "$SOLVER_PATH" "$CNF_PATH" > "$OUTPUT_FILE" 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "TIMEOUT after ${{TIMEOUT}}s" >> "$OUTPUT_FILE"
fi

echo "Completed with exit code: $EXIT_CODE"
""")
    os.chmod(wrapper_script, 0o755)
    
    return wrapper_script, total_tasks


def submit_job_array(wrapper_script: str, total_tasks: int, work_dir: str, job_name: str) -> int:
    """Submit the job array and return the job ID."""
    
    # Limit array size and concurrent tasks
    array_spec = f"0-{total_tasks - 1}%{MAX_CONCURRENT_TASKS}"
    
    sbatch_cmd = f"""sbatch \
        --job-name={job_name} \
        --array={array_spec} \
        --time={SLURM_TIME} \
        --mem=8G \
        --cpus-per-task=1 \
        --qos=coc-ice \
        --output={work_dir}/logs/task_%a.log \
        --error={work_dir}/logs/task_%a.err \
        {wrapper_script}"""
    
    os.makedirs(f"{work_dir}/logs", exist_ok=True)
    
    logger.info(f"Submitting job array with {total_tasks} tasks (max {MAX_CONCURRENT_TASKS} concurrent)")
    result = subprocess.run(sbatch_cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to submit job array: {result.stderr}")
    
    # Parse job ID from "Submitted batch job 12345"
    job_id = int(result.stdout.strip().split()[-1])
    logger.info(f"Submitted job array {job_id}")
    
    return job_id


def wait_for_job(job_id: int, check_interval: int = 60) -> bool:
    """Wait for a job array to complete."""
    logger.info(f"Waiting for job {job_id} to complete...")
    
    while True:
        # Check if job is still running
        result = subprocess.run(
            f"squeue -j {job_id} -h | wc -l",
            shell=True, capture_output=True, text=True
        )
        
        remaining = int(result.stdout.strip())
        
        if remaining == 0:
            logger.info(f"Job {job_id} completed")
            return True
        
        logger.info(f"Job {job_id}: {remaining} tasks remaining, checking again in {check_interval}s")
        time.sleep(check_interval)


def parse_solving_time(log_file: str) -> Optional[float]:
    """Parse solving time from a .solving.log file."""
    if not os.path.exists(log_file):
        return None
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Check for timeout
    if "TIMEOUT" in content:
        return None
    
    # Look for "process-time" line
    for line in content.split('\n'):
        if 'process-time' in line.lower():
            parts = line.split()
            for i, part in enumerate(parts):
                if part == 'seconds' and i > 0:
                    try:
                        return float(parts[i-1])
                    except ValueError:
                        continue
    
    return None


def collect_results_for_solver(solver: SolverInfo, cnf_files: List[str]) -> Dict:
    """Collect results for a single solver."""
    total_time = 0.0
    solved = 0
    timeout = 0
    errors = 0
    
    for cnf in cnf_files:
        log_file = os.path.join(solver.result_dir, f"{cnf}.solving.log")
        time_taken = parse_solving_time(log_file)
        
        if time_taken is not None:
            total_time += time_taken
            solved += 1
        elif os.path.exists(log_file):
            # File exists but no time = timeout or error
            total_time += PENALTY_SECONDS
            timeout += 1
        else:
            # No log file = didn't run
            total_time += PENALTY_SECONDS
            errors += 1
    
    par2 = total_time / len(cnf_files) if cnf_files else 0
    
    return {
        "par2": par2,
        "solved": solved,
        "timeout": timeout,
        "errors": errors,
        "total": len(cnf_files),
    }


def update_database_results(solvers: List[SolverInfo], cnf_files: List[str]):
    """Update database with evaluation results."""
    for solver in solvers:
        results = collect_results_for_solver(solver, cnf_files)
        
        logger.info(f"Code {solver.code_id}: PAR2={results['par2']:.2f}, "
                   f"Solved={results['solved']}/{results['total']}, "
                   f"Timeout={results['timeout']}, Errors={results['errors']}")
        
        # Update database
        code_result = get_code_result(solver.code_id)
        if code_result:
            code_result.par2 = results['par2']
            code_result.status = CodeStatus.Evaluated
            update_code_result(code_result)
            logger.info(f"Updated database for code {solver.code_id}")


def evaluate_algorithm(
    algorithm_id: str,
    solvers: List[SolverInfo],
    cnf_files: List[str],
    benchmark_path: str,
    dry_run: bool = False,
) -> bool:
    """Evaluate all solvers for a single algorithm."""
    
    logger.info(f"=" * 60)
    logger.info(f"Evaluating algorithm: {algorithm_id}")
    logger.info(f"  Solvers: {len(solvers)}")
    logger.info(f"  CNFs: {len(cnf_files)}")
    logger.info(f"  Total tasks: {len(solvers) * len(cnf_files)}")
    logger.info(f"=" * 60)
    
    work_dir = f"data/results/sequential_eval/{algorithm_id}"
    
    # Create the mega job array
    wrapper_script, total_tasks = create_mega_job_array(
        solvers, cnf_files, benchmark_path, work_dir
    )
    
    if dry_run:
        logger.info(f"[DRY RUN] Would submit {total_tasks} tasks")
        logger.info(f"[DRY RUN] Wrapper script: {wrapper_script}")
        return True
    
    # Submit and wait
    job_id = submit_job_array(wrapper_script, total_tasks, work_dir, f"eval_{algorithm_id[:8]}")
    wait_for_job(job_id)
    
    # Collect results and update database
    update_database_results(solvers, cnf_files)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Sequential evaluation of LLM-generated solvers"
    )
    parser.add_argument(
        "--tag", "-t",
        required=True,
        help="Generation tag to evaluate"
    )
    parser.add_argument(
        "--benchmark-path", "-b",
        default=SAT2025_BENCHMARK_PATH,
        help=f"Path to benchmark CNFs (default: {SAT2025_BENCHMARK_PATH})"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Don't actually submit jobs, just show what would be done"
    )
    parser.add_argument(
        "--algorithm", "-a",
        help="Evaluate only this algorithm ID (default: all)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=TIMEOUT_SECONDS,
        help=f"Timeout in seconds per CNF (default: {TIMEOUT_SECONDS})"
    )
    parser.add_argument(
        "--penalty",
        type=int,
        default=PENALTY_SECONDS,
        help=f"Penalty for timeout/error in seconds (default: {PENALTY_SECONDS})"
    )
    parser.add_argument(
        "--cnf-file",
        help="Path to file containing CNF filenames (one per line). If not provided, scans benchmark directory."
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip solvers that aren't already built (don't attempt to build them)"
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Run evaluations directly using multiprocessing instead of SLURM job arrays"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=DIRECT_WORKERS,
        help=f"Number of parallel workers for direct evaluation (default: {DIRECT_WORKERS})"
    )
    
    args = parser.parse_args()
    
    timeout = args.timeout
    penalty = args.penalty
    slurm_time = f"{(timeout + 120) // 3600:02d}:{((timeout + 120) % 3600) // 60:02d}:00"
    
    logger.info(f"Settings: timeout={timeout}s, penalty={penalty}s")
    if args.direct:
        logger.info(f"Mode: DIRECT evaluation with {args.workers} workers")
    else:
        logger.info(f"Mode: SLURM job arrays, slurm_time={slurm_time}")
    
    # Get CNF files
    cnf_files = get_cnf_files(args.benchmark_path, args.cnf_file)
    if not cnf_files:
        logger.error(f"No CNF files found in {args.benchmark_path}")
        sys.exit(1)
    logger.info(f"Found {len(cnf_files)} CNF files")
    
    # Get solvers grouped by algorithm (build if needed)
    solvers_by_algorithm = get_solvers_for_tag(args.tag, skip_build=args.skip_build)
    
    if not solvers_by_algorithm:
        logger.error(f"No solvers found for tag: {args.tag}")
        sys.exit(1)
    
    total_solvers = sum(len(s) for s in solvers_by_algorithm.values())
    logger.info(f"Found {len(solvers_by_algorithm)} algorithms with {total_solvers} solvers to evaluate")
    
    # Filter to specific algorithm if requested
    if args.algorithm:
        if args.algorithm not in solvers_by_algorithm:
            logger.error(f"Algorithm {args.algorithm} not found in tag {args.tag}")
            sys.exit(1)
        solvers_by_algorithm = {args.algorithm: solvers_by_algorithm[args.algorithm]}
    
    if args.direct:
        # Direct evaluation mode - run all solvers together using multiprocessing
        all_solvers = []
        for solvers in solvers_by_algorithm.values():
            all_solvers.extend(solvers)
        
        total_tasks = len(all_solvers) * len(cnf_files)
        logger.info(f"=" * 60)
        logger.info(f"DIRECT EVALUATION MODE")
        logger.info(f"  Total solvers: {len(all_solvers)}")
        logger.info(f"  CNFs per solver: {len(cnf_files)}")
        logger.info(f"  Total tasks: {total_tasks}")
        logger.info(f"  Workers: {args.workers}")
        logger.info(f"=" * 60)
        
        if args.dry_run:
            logger.info(f"[DRY RUN] Would run {total_tasks} evaluation tasks")
            return
        
        # Run all evaluations
        results = evaluate_solvers_direct(
            all_solvers,
            cnf_files,
            args.benchmark_path,
            timeout=timeout,
            num_workers=args.workers
        )
        
        # Update database
        for solver in all_solvers:
            if solver.code_id in results:
                solver_results = results[solver.code_id]
                logger.info(f"Code {solver.code_id[:16]}: PAR2={solver_results['par2']:.2f}, "
                           f"Solved={solver_results['solved']}/{solver_results['total']}")
                
                code_result = get_code_result(solver.code_id)
                if code_result:
                    code_result.par2 = solver_results['par2']
                    code_result.status = CodeStatus.Evaluated
                    update_code_result(code_result)
    else:
        # SLURM job array mode - evaluate algorithm by algorithm
        for algorithm_id, solvers in solvers_by_algorithm.items():
            try:
                evaluate_algorithm(
                    algorithm_id,
                    solvers,
                    cnf_files,
                    args.benchmark_path,
                    dry_run=args.dry_run,
                )
            except Exception as e:
                logger.error(f"Failed to evaluate algorithm {algorithm_id}: {e}")
                continue
    
    logger.info("=" * 60)
    logger.info("All evaluations complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()