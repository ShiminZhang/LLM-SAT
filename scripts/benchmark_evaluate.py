#!/usr/bin/env python3
"""
Build and evaluate benchmark solvers, then compute comparison metrics.

Takes generated codes from benchmark_generate_codes.py, builds solvers,
evaluates on CNF subset, and computes comparison metrics.

Usage:
    # Build all solvers
    python scripts/benchmark_evaluate.py --input outputs/benchmark/generated_codes.json --build

    # Submit evaluation jobs (SLURM)
    python scripts/benchmark_evaluate.py --input outputs/benchmark/generated_codes.json --evaluate

    # Collect results and compute metrics
    python scripts/benchmark_evaluate.py --input outputs/benchmark/generated_codes.json --analyze
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
import glob
import re
import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llmsat.llmsat import BASE_SOLVER_PATH, setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

# Configuration
DEFAULT_CNF_FILE = "data/cnf_subset_100.txt"
DEFAULT_BENCHMARK_DIR = "data/benchmarks/satcomp2025"
DEFAULT_TIMEOUT = 300  # 5 minutes
DEFAULT_PENALTY = 600  # 10 minutes (PAR-2)


def compute_code_hash(code: str) -> str:
    """Compute a short hash of the code for identification."""
    return hashlib.md5(code.encode()).hexdigest()[:12]


def filter_code(code: str) -> Optional[str]:
    """Extract kissat_restarting function from generated code and fix signature."""
    # Handle escape sequences
    code = code.replace("\\n", "\n").replace("\\t", "\t")

    # Try to find the function with various signatures the LLM might produce
    pattern = r'bool\s+kissat_restarting\s*\([^)]*\)\s*\{'
    match = re.search(pattern, code)

    if not match:
        return None

    # Find matching closing brace
    start = match.start()
    brace_count = 0
    end = start

    for i, char in enumerate(code[start:], start):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                end = i + 1
                break

    if brace_count != 0:
        return None

    extracted = code[start:end]

    # Fix the function signature to match what Kissat expects
    # Replace any variant of the signature with the correct one
    signature_pattern = r'bool\s+kissat_restarting\s*\([^)]*\)'
    correct_signature = 'bool kissat_restarting (kissat *solver)'
    fixed = re.sub(signature_pattern, correct_signature, extracted, count=1)

    return fixed


def build_solver(code: str, solver_path: str) -> Tuple[bool, str]:
    """Build a solver with the given code. Returns (success, error_message)."""

    # Filter the code to extract just the function
    filtered_code = filter_code(code)
    if filtered_code is None:
        return False, "Could not find kissat_restarting function in code"

    # Remove existing solver directory
    if os.path.exists(solver_path):
        shutil.rmtree(solver_path)

    # Copy base solver
    shutil.copytree(BASE_SOLVER_PATH, solver_path)

    # Inject code into restart.c
    restart_file = f"{solver_path}/src/restart.c"

    with open(restart_file, "r") as f:
        lines = f.readlines()

    # Find LLMSAT markers
    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if "LLMSAT" in line and "start" in line.lower():
            start_idx = i
        elif start_idx is not None and "LLMSAT" in line and "end" in line.lower():
            end_idx = i
            break

    if start_idx is None or end_idx is None:
        return False, "Could not find LLMSAT markers in restart.c"

    # Write modified file
    with open(restart_file, "w") as f:
        f.writelines(lines[:start_idx + 1])
        f.write(filtered_code.rstrip() + "\n")
        f.writelines(lines[end_idx:])

    # Compile
    try:
        configure = subprocess.run(
            ["./configure"],
            cwd=solver_path,
            capture_output=True,
            text=True,
            timeout=60
        )

        if configure.returncode != 0:
            return False, f"Configure failed: {configure.stderr}"

        make = subprocess.run(
            ["make", "-j4"],
            cwd=solver_path,
            capture_output=True,
            text=True,
            timeout=300
        )

        if make.returncode != 0:
            return False, f"Make failed: {make.stderr}"

        # Verify binary exists
        binary = f"{solver_path}/build/kissat"
        if not os.path.exists(binary):
            return False, "Build completed but binary not found"

        return True, ""

    except subprocess.TimeoutExpired:
        return False, "Build timed out"
    except Exception as e:
        return False, str(e)


def parse_solving_time(log_file: str, timeout: int, penalty: int) -> Optional[float]:
    """Parse solving time from a log file."""
    if not os.path.exists(log_file):
        return None

    try:
        with open(log_file, 'r') as f:
            content = f.read()
    except Exception:
        return penalty

    if not content.strip():
        return penalty

    if "TIMEOUT" in content or "CANCELLED" in content or "TIME LIMIT" in content:
        return penalty

    for line in content.split('\n'):
        if 'process-time' in line:
            match = re.search(r'(\d+\.?\d*)\s+seconds', line)
            if match:
                time = float(match.group(1))
                return penalty if time >= timeout else time

    for line in content.split('\n'):
        if 'error' in line.lower():
            return penalty

    return penalty


def submit_all_evaluation_jobs(results: List[Dict], output_dir: str, cnf_file: str,
                               benchmark_dir: str, timeout: int) -> Optional[int]:
    """Submit a SINGLE job array for ALL solver evaluations.

    This avoids hitting job limits by using one job array where:
    - Each task evaluates one (solver, cnf) pair
    - Task index = solver_idx * num_cnfs + cnf_idx
    """

    # Get successful builds only
    successful_results = [r for r in results if r.get('build_success', False)]

    if not successful_results:
        logger.error("No successfully built solvers to evaluate")
        return None

    # Read CNF files and prepend benchmark directory for full paths
    with open(cnf_file, 'r') as f:
        cnf_files = [os.path.join(benchmark_dir, line.strip()) for line in f if line.strip()]

    num_solvers = len(successful_results)
    num_cnfs = len(cnf_files)
    total_tasks = num_solvers * num_cnfs

    print(f"Submitting single job array: {num_solvers} solvers × {num_cnfs} CNFs = {total_tasks} tasks")

    # Create result directories
    for result in successful_results:
        os.makedirs(result['result_dir'], exist_ok=True)

    # Create task mapping file (task_id -> solver_binary, result_dir)
    task_map_file = os.path.join(output_dir, "task_mapping.txt")
    with open(task_map_file, 'w') as f:
        for solver_idx, result in enumerate(successful_results):
            binary = os.path.abspath(f"{result['solver_path']}/build/kissat")
            result_dir = os.path.abspath(result['result_dir'])
            f.write(f"{solver_idx}\t{binary}\t{result_dir}\n")

    # Create CNF list file
    cnf_list_file = os.path.join(output_dir, "cnf_list.txt")
    with open(cnf_list_file, 'w') as f:
        for cnf in cnf_files:
            f.write(f"{cnf}\n")

    # Create job script
    log_dir = os.path.join(output_dir, "slurm_logs")
    os.makedirs(log_dir, exist_ok=True)

    job_script = f"""#!/bin/bash
#SBATCH --job-name=bench_eval
#SBATCH --array=0-{total_tasks - 1}%200
#SBATCH --time=0-00:10:00
#SBATCH --account=def-vganesh
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output={log_dir}/slurm_%A_%a.log

# Calculate solver and CNF indices from task ID
NUM_CNFS={num_cnfs}
SOLVER_IDX=$((SLURM_ARRAY_TASK_ID / NUM_CNFS))
CNF_IDX=$((SLURM_ARRAY_TASK_ID % NUM_CNFS))

# Read task mapping
TASK_MAP="{os.path.abspath(task_map_file)}"
SOLVER_LINE=$(awk -v idx="$SOLVER_IDX" '$1 == idx {{print $2 "\\t" $3}}' "$TASK_MAP")
SOLVER_BINARY=$(echo "$SOLVER_LINE" | cut -f1)
RESULT_DIR=$(echo "$SOLVER_LINE" | cut -f2)

# Read CNF file
CNF_LIST="{os.path.abspath(cnf_list_file)}"
CNF=$(sed -n "$((CNF_IDX + 1))p" "$CNF_LIST")
CNF_NAME=$(basename "$CNF")

# Run solver
timeout {timeout}s "$SOLVER_BINARY" "$CNF" > "$RESULT_DIR/${{CNF_NAME}}.solving.log" 2>&1

if [ $? -eq 124 ]; then
    echo "TIMEOUT" >> "$RESULT_DIR/${{CNF_NAME}}.solving.log"
fi
"""

    script_path = os.path.join(output_dir, "eval_all_job.sh")
    with open(script_path, 'w') as f:
        f.write(job_script)

    # Submit job
    result = subprocess.run(
        ["sbatch", script_path],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        logger.error(f"Failed to submit job: {result.stderr}")
        return None

    # Extract job ID
    match = re.search(r'Submitted batch job (\d+)', result.stdout)
    if match:
        job_id = int(match.group(1))
        print(f"Submitted job array {job_id} with {total_tasks} tasks")
        return job_id

    return None


def collect_results(result_dir: str, cnf_file: str, timeout: int, penalty: int) -> Dict:
    """Collect evaluation results from log files."""

    with open(cnf_file, 'r') as f:
        cnf_files = [line.strip() for line in f if line.strip()]

    times = []
    solved = 0

    for cnf in cnf_files:
        cnf_name = os.path.basename(cnf)
        log_file = f"{result_dir}/{cnf_name}.solving.log"

        time = parse_solving_time(log_file, timeout, penalty)
        if time is not None:
            times.append(time)
            if time < penalty:
                solved += 1

    if not times:
        return {"par2": None, "solved": 0, "total": len(cnf_files)}

    par2 = sum(times) / len(times)

    return {
        "par2": par2,
        "solved": solved,
        "total": len(cnf_files),
        "solve_rate": solved / len(cnf_files),
        "times": times
    }


def compute_edit_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    m, n = len(s1), len(s2)

    # Use only two rows for space efficiency
    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                curr[j] = prev[j-1]
            else:
                curr[j] = 1 + min(prev[j], curr[j-1], prev[j-1])
        prev, curr = curr, prev

    return prev[n]


def compute_metrics(results: List[Dict], generated_codes: List[Dict]) -> Dict:
    """Compute comprehensive metrics for benchmark comparison."""

    # Separate by model type
    base_results = [r for r in results if r['model_type'] == 'base']
    ft_results = [r for r in results if r['model_type'] == 'finetuned']

    base_codes = [c for c in generated_codes if c['model_type'] == 'base']
    ft_codes = [c for c in generated_codes if c['model_type'] == 'finetuned']

    metrics = {
        "base": compute_model_metrics(base_results, base_codes),
        "finetuned": compute_model_metrics(ft_results, ft_codes),
        "comparison": {}
    }

    # Compute pairwise edit distances within each model (can be slow for long codes)
    print("  Computing edit distances (this may take a moment)...")
    if base_codes:
        base_distances = []
        for i, c1 in enumerate(base_codes):
            for j, c2 in enumerate(base_codes[i+1:], i+1):
                # Use a simpler metric for speed: normalized length difference + sample char diff
                dist = abs(len(c1['generated_code']) - len(c2['generated_code']))
                # Sample first 500 chars for quick comparison
                sample1 = c1['generated_code'][:500]
                sample2 = c2['generated_code'][:500]
                diff_chars = sum(1 for a, b in zip(sample1, sample2) if a != b)
                diff_chars += abs(len(sample1) - len(sample2))
                base_distances.append(dist + diff_chars)
        if base_distances:
            metrics["base"]["avg_pairwise_distance"] = sum(base_distances) / len(base_distances)

    if ft_codes:
        ft_distances = []
        for i, c1 in enumerate(ft_codes):
            for j, c2 in enumerate(ft_codes[i+1:], i+1):
                dist = abs(len(c1['generated_code']) - len(c2['generated_code']))
                sample1 = c1['generated_code'][:500]
                sample2 = c2['generated_code'][:500]
                diff_chars = sum(1 for a, b in zip(sample1, sample2) if a != b)
                diff_chars += abs(len(sample1) - len(sample2))
                ft_distances.append(dist + diff_chars)
        if ft_distances:
            metrics["finetuned"]["avg_pairwise_distance"] = sum(ft_distances) / len(ft_distances)
    print("  Done computing distances.")

    # Paired comparison (same algorithm, different model)
    if base_results and ft_results:
        paired = compute_paired_comparison(base_results, ft_results)
        metrics["comparison"] = paired

    return metrics


def compute_model_metrics(results: List[Dict], codes: List[Dict]) -> Dict:
    """Compute metrics for a single model."""

    if not results:
        return {}

    # Build success rate
    build_successes = [r for r in results if r.get('build_success', False)]
    build_rate = len(build_successes) / len(results) if results else 0

    # PAR2 scores (only for successful builds)
    par2_scores = [r['par2'] for r in results if r.get('par2') is not None]

    # Solve rates
    solve_rates = [r['solve_rate'] for r in results if r.get('solve_rate') is not None]

    # Code lengths
    code_lengths = [len(c['generated_code']) for c in codes]

    metrics = {
        "n_total": len(results),
        "n_build_success": len(build_successes),
        "build_success_rate": build_rate,
    }

    if par2_scores:
        metrics.update({
            "par2_mean": sum(par2_scores) / len(par2_scores),
            "par2_std": (sum((x - sum(par2_scores)/len(par2_scores))**2 for x in par2_scores) / len(par2_scores)) ** 0.5,
            "par2_min": min(par2_scores),
            "par2_max": max(par2_scores),
        })

    if solve_rates:
        metrics.update({
            "solve_rate_mean": sum(solve_rates) / len(solve_rates),
        })

    if code_lengths:
        metrics.update({
            "code_length_mean": sum(code_lengths) / len(code_lengths),
            "code_length_std": (sum((x - sum(code_lengths)/len(code_lengths))**2 for x in code_lengths) / len(code_lengths)) ** 0.5,
            "code_length_min": min(code_lengths),
            "code_length_max": max(code_lengths),
        })

    return metrics


def compute_paired_comparison(base_results: List[Dict], ft_results: List[Dict]) -> Dict:
    """Compute paired statistical comparison."""

    # Match by algorithm_id
    base_by_algo = {r['algorithm_id']: r for r in base_results}
    ft_by_algo = {r['algorithm_id']: r for r in ft_results}

    common_algos = set(base_by_algo.keys()) & set(ft_by_algo.keys())

    if not common_algos:
        return {"error": "No common algorithms for paired comparison"}

    # Paired PAR2 differences
    par2_diffs = []
    wins_base = 0
    wins_ft = 0
    ties = 0

    for algo_id in common_algos:
        base_par2 = base_by_algo[algo_id].get('par2')
        ft_par2 = ft_by_algo[algo_id].get('par2')

        if base_par2 is not None and ft_par2 is not None:
            diff = base_par2 - ft_par2  # Positive means fine-tuned is better
            par2_diffs.append(diff)

            if diff > 1:  # Fine-tuned better by at least 1 second
                wins_ft += 1
            elif diff < -1:  # Base better
                wins_base += 1
            else:
                ties += 1

    comparison = {
        "n_paired": len(par2_diffs),
        "wins_finetuned": wins_ft,
        "wins_base": wins_base,
        "ties": ties,
    }

    if par2_diffs:
        mean_diff = sum(par2_diffs) / len(par2_diffs)
        comparison["mean_par2_improvement"] = mean_diff

        # Simple paired t-test
        if len(par2_diffs) > 1:
            var = sum((d - mean_diff)**2 for d in par2_diffs) / (len(par2_diffs) - 1)
            se = (var / len(par2_diffs)) ** 0.5
            if se > 0:
                t_stat = mean_diff / se
                comparison["t_statistic"] = t_stat
                comparison["standard_error"] = se

    return comparison


def main():
    parser = argparse.ArgumentParser(description="Build, evaluate, and analyze benchmark solvers")
    parser.add_argument("--input", "-i", required=True, help="Input JSON from benchmark_generate_codes.py")
    parser.add_argument("--output", "-o", default=None, help="Output directory (default: same as input)")
    parser.add_argument("--cnf-file", default=DEFAULT_CNF_FILE, help="File with list of CNFs to evaluate")
    parser.add_argument("--benchmark-dir", default=DEFAULT_BENCHMARK_DIR, help="Directory containing CNF files")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout per CNF (seconds)")
    parser.add_argument("--penalty", type=int, default=DEFAULT_PENALTY, help="Penalty for timeout (PAR-2)")

    # Actions
    parser.add_argument("--build", action="store_true", help="Build all solvers")
    parser.add_argument("--evaluate", action="store_true", help="Submit evaluation jobs")
    parser.add_argument("--analyze", action="store_true", help="Collect results and compute metrics")
    parser.add_argument("--all", action="store_true", help="Run all steps (build, evaluate, analyze)")

    args = parser.parse_args()

    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.dirname(args.input)

    os.makedirs(output_dir, exist_ok=True)

    # Load generated codes
    with open(args.input, 'r') as f:
        generated_codes = json.load(f)

    print(f"Loaded {len(generated_codes)} generated codes")

    # Results file
    results_file = os.path.join(output_dir, "evaluation_results.json")

    # Build solvers
    if args.build or args.all:
        print(f"\n{'='*60}")
        print("BUILDING SOLVERS")
        print(f"{'='*60}")

        results = []

        for i, code_entry in enumerate(generated_codes, 1):
            code_hash = compute_code_hash(code_entry['generated_code'])
            model_type = code_entry['model_type']
            algo_id = code_entry['algorithm_id'][:12]

            solver_name = f"{model_type}_{algo_id}_{code_hash}"
            solver_path = os.path.join(output_dir, "solvers", solver_name)
            result_dir = os.path.join(solver_path, "results")

            print(f"\n[{i}/{len(generated_codes)}] Building {solver_name}...")

            success, error = build_solver(code_entry['generated_code'], solver_path)

            result = {
                "algorithm_id": code_entry['algorithm_id'],
                "model_type": model_type,
                "code_hash": code_hash,
                "solver_name": solver_name,
                "solver_path": solver_path,
                "result_dir": result_dir,
                "build_success": success,
                "build_error": error if not success else None,
            }
            results.append(result)

            if success:
                print(f"  SUCCESS")
            else:
                print(f"  FAILED: {error[:100]}")

        # Save intermediate results
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Summary
        successful = sum(1 for r in results if r['build_success'])
        print(f"\n{'='*60}")
        print(f"BUILD COMPLETE: {successful}/{len(results)} successful")
        print(f"{'='*60}")

    # Submit evaluation jobs
    if args.evaluate or args.all:
        print(f"\n{'='*60}")
        print("SUBMITTING EVALUATION JOBS")
        print(f"{'='*60}")

        # Load results
        with open(results_file, 'r') as f:
            results = json.load(f)

        # Submit single job array for ALL solvers (avoids job limits)
        job_id = submit_all_evaluation_jobs(
            results,
            output_dir,
            args.cnf_file,
            args.benchmark_dir,
            args.timeout
        )

        if job_id:
            # Store job ID in all results
            for result in results:
                if result['build_success']:
                    result['job_id'] = job_id

            # Save updated results
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"\n{'='*60}")
            print(f"SUBMITTED SINGLE JOB ARRAY: {job_id}")
            print(f"Wait for jobs to complete, then run --analyze")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print(f"FAILED to submit evaluation jobs")
            print(f"{'='*60}")

    # Analyze results
    if args.analyze or args.all:
        print(f"\n{'='*60}")
        print("ANALYZING RESULTS")
        print(f"{'='*60}")

        # Load results
        with open(results_file, 'r') as f:
            results = json.load(f)

        # Collect PAR2 scores
        for result in results:
            if not result['build_success']:
                continue

            print(f"\nCollecting results for {result['solver_name']}...")

            eval_results = collect_results(
                result['result_dir'],
                args.cnf_file,
                args.timeout,
                args.penalty
            )

            result.update(eval_results)

            if eval_results.get('par2'):
                print(f"  PAR2: {eval_results['par2']:.2f}, Solved: {eval_results['solved']}/{eval_results['total']}")

        # Save updated results
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Compute comprehensive metrics
        metrics = compute_metrics(results, generated_codes)

        metrics_file = os.path.join(output_dir, "benchmark_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Print summary
        print(f"\n{'='*60}")
        print("BENCHMARK RESULTS SUMMARY")
        print(f"{'='*60}")

        for model in ["base", "finetuned"]:
            m = metrics.get(model, {})
            print(f"\n{model.upper()} MODEL:")
            print(f"  Build success rate: {m.get('build_success_rate', 0)*100:.1f}%")
            if 'par2_mean' in m:
                print(f"  PAR2 (mean +/- std): {m['par2_mean']:.2f} +/- {m['par2_std']:.2f}")
                print(f"  PAR2 (min, max): ({m['par2_min']:.2f}, {m['par2_max']:.2f})")
            if 'solve_rate_mean' in m:
                print(f"  Avg solve rate: {m['solve_rate_mean']*100:.1f}%")
            if 'code_length_mean' in m:
                print(f"  Code length (mean +/- std): {m['code_length_mean']:.0f} +/- {m['code_length_std']:.0f}")
            if 'avg_pairwise_edit_distance' in m:
                print(f"  Avg pairwise edit distance: {m['avg_pairwise_edit_distance']:.0f}")

        comp = metrics.get("comparison", {})
        if comp and 'n_paired' in comp:
            print(f"\nPAIRED COMPARISON ({comp['n_paired']} pairs):")
            print(f"  Fine-tuned wins: {comp['wins_finetuned']}")
            print(f"  Base wins: {comp['wins_base']}")
            print(f"  Ties: {comp['ties']}")
            if 'mean_par2_improvement' in comp:
                print(f"  Mean PAR2 improvement: {comp['mean_par2_improvement']:.2f}s")
            if 't_statistic' in comp:
                print(f"  t-statistic: {comp['t_statistic']:.3f}")

        print(f"\nDetailed metrics saved to: {metrics_file}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()