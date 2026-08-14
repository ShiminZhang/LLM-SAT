#!/usr/bin/env python3
"""Submit and collect multiple independent evaluation trials for a single algorithm/code.

Each trial writes to its own result directory so results are never overwritten.
The existing solver binary is reused across all trials (skip-build).

Usage:
    # Submit trial 1 (full eval):
    python evaluate_multi.py --algorithm-id <alg_id> --code-id <code_id> --seed 1

    # Submit trial 2 (quick eval):
    python evaluate_multi.py --algorithm-id <alg_id> --code-id <code_id> --seed 2 --quick-eval

    # Dry run:
    python evaluate_multi.py --algorithm-id <alg_id> --code-id <code_id> --seed 1 --dry-run

    # Collect results for a trial:
    python evaluate_multi.py --algorithm-id <alg_id> --code-id <code_id> --seed 1 --collect

    # Print PAR2 summary across all seeds:
    python evaluate_multi.py --algorithm-id <alg_id> --code-id <code_id> --summary
"""

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from llmsat.pipelines.evaluation_nersc import (
    EvaluationPipeline,
    QUICK_EVAL_TIMEOUT_SECONDS,
    QUICK_EVAL_WALL_TIME,
    QUICK_EVAL_PAR2_PENALTY,
    QUICK_EVAL_BENCHMARK_LIST,
)
from llmsat.pipelines.evaluation import (
    SLURM_TIMEOUT_SECONDS,
    PAR2_PENALTY,
    SAT2025_BENCHMARK_PATH,
)
from llmsat.utils.aws import get_algorithm_result, get_code_result
from llmsat.utils.paths import get_solver_dir


TRIALS_BASE = "solvers/multi_eval"


def result_dir(alg_id, code_id, seed, quick_eval):
    mode = "quick" if quick_eval else "full"
    return os.path.join(TRIALS_BASE, f"seed{seed}_{mode}", f"algorithm_{alg_id}", f"result_code_{code_id}")


def summary_path(alg_id, code_id):
    return os.path.join(TRIALS_BASE, f"algorithm_{alg_id}", f"summary_{code_id[:16]}.json")


def find_solver_binary(alg_id, code_id):
    """Find the existing kissat binary by globbing solver directories."""
    candidates = []
    # With generation tag:  solvers/<tag>/<role>/algorithm_<id>/code_<id>/kissat
    candidates += glob.glob(os.path.join("solvers", "*", "*", f"algorithm_{alg_id}*", f"code_{code_id}*", "kissat"))
    # Flat (no generation tag):  solvers/algorithm_<id>/code_<id>/kissat
    candidates += glob.glob(os.path.join("solvers", f"algorithm_{alg_id}*", f"code_{code_id}*", "kissat"))
    # Prefer original (non-multi_eval) builds
    originals = [m for m in candidates if "multi_eval" not in m]
    return originals[0] if originals else (candidates[0] if candidates else None)


def find_solver_dir(alg_id, code_id):
    binary = find_solver_binary(alg_id, code_id)
    return os.path.dirname(binary) if binary else None


def parse_solving_time(log_path, penalty):
    try:
        lines = open(log_path).readlines()
    except Exception:
        return penalty
    if not lines:
        return penalty
    for line in reversed(lines):
        if "process-time" in line:
            m = re.search(r"(\d*\.?\d+)\s+seconds", line)
            if m:
                return float(m.group(1))
        if "error" in line.lower():
            return penalty
        if "TIMEOUT" in line or "timeout" in line.lower():
            return penalty
        if "CANCELLED" in line or "TIME LIMIT" in line:
            return penalty
    return penalty


def collect_par2(rdir, penalty, expected):
    logs = glob.glob(os.path.join(rdir, "*.solving.log"))
    if not logs:
        return None, 0
    times = {os.path.basename(p): parse_solving_time(p, penalty) for p in logs}
    par2 = sum(times.values()) / len(times)
    return par2, len(times)


def do_submit(alg_id, code_id, seed, quick_eval, dry_run):
    pipeline = EvaluationPipeline()

    if quick_eval:
        if not os.path.exists(QUICK_EVAL_BENCHMARK_LIST):
            print(f"ERROR: quick-eval list not found: {QUICK_EVAL_BENCHMARK_LIST}")
            sys.exit(1)
        with open(QUICK_EVAL_BENCHMARK_LIST) as f:
            cnf_files = [l.strip() for l in f if l.strip()]
        pipeline.timeout = QUICK_EVAL_TIMEOUT_SECONDS
        pipeline.wall_time = QUICK_EVAL_WALL_TIME
        pipeline.par2_penalty = QUICK_EVAL_PAR2_PENALTY
        pipeline.cnf_files = cnf_files
        expected = len(cnf_files)
        mode = "quick-eval"
    else:
        cnf_files = None
        expected = None
        mode = "full-eval"

    # Find solver binary
    sdir = find_solver_dir(alg_id, code_id)
    if sdir is None:
        print(f"ERROR: kissat binary not found for alg={alg_id[:16]} code={code_id[:16]}")
        print("       Build it first without --seed, or check the solvers/ directory.")
        sys.exit(1)

    rdir = result_dir(alg_id, code_id, seed, quick_eval)
    os.makedirs(rdir, exist_ok=True)

    print(f"Algorithm : {alg_id}")
    print(f"Code      : {code_id}")
    print(f"Seed      : {seed}  ({mode})")
    print(f"Solver    : {sdir}")
    print(f"Result dir: {rdir}")
    if expected:
        print(f"CNFs      : {expected}")

    slurm_ids = pipeline.slurm_run_evaluate(
        solver_path=sdir,
        benchmark_path=SAT2025_BENCHMARK_PATH,
        result_dir=rdir,
        dry_run=dry_run,
        cnf_files=cnf_files,
    )

    if dry_run:
        print("\n[DRY-RUN] Not submitted.")
        return

    if slurm_ids:
        print(f"\nSubmitted SLURM jobs: {slurm_ids}")
        qflag = " --quick-eval" if quick_eval else ""
        print(f"\nAfter jobs finish, collect with:")
        print(f"  python evaluate_multi.py --algorithm-id {alg_id[:16]} --code-id {code_id[:16]} --seed {seed}{qflag} --collect")


def do_collect(alg_id, code_id, seed, quick_eval):
    penalty = QUICK_PAR2_PENALTY if quick_eval else PAR2_PENALTY
    rdir = result_dir(alg_id, code_id, seed, quick_eval)
    mode = "quick" if quick_eval else "full"

    par2, n = collect_par2(rdir, penalty, None)

    print(f"\nTrial seed={seed} ({mode})")
    print(f"  result_dir : {rdir}")
    print(f"  logs found : {n}")
    print(f"  PAR2       : {par2:.4f}" if par2 is not None else "  PAR2       : pending (no logs)")

    if par2 is None:
        return

    # Save to summary
    spath = summary_path(alg_id, code_id)
    os.makedirs(os.path.dirname(spath), exist_ok=True)
    try:
        with open(spath) as f:
            data = json.load(f)
    except Exception:
        data = {}

    key = f"seed{seed}_{mode}"
    data[key] = {"par2": par2, "n_logs": n, "result_dir": rdir}
    with open(spath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved to   : {spath}")


def do_summary(alg_id, code_id):
    spath = summary_path(alg_id, code_id)
    if not os.path.exists(spath):
        print(f"No summary found at {spath}")
        print("Run --collect for each seed first.")
        return

    with open(spath) as f:
        data = json.load(f)

    print(f"\nPAR2 Summary — alg={alg_id[:16]}  code={code_id[:16]}")
    print(f"  {'trial':<20}  {'PAR2':>10}  {'n_logs':>8}")
    print("  " + "-" * 42)
    par2_vals = []
    for key in sorted(data):
        entry = data[key]
        print(f"  {key:<20}  {entry['par2']:>10.4f}  {entry['n_logs']:>8}")
        par2_vals.append(entry["par2"])
    if len(par2_vals) > 1:
        mean = sum(par2_vals) / len(par2_vals)
        vari = sum((x - mean) ** 2 for x in par2_vals) / len(par2_vals)
        print("  " + "-" * 42)
        print(f"  {'mean':<20}  {mean:>10.4f}")
        print(f"  {'std':<20}  {vari**0.5:>10.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-trial evaluation for a single algorithm/code"
    )
    parser.add_argument("--algorithm-id", "-a", required=True)
    parser.add_argument("--code-id", "-c", required=True)
    parser.add_argument("--seed", type=int, default=None,
                        help="Trial index (required for submit/collect)")
    parser.add_argument("--quick-eval", action="store_true")
    parser.add_argument("--dry-run", "-n", action="store_true")
    parser.add_argument("--collect", action="store_true",
                        help="Collect results for this seed")
    parser.add_argument("--summary", action="store_true",
                        help="Print PAR2 across all collected seeds")
    args = parser.parse_args()

    alg_id  = args.algorithm_id
    code_id = args.code_id

    if args.summary:
        do_summary(alg_id, code_id)
        return

    if args.seed is None:
        parser.error("--seed is required for submit and collect")

    if args.collect:
        do_collect(alg_id, code_id, args.seed, args.quick_eval)
    else:
        do_submit(alg_id, code_id, args.seed, args.quick_eval, args.dry_run)


if __name__ == "__main__":
    main()
