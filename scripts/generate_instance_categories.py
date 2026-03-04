#!/usr/bin/env python3
"""
Generate instance_categories.json from baseline solver results.

Parses baseline .solving.log files for SAT/UNSAT determination and
reads baseline_solving_times.json for difficulty (baseline_time).

Usage:
    python scripts/generate_instance_categories.py
"""

import json
import os
import sys

RESULT_DIRS = [
    "solvers/baseline/result",
    "solvers/base/result",
    "solvers/base/result_quick",
]
BASELINE_SOLVING_TIMES = "data/results/baseline/baseline_solving_times.json"
OUTPUT_PATH = "data/benchmarks/instance_categories.json"


def parse_satisfiability(log_path: str) -> str:
    """Parse the first line of a solving log for SAT/UNSAT."""
    try:
        with open(log_path, "r") as f:
            first_line = f.readline().strip()
    except Exception:
        return "UNKNOWN"

    if first_line.startswith("s SATISFIABLE"):
        return "SAT"
    elif first_line.startswith("s UNSATISFIABLE"):
        return "UNSAT"
    else:
        return "UNKNOWN"


def main():
    if not os.path.exists(BASELINE_SOLVING_TIMES):
        print(f"ERROR: {BASELINE_SOLVING_TIMES} not found.")
        sys.exit(1)
    with open(BASELINE_SOLVING_TIMES) as f:
        solving_times = json.load(f)

    # Build SAT/UNSAT map from solver logs (check multiple result dirs).
    # Log filenames: {instance_name}.cnf.solving.log or {instance_name}.normalised.cnf.solving.log
    # solving_times keys: {instance_name} (no .cnf)
    sat_map = {}
    for result_dir in RESULT_DIRS:
        if not os.path.isdir(result_dir):
            print(f"Skipping missing directory: {result_dir}")
            continue
        for filename in os.listdir(result_dir):
            if not filename.endswith(".solving.log"):
                continue
            # Strip .solving.log, then strip .cnf or .normalised.cnf to get instance name
            base = filename.removesuffix(".solving.log")
            if base.endswith(".normalised.cnf"):
                instance_name = base.removesuffix(".normalised.cnf")
            elif base.endswith(".cnf"):
                instance_name = base.removesuffix(".cnf")
            else:
                instance_name = base

            # Don't overwrite a known SAT/UNSAT with UNKNOWN from another dir
            if instance_name in sat_map and sat_map[instance_name] != "UNKNOWN":
                continue

            log_path = os.path.join(result_dir, filename)
            sat_map[instance_name] = parse_satisfiability(log_path)

    # Build categories for all instances in solving_times
    categories = {}
    sat_count = unsat_count = unknown_count = 0
    for instance_name, time_sec in solving_times.items():
        satisfiability = sat_map.get(instance_name, "UNKNOWN")
        categories[instance_name] = {
            "satisfiability": satisfiability,
            "baseline_time": time_sec,
        }
        if satisfiability == "SAT":
            sat_count += 1
        elif satisfiability == "UNSAT":
            unsat_count += 1
        else:
            unknown_count += 1

    print(f"Total instances: {len(categories)}")
    print(f"  SAT:     {sat_count}")
    print(f"  UNSAT:   {unsat_count}")
    print(f"  UNKNOWN: {unknown_count}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(categories, f, indent=2)

    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
