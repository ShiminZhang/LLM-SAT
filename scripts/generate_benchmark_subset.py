#!/usr/bin/env python3
"""
Generate a stratified 100-CNF benchmark subset from base solver solving times.

Reads solvers/base/solving_times.json, buckets CNFs by difficulty, and samples
100 CNFs proportionally from each bucket.

Usage:
    python scripts/generate_benchmark_subset.py
"""

import json
import os
import random
import sys

SOLVING_TIMES_PATH = "solvers/base/solving_times.json"
OUTPUT_PATH = "data/benchmarks/satcomp2025_quick50.txt"
SUBSET_SIZE = 50
RANDOM_SEED = 42

# Difficulty tiers based on base solver time (seconds)
TIERS = [
    ("Easy",    lambda t: t < 10),
    ("Medium",  lambda t: 10 <= t < 1000),
    ("Hard",    lambda t: 1000 <= t < 5000),
    ("Timeout", lambda t: t >= 5000),
]


def main():
    if not os.path.exists(SOLVING_TIMES_PATH):
        print(f"ERROR: {SOLVING_TIMES_PATH} not found.")
        print("Run the base solver evaluation first:")
        print("  bash scripts/run_base_solver_eval.sh")
        print("  bash scripts/run_base_solver_eval.sh --collect")
        sys.exit(1)

    with open(SOLVING_TIMES_PATH) as f:
        solving_times = json.load(f)

    print(f"Loaded {len(solving_times)} solving times from {SOLVING_TIMES_PATH}")

    # Bucket CNFs into tiers
    buckets = {name: [] for name, _ in TIERS}
    for cnf_name, time_sec in solving_times.items():
        for tier_name, tier_pred in TIERS:
            if tier_pred(time_sec):
                buckets[tier_name].append(cnf_name)
                break

    total = len(solving_times)
    print(f"\nDifficulty distribution (total: {total}):")
    for tier_name, _ in TIERS:
        count = len(buckets[tier_name])
        pct = 100.0 * count / total if total > 0 else 0
        print(f"  {tier_name:>8s}: {count:4d}  ({pct:5.1f}%)")

    # Proportional sampling
    rng = random.Random(RANDOM_SEED)
    selected = []
    remaining = SUBSET_SIZE

    # Compute proportional sizes (round down, then distribute remainder)
    tier_sizes = {}
    for tier_name, _ in TIERS:
        count = len(buckets[tier_name])
        tier_sizes[tier_name] = int(SUBSET_SIZE * count / total) if total > 0 else 0

    # Distribute remainder to largest tiers first
    allocated = sum(tier_sizes.values())
    remainder = SUBSET_SIZE - allocated
    # Sort tiers by fractional part descending for fair rounding
    fractional = []
    for tier_name, _ in TIERS:
        count = len(buckets[tier_name])
        exact = SUBSET_SIZE * count / total if total > 0 else 0
        frac = exact - int(exact)
        fractional.append((frac, tier_name))
    fractional.sort(reverse=True)
    for _, tier_name in fractional:
        if remainder <= 0:
            break
        tier_sizes[tier_name] += 1
        remainder -= 1

    print(f"\nSampling {SUBSET_SIZE} CNFs:")
    for tier_name, _ in TIERS:
        sample_size = min(tier_sizes[tier_name], len(buckets[tier_name]))
        tier_selected = rng.sample(buckets[tier_name], sample_size)
        selected.extend(tier_selected)
        print(f"  {tier_name:>8s}: {sample_size:4d} sampled from {len(buckets[tier_name])}")

    # Sort for deterministic output
    selected.sort()

    # Write output
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for cnf_name in selected:
            # Ensure .cnf extension
            if not cnf_name.endswith(".cnf"):
                cnf_name = cnf_name + ".cnf"
            f.write(cnf_name + "\n")

    print(f"\nWrote {len(selected)} CNFs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
