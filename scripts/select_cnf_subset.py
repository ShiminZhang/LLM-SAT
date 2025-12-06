#!/usr/bin/env python3
"""
Select a subset of CNF instances that provide good discrimination between solvers.

This script analyzes solving logs from existing evaluations to find CNFs that:
1. Don't all solve instantly (no discrimination)
2. Don't all timeout (no discrimination)  
3. Have varied solve times across different solvers (good discrimination)

The goal is to find "medium difficulty" CNFs where solver quality matters.
"""

import os
import sys
import re
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llmsat.llmsat import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

# Penalty for timeout/error (should match evaluation settings)
DEFAULT_PENALTY = 10000


def parse_solving_time(log_path: str, penalty: float = DEFAULT_PENALTY) -> Optional[float]:
    """Parse solving time from a .solving.log file."""
    if not os.path.exists(log_path):
        return None
    
    try:
        with open(log_path, 'r') as f:
            content = f.read()
    except Exception as e:
        return penalty
    
    if not content.strip():
        return penalty
    
    # Check for timeout marker
    if "TIMEOUT" in content:
        return penalty
    
    # Look for "process-time" line
    for line in content.split('\n'):
        if 'process-time' in line:
            match = re.search(r'(\d+\.?\d*)\s+seconds', line)
            if match:
                return float(match.group(1))
    
    # Check for error indicators
    for line in content.split('\n'):
        if 'error' in line.lower() or 'CANCELLED' in line or 'TIME LIMIT' in line:
            return penalty
    
    return penalty


def find_result_directories(base_path: str = "solvers") -> List[str]:
    """Find all result directories with solving logs."""
    result_dirs = []
    
    for algo_dir in os.listdir(base_path):
        if not algo_dir.startswith("algorithm_"):
            continue
        
        algo_path = os.path.join(base_path, algo_dir)
        result_path = os.path.join(algo_path, "result")
        
        if not os.path.isdir(result_path):
            continue
        
        for code_dir in os.listdir(result_path):
            if not code_dir.startswith("code_"):
                continue
            
            code_result_path = os.path.join(result_path, code_dir)
            if os.path.isdir(code_result_path):
                # Check if it has solving logs
                logs = [f for f in os.listdir(code_result_path) if f.endswith('.solving.log')]
                if logs:
                    result_dirs.append(code_result_path)
    
    return result_dirs


def collect_solving_times(result_dirs: List[str], penalty: float = DEFAULT_PENALTY) -> Dict[str, Dict[str, float]]:
    """
    Collect solving times for all CNFs across all solvers.
    
    Returns: {cnf_name: {solver_id: solve_time}}
    """
    cnf_times: Dict[str, Dict[str, float]] = defaultdict(dict)
    
    for result_dir in result_dirs:
        solver_id = os.path.basename(result_dir)
        
        for log_file in os.listdir(result_dir):
            if not log_file.endswith('.solving.log'):
                continue
            
            cnf_name = log_file.replace('.solving.log', '')
            log_path = os.path.join(result_dir, log_file)
            
            solve_time = parse_solving_time(log_path, penalty)
            if solve_time is not None:
                cnf_times[cnf_name][solver_id] = solve_time
    
    return cnf_times


def score_cnf_discrimination(times: Dict[str, float], penalty: float = DEFAULT_PENALTY) -> Tuple[float, dict]:
    """
    Score how well a CNF discriminates between solvers.
    
    Good CNFs have:
    - Some solvers solve quickly, some take longer (high variance in non-timeout times)
    - Not all timeouts (at least some solve)
    - Not all instant solves (some challenge)
    
    Returns: (score, stats_dict)
    """
    if not times:
        return 0.0, {}
    
    values = list(times.values())
    n_solvers = len(values)
    
    # Count timeouts vs solved
    n_timeout = sum(1 for t in values if t >= penalty)
    n_solved = n_solvers - n_timeout
    
    # If all timeout or all solve instantly, low discrimination
    if n_timeout == n_solvers:
        return 0.0, {"reason": "all_timeout", "n_solvers": n_solvers}
    
    solved_times = [t for t in values if t < penalty]
    
    if not solved_times:
        return 0.0, {"reason": "no_solved", "n_solvers": n_solvers}
    
    min_time = min(solved_times)
    max_time = max(solved_times)
    mean_time = statistics.mean(solved_times)
    
    # If all solve in < 1 second, too easy
    if max_time < 1.0:
        return 0.1, {"reason": "too_easy", "max_time": max_time}
    
    # Calculate variance-based score
    if len(solved_times) > 1:
        std_time = statistics.stdev(solved_times)
        cv = std_time / mean_time if mean_time > 0 else 0  # Coefficient of variation
    else:
        std_time = 0
        cv = 0
    
    # Score components:
    # 1. Solve ratio (want ~50-80% solved, not all or none)
    solve_ratio = n_solved / n_solvers
    solve_score = 1.0 - abs(solve_ratio - 0.7) * 2  # Peak at 70% solved
    solve_score = max(0, solve_score)
    
    # 2. Time spread (want high variance in solve times)
    spread_score = min(1.0, cv)  # CV of 1+ is good
    
    # 3. Range score (want meaningful time differences)
    time_range = max_time - min_time
    range_score = min(1.0, time_range / 100)  # 100+ seconds range is good
    
    # 4. Not too hard (want some fast solves)
    fast_solves = sum(1 for t in solved_times if t < 60)
    fast_ratio = fast_solves / len(solved_times) if solved_times else 0
    fast_score = min(1.0, fast_ratio * 2)  # Want at least 50% solving in < 60s
    
    # Combined score
    score = (solve_score * 0.3 + spread_score * 0.3 + range_score * 0.2 + fast_score * 0.2)
    
    stats = {
        "n_solvers": n_solvers,
        "n_solved": n_solved,
        "n_timeout": n_timeout,
        "solve_ratio": solve_ratio,
        "min_time": min_time,
        "max_time": max_time,
        "mean_time": mean_time,
        "std_time": std_time,
        "cv": cv,
        "score": score,
    }
    
    return score, stats


def select_cnf_subset(
    cnf_times: Dict[str, Dict[str, float]],
    n_select: int = 100,
    penalty: float = DEFAULT_PENALTY,
    min_solvers: int = 2,
) -> List[Tuple[str, float, dict]]:
    """
    Select the best CNFs for discrimination.
    
    Returns: List of (cnf_name, score, stats)
    """
    scored_cnfs = []
    
    for cnf_name, times in cnf_times.items():
        if len(times) < min_solvers:
            continue
        
        score, stats = score_cnf_discrimination(times, penalty)
        scored_cnfs.append((cnf_name, score, stats))
    
    # Sort by score descending
    scored_cnfs.sort(key=lambda x: x[1], reverse=True)
    
    return scored_cnfs[:n_select]


def main():
    parser = argparse.ArgumentParser(
        description="Select CNF subset for efficient solver evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze existing results and select 100 best CNFs
  python scripts/select_cnf_subset.py --n-select 100

  # Output to specific file
  python scripts/select_cnf_subset.py --n-select 100 --output data/cnf_subset_100.txt
  
  # Show detailed stats
  python scripts/select_cnf_subset.py --n-select 50 --verbose
        """
    )
    parser.add_argument("--n-select", type=int, default=100,
                       help="Number of CNFs to select (default: 100)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for selected CNF list (default: data/cnf_subset_{n}.txt)")
    parser.add_argument("--penalty", type=float, default=DEFAULT_PENALTY,
                       help=f"Penalty value for timeouts (default: {DEFAULT_PENALTY})")
    parser.add_argument("--min-solvers", type=int, default=2,
                       help="Minimum solvers that evaluated a CNF to include it (default: 2)")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed statistics for each selected CNF")
    parser.add_argument("--solvers-path", type=str, default="solvers",
                       help="Path to solvers directory (default: solvers)")
    
    args = parser.parse_args()
    
    # Find result directories
    logger.info(f"Searching for result directories in {args.solvers_path}...")
    result_dirs = find_result_directories(args.solvers_path)
    
    if not result_dirs:
        logger.error("No result directories found with solving logs")
        sys.exit(1)
    
    logger.info(f"Found {len(result_dirs)} result directories with solving logs")
    
    # Collect solving times
    logger.info("Collecting solving times from logs...")
    cnf_times = collect_solving_times(result_dirs, args.penalty)
    
    logger.info(f"Found data for {len(cnf_times)} unique CNF instances")
    
    # Select best CNFs
    logger.info(f"Scoring and selecting top {args.n_select} CNFs...")
    selected = select_cnf_subset(
        cnf_times,
        n_select=args.n_select,
        penalty=args.penalty,
        min_solvers=args.min_solvers,
    )
    
    if not selected:
        logger.error("No CNFs met the selection criteria")
        sys.exit(1)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"Selected {len(selected)} CNFs for evaluation")
    print(f"{'='*80}\n")
    
    if args.verbose:
        for i, (cnf_name, score, stats) in enumerate(selected[:20], 1):
            print(f"{i:3d}. {cnf_name[:60]}...")
            print(f"     Score: {score:.3f} | Solved: {stats.get('n_solved', 'N/A')}/{stats.get('n_solvers', 'N/A')} | "
                  f"Time range: {stats.get('min_time', 0):.1f}s - {stats.get('max_time', 0):.1f}s")
    
    # Summary statistics
    scores = [s[1] for s in selected]
    print(f"\nScore statistics:")
    print(f"  Min: {min(scores):.3f}")
    print(f"  Max: {max(scores):.3f}")
    print(f"  Mean: {statistics.mean(scores):.3f}")
    
    # Save to file
    if args.output is None:
        args.output = f"data/cnf_subset_{args.n_select}.txt"
    
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    
    with open(args.output, 'w') as f:
        for cnf_name, score, stats in selected:
            f.write(f"{cnf_name}\n")
    
    logger.info(f"Saved selected CNF list to {args.output}")
    
    # Also save detailed stats as JSON
    stats_output = args.output.replace('.txt', '_stats.json')
    with open(stats_output, 'w') as f:
        json.dump([{"cnf": c, "score": s, "stats": st} for c, s, st in selected], f, indent=2)
    
    logger.info(f"Saved detailed stats to {stats_output}")
    
    print(f"\n✓ CNF subset saved to: {args.output}")
    print(f"✓ Use this with: --benchmarks {args.output}")


if __name__ == "__main__":
    main()