#!/usr/bin/env python3
"""
Check the status of algorithms in the CHATGPT_DATA_GENERATION_TABLE.
Shows which algorithms are ready for evaluation and their current status.
"""

import sys
import os
from collections import defaultdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llmsat.llmsat import (
    CHATGPT_DATA_GENERATION_TABLE,
    AlgorithmStatus,
    CodeStatus,
    setup_logging,
    get_logger
)
from llmsat.utils.aws import (
    get_ids_from_router_table,
    get_algorithm_result,
    get_code_result,
    connect_to_db
)

setup_logging()
logger = get_logger(__name__)

def get_all_generation_tags():
    """Get all unique generation tags from the router table."""
    conn = connect_to_db()
    cur = conn.cursor()
    cur.execute(f"SELECT DISTINCT type FROM {CHATGPT_DATA_GENERATION_TABLE};")
    rows = cur.fetchall()
    return [row[0] for row in rows if row[0]]

def analyze_generation_tag(generation_tag, verbose=False):
    """Analyze algorithms for a specific generation tag."""
    if verbose:
        print(f"  Fetching algorithm IDs for tag '{generation_tag}'...", flush=True)

    algorithm_ids = get_ids_from_router_table(CHATGPT_DATA_GENERATION_TABLE, generation_tag)

    if not algorithm_ids:
        return None

    if verbose:
        print(f"  Found {len(algorithm_ids)} algorithms, analyzing...", flush=True)

    status_counts = defaultdict(int)
    code_status_counts = defaultdict(int)
    ready_for_eval = []
    evaluating = []
    evaluated = []
    build_failed_codes = 0
    total_codes = 0

    for i, algo_id in enumerate(algorithm_ids):
        if verbose and i % 10 == 0:
            print(f"    Processing algorithm {i+1}/{len(algorithm_ids)}...", flush=True)

        try:
            algo = get_algorithm_result(algo_id)
            if not algo:
                continue

            status_counts[algo.status] += 1

            # Count code statuses
            code_ids = algo.code_id_list or []
            for code_id in code_ids:
                total_codes += 1
                code = get_code_result(code_id)
                if code:
                    code_status_counts[code.status] += 1
                    if code.status == CodeStatus.BuildFailed:
                        build_failed_codes += 1

            # Categorize algorithms by readiness
            if algo.status == AlgorithmStatus.Generated or algo.status == AlgorithmStatus.CodeGenerated:
                ready_for_eval.append(algo_id)
            elif algo.status == AlgorithmStatus.Evaluating:
                evaluating.append(algo_id)
            elif algo.status == AlgorithmStatus.Evaluated:
                evaluated.append(algo_id)
        except Exception as e:
            logger.warning(f"Error processing algorithm {algo_id}: {e}")
            continue

    return {
        'tag': generation_tag,
        'total_algorithms': len(algorithm_ids),
        'status_counts': dict(status_counts),
        'code_status_counts': dict(code_status_counts),
        'total_codes': total_codes,
        'build_failed_codes': build_failed_codes,
        'ready_for_eval': ready_for_eval,
        'evaluating': evaluating,
        'evaluated': evaluated,
    }

def print_summary(tag_info):
    """Print a formatted summary of the generation tag."""
    print(f"\n{'='*80}")
    print(f"Generation Tag: {tag_info['tag']}")
    print(f"{'='*80}")
    print(f"Total Algorithms: {tag_info['total_algorithms']}")
    print(f"Total Code Implementations: {tag_info['total_codes']}")
    print(f"\nAlgorithm Status Breakdown:")
    for status, count in tag_info['status_counts'].items():
        print(f"  {status}: {count}")

    if tag_info['code_status_counts']:
        print(f"\nCode Status Breakdown:")
        for status, count in tag_info['code_status_counts'].items():
            print(f"  {status}: {count}")
        if tag_info['build_failed_codes'] > 0:
            print(f"  Build Failures: {tag_info['build_failed_codes']} ({tag_info['build_failed_codes']/tag_info['total_codes']*100:.1f}%)")

    print(f"\n{'─'*80}")
    print(f"Ready for Evaluation: {len(tag_info['ready_for_eval'])} algorithms")
    print(f"Currently Evaluating:  {len(tag_info['evaluating'])} algorithms")
    print(f"Completed Evaluation:  {len(tag_info['evaluated'])} algorithms")

    if tag_info['ready_for_eval']:
        print(f"\n✓ You can run evaluation for {len(tag_info['ready_for_eval'])} algorithms")
        print(f"  Command: sbatch scripts/start_evaluation.sh")
        print(f"  (Make sure to set generation_tag='{tag_info['tag']}' in evaluation.py)")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Check CHATGPT_DATA_GENERATION_TABLE status")
    parser.add_argument("--tag", type=str, help="Specific generation tag to check")
    parser.add_argument("--all", action="store_true", help="Show all generation tags")
    parser.add_argument("--list-ready", action="store_true", help="List algorithm IDs ready for evaluation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show progress during analysis")
    args = parser.parse_args()

    try:
        if args.all:
            # Show all tags
            print("Fetching generation tags from database...", flush=True)
            tags = get_all_generation_tags()
            print(f"\nFound {len(tags)} generation tag(s) in database:")
            for tag in tags:
                tag_info = analyze_generation_tag(tag, verbose=args.verbose)
                if tag_info:
                    print_summary(tag_info)
        elif args.tag:
            # Show specific tag
            if args.verbose:
                print(f"Analyzing tag '{args.tag}'...", flush=True)
            tag_info = analyze_generation_tag(args.tag, verbose=args.verbose)
            if tag_info:
                print_summary(tag_info)
                if args.list_ready and tag_info['ready_for_eval']:
                    print(f"\nAlgorithm IDs ready for evaluation:")
                    for algo_id in tag_info['ready_for_eval'][:10]:  # Show first 10
                        print(f"  {algo_id}")
                    if len(tag_info['ready_for_eval']) > 10:
                        print(f"  ... and {len(tag_info['ready_for_eval']) - 10} more")
            else:
                print(f"No data found for generation tag: {args.tag}")
        else:
            # Default: show all tags with summary
            print("Fetching generation tags from database...", flush=True)
            tags = get_all_generation_tags()
            print(f"\n{'='*80}")
            print(f"CHATGPT_DATA_GENERATION_TABLE Status Summary")
            print(f"{'='*80}")
            print(f"\nTotal generation tags: {len(tags)}")

            if not tags:
                print("\nNo generation tags found in database.")
                print("\nMake sure:")
                print("  1. DB_PASS environment variable is set")
                print("  2. Database is accessible")
                print("  3. You have run chatgpt_data_generation.py to generate data")
                return

            if args.verbose:
                print("\nAnalyzing tags (this may take a while)...", flush=True)

            for tag in tags:
                if args.verbose:
                    print(f"\nProcessing tag: {tag}", flush=True)
                tag_info = analyze_generation_tag(tag, verbose=args.verbose)
                if tag_info:
                    print(f"\n{tag}: {tag_info['total_algorithms']} algorithms, "
                          f"{len(tag_info['ready_for_eval'])} ready, "
                          f"{len(tag_info['evaluating'])} evaluating, "
                          f"{len(tag_info['evaluated'])} evaluated")

            print(f"\nUse --tag <TAG> for detailed breakdown")
            print(f"Use --all to see all tags in detail")
            print(f"Use --verbose/-v to see progress during analysis")

    except KeyError as e:
        if "DB_PASS" in str(e):
            print("\nError: DB_PASS environment variable not set!")
            print("Run: export DB_PASS=\"Damn123,\"")
            print("Or: source export_aws_db_pw.sh")
            sys.exit(1)
        else:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()