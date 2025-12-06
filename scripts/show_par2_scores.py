#!/usr/bin/env python3
"""
Display PAR2 scores for all codes in a generation tag.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llmsat.llmsat import (
    CHATGPT_DATA_GENERATION_TABLE,
    CodeStatus,
    setup_logging,
    get_logger
)
from llmsat.utils.aws import (
    get_ids_from_router_table,
    get_algorithm_result,
    get_code_result,
)

setup_logging()
logger = get_logger(__name__)


def show_par2_scores(generation_tag, sort_by_par2=False):
    """Display PAR2 scores for all codes in a generation tag."""

    print(f"\nFetching results for generation tag: {generation_tag}")
    print("=" * 100)

    algorithm_ids = get_ids_from_router_table(CHATGPT_DATA_GENERATION_TABLE, generation_tag)

    if not algorithm_ids:
        print(f"No algorithms found for tag: {generation_tag}")
        return

    print(f"Found {len(algorithm_ids)} algorithms\n")

    # Collect all results
    all_results = []

    for algo_idx, algo_id in enumerate(algorithm_ids, 1):
        try:
            algo = get_algorithm_result(algo_id)
            if not algo:
                continue

            print(f"Algorithm {algo_idx}/{len(algorithm_ids)}: {algo_id[:16]}...")
            print(f"  Status: {algo.status}")

            code_ids = algo.code_id_list or []
            if not code_ids:
                print(f"  No codes generated yet")
                continue

            print(f"  Codes: {len(code_ids)}")

            for code_idx, code_id in enumerate(code_ids, 1):
                code = get_code_result(code_id)
                if not code:
                    print(f"    Code {code_idx}: {code_id[:16]}... - NOT FOUND")
                    continue

                par2_str = f"{code.par2:.2f}" if code.par2 is not None else "N/A"
                build_str = "✓" if code.build_success else "✗"

                result_line = f"    Code {code_idx}: {code_id[:16]}... | Status: {code.status:15s} | Build: {build_str} | PAR2: {par2_str}"
                print(result_line)

                all_results.append({
                    'algo_id': algo_id,
                    'algo_idx': algo_idx,
                    'code_id': code_id,
                    'code_idx': code_idx,
                    'status': code.status,
                    'build_success': code.build_success,
                    'par2': code.par2,
                })

            print()

        except Exception as e:
            logger.error(f"Error processing algorithm {algo_id}: {e}")
            continue

    # Print summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    total_codes = len(all_results)
    evaluated_codes = [r for r in all_results if r['par2'] is not None]
    build_failed = [r for r in all_results if not r['build_success']]

    print(f"Total codes: {total_codes}")
    print(f"Build failures: {len(build_failed)} ({len(build_failed)/total_codes*100:.1f}%)")
    print(f"Evaluated (with PAR2): {len(evaluated_codes)} ({len(evaluated_codes)/total_codes*100:.1f}%)")

    # Try to load baseline PAR2 for comparison
    baseline_par2 = None
    baseline_json_path = "data/results/baseline/baseline_solving_times.json"
    if os.path.exists(baseline_json_path):
        try:
            import json
            with open(baseline_json_path, 'r') as f:
                baseline_times = json.load(f)
            if baseline_times:
                baseline_par2 = sum(baseline_times.values()) / len(baseline_times)
        except Exception as e:
            logger.warning(f"Could not load baseline PAR2: {e}")

    if evaluated_codes:
        par2_values = [r['par2'] for r in evaluated_codes]
        print(f"\nPAR2 Statistics:")
        if baseline_par2 is not None:
            print(f"  Baseline:       {baseline_par2:.2f}")
        print(f"  Best (lowest):  {min(par2_values):.2f}")
        print(f"  Worst (highest): {max(par2_values):.2f}")
        print(f"  Average:        {sum(par2_values)/len(par2_values):.2f}")
        
        if baseline_par2 is not None:
            better_than_baseline = len([p for p in par2_values if p < baseline_par2])
            print(f"\n  Better than baseline: {better_than_baseline}/{len(par2_values)} "
                  f"({better_than_baseline/len(par2_values)*100:.1f}%)")

        if sort_by_par2:
            print(f"\n{'─'*100}")
            print("TOP 10 BEST CODES (by PAR2 score):")
            if baseline_par2 is not None:
                print(f"(Baseline PAR2: {baseline_par2:.2f})")
            print(f"{'─'*100}")

            sorted_results = sorted(evaluated_codes, key=lambda x: x['par2'])
            for i, r in enumerate(sorted_results[:10], 1):
                vs_baseline = ""
                if baseline_par2 is not None:
                    diff = r['par2'] - baseline_par2
                    vs_baseline = f" ({diff:+.2f} vs baseline)"
                print(f"{i:2d}. Algo {r['algo_idx']:3d} Code {r['code_idx']:2d} | "
                      f"PAR2: {r['par2']:8.2f}{vs_baseline} | "
                      f"Code ID: {r['code_id'][:16]}...")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Show PAR2 scores for generated codes")
    parser.add_argument("--tag", type=str, required=True, help="Generation tag to check")
    parser.add_argument("--sort", action="store_true", help="Sort by PAR2 and show top 10")
    args = parser.parse_args()

    try:
        show_par2_scores(args.tag, sort_by_par2=args.sort)
    except KeyError as e:
        if "DB_PASS" in str(e):
            print("\nError: DB_PASS environment variable not set!")
            print("Run: export DB_PASS=\"Damn123,\"")
            print("Or: source export_aws_db_pw.sh")
            sys.exit(1)
        else:
            raise
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()