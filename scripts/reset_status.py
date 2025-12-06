#!/usr/bin/env python
"""Reset algorithm and code result statuses to allow re-evaluation or re-coding."""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llmsat.utils.aws import connect_to_db, get_ids_from_router_table, get_algorithm_result
from llmsat.llmsat import CHATGPT_DATA_GENERATION_TABLE, setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

def show_status():
    """Show current status counts."""
    conn = connect_to_db()
    cur = conn.cursor()

    cur.execute("SELECT status, COUNT(*) FROM algorithm_results GROUP BY status")
    print("\nCurrent algorithm statuses:", cur.fetchall())

    cur.execute("SELECT status, COUNT(*) FROM code_results GROUP BY status")
    print("Current code statuses:", cur.fetchall())
    print()

    cur.close()
    conn.close()

def reset_all_for_reevaluation():
    """Reset all algorithms/codes to allow re-evaluation (original behavior)."""
    conn = connect_to_db()
    cur = conn.cursor()

    # Reset all evaluated/evaluating algorithms back to code_generated
    cur.execute("UPDATE algorithm_results SET status = 'code_generated' WHERE status IN ('evaluated', 'evaluating')")
    algo_count = cur.rowcount
    conn.commit()
    print(f"Reset {algo_count} algorithms to code_generated")

    # Also reset code results to generated so they get re-evaluated
    cur.execute("UPDATE code_results SET status = 'generated', build_success = NULL WHERE status IN ('evaluated', 'evaluating', 'build_failed')")
    code_count = cur.rowcount
    conn.commit()
    print(f"Reset {code_count} code results to generated")

    cur.close()
    conn.close()

def reset_single_algorithm_for_recoding(algorithm_id):
    """Reset one algorithm to 'generated' status for code regeneration with coder.py."""
    logger.info(f"Resetting algorithm {algorithm_id[:16]}... for recoding")

    algo = get_algorithm_result(algorithm_id)
    if not algo:
        logger.error(f"Algorithm {algorithm_id} not found")
        return False

    logger.info(f"Current status: {algo.status}")
    logger.info(f"Current codes: {len(algo.code_id_list or [])}")

    conn = connect_to_db()
    cur = conn.cursor()

    # Delete all codes for this algorithm
    if algo.code_id_list:
        code_ids_tuple = tuple(algo.code_id_list)
        if len(code_ids_tuple) == 1:
            cur.execute("DELETE FROM code_results WHERE id = %s", (code_ids_tuple[0],))
        else:
            cur.execute(f"DELETE FROM code_results WHERE id IN %s", (code_ids_tuple,))
        deleted_count = cur.rowcount
        logger.info(f"Deleted {deleted_count} codes")

    # Reset algorithm to 'generated' status with empty code list
    cur.execute("""
        UPDATE algorithm_results
        SET status = 'generated', code_id_list = '[]', par2 = NULL, error_rate = NULL
        WHERE id = %s
    """, (algorithm_id,))
    conn.commit()

    cur.close()
    conn.close()

    logger.info(f"✓ Algorithm reset to 'generated' status")
    logger.info(f"✓ Ready for code generation with coder.py")
    return True

def reset_tag_for_reevaluation(generation_tag):
    """Reset all algorithms in a tag to allow re-evaluation."""
    logger.info(f"Resetting algorithms from tag: {generation_tag}")

    algorithm_ids = get_ids_from_router_table(CHATGPT_DATA_GENERATION_TABLE, generation_tag)
    if not algorithm_ids:
        logger.error(f"No algorithms found for tag: {generation_tag}")
        return False

    logger.info(f"Found {len(algorithm_ids)} algorithms in tag")

    conn = connect_to_db()
    cur = conn.cursor()

    # Reset algorithms
    algo_ids_tuple = tuple(algorithm_ids)
    if len(algo_ids_tuple) == 1:
        cur.execute("UPDATE algorithm_results SET status = 'code_generated' WHERE id = %s AND status IN ('evaluated', 'evaluating')", (algo_ids_tuple[0],))
    else:
        cur.execute(f"UPDATE algorithm_results SET status = 'code_generated' WHERE id IN %s AND status IN ('evaluated', 'evaluating')", (algo_ids_tuple,))
    algo_count = cur.rowcount
    conn.commit()
    logger.info(f"Reset {algo_count} algorithms to code_generated")

    # Get all code IDs for these algorithms
    if len(algo_ids_tuple) == 1:
        cur.execute("SELECT id FROM code_results WHERE algorithm IN %s", ((algo_ids_tuple[0],),))
    else:
        cur.execute(f"SELECT id FROM code_results WHERE algorithm IN %s", (algo_ids_tuple,))
    code_ids = [row[0] for row in cur.fetchall()]

    if code_ids:
        code_ids_tuple = tuple(code_ids)
        if len(code_ids_tuple) == 1:
            cur.execute("UPDATE code_results SET status = 'generated', build_success = NULL WHERE id = %s AND status IN ('evaluated', 'evaluating', 'build_failed')", (code_ids_tuple[0],))
        else:
            cur.execute(f"UPDATE code_results SET status = 'generated', build_success = NULL WHERE id IN %s AND status IN ('evaluated', 'evaluating', 'build_failed')", (code_ids_tuple,))
        code_count = cur.rowcount
        conn.commit()
        logger.info(f"Reset {code_count} code results to generated")

    cur.close()
    conn.close()
    logger.info("✓ Tag reset complete")
    return True

def list_algorithms_in_tag(generation_tag, limit=5):
    """List algorithms from a tag."""
    algorithm_ids = get_ids_from_router_table(CHATGPT_DATA_GENERATION_TABLE, generation_tag)

    if not algorithm_ids:
        print(f"No algorithms found for tag: {generation_tag}")
        return

    print(f"\nFound {len(algorithm_ids)} algorithms. Showing first {limit}:")
    print("="*80)

    for i, algo_id in enumerate(algorithm_ids[:limit], 1):
        algo = get_algorithm_result(algo_id)
        if algo:
            num_codes = len(algo.code_id_list or [])
            print(f"{i}. Algorithm ID: {algo_id}")
            print(f"   Status: {algo.status}")
            print(f"   Codes: {num_codes}")
            print()

    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Reset algorithm and code statuses")
    parser.add_argument("--show", action="store_true", help="Show current status counts")
    parser.add_argument("--reset-all", action="store_true", help="Reset ALL algorithms for re-evaluation")
    parser.add_argument("--reset-tag", type=str, help="Reset specific generation tag for re-evaluation")
    parser.add_argument("--reset-algorithm", type=str, help="Reset ONE algorithm for recoding (deletes codes)")
    parser.add_argument("--list-tag", type=str, help="List algorithms in a generation tag")
    parser.add_argument("--limit", type=int, default=5, help="Limit for --list-tag (default: 5)")

    args = parser.parse_args()

    if args.show:
        show_status()
    elif args.reset_all:
        print("Resetting ALL algorithms for re-evaluation...")
        show_status()
        reset_all_for_reevaluation()
        show_status()
    elif args.reset_tag:
        show_status()
        reset_tag_for_reevaluation(args.reset_tag)
        show_status()
    elif args.reset_algorithm:
        show_status()
        success = reset_single_algorithm_for_recoding(args.reset_algorithm)
        show_status()
        if success:
            print("\n" + "="*80)
            print("Next steps:")
            print("="*80)
            print("\n1. Generate codes using coder.py:")
            print("   sbatch scripts/start_coder.sh 1")
            print("\n2. After codes generate, evaluate:")
            print(f"   python src/llmsat/pipelines/evaluation.py --algorithm_id {args.reset_algorithm}")
            print("\n3. Check results:")
            print("   python scripts/check_generation_status.py --tag <your_tag>")
            print("="*80 + "\n")
    elif args.list_tag:
        list_algorithms_in_tag(args.list_tag, args.limit)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Show current status:")
        print("  python scripts/reset_status.py --show")
        print("\n  # List algorithms in a tag:")
        print('  python scripts/reset_status.py --list-tag "chatgpt_data_generation_gpt4o_3"')
        print("\n  # Reset one algorithm for recoding:")
        print("  python scripts/reset_status.py --reset-algorithm <ALGORITHM_ID>")
        print("\n  # Reset a tag for re-evaluation:")
        print('  python scripts/reset_status.py --reset-tag "chatgpt_data_generation_gpt4o_3"')
        print("\n  # Reset ALL for re-evaluation:")
        print("  python scripts/reset_status.py --reset-all")

if __name__ == "__main__":
    main()