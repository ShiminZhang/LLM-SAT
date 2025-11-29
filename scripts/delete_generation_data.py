#!/usr/bin/env python3
"""
Delete algorithms, codes, and router table entries for a specific generation tag.
Use with caution - this permanently deletes data from the database.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llmsat.llmsat import (
    CHATGPT_DATA_GENERATION_TABLE,
    setup_logging,
    get_logger
)
from llmsat.utils.aws import (
    get_ids_from_router_table,
    get_algorithm_result,
    remove_algorithm_result,
    remove_code_result,
    connect_to_db,
    clear_tables,
    clear_router_table,
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


def delete_router_entries_by_tag(generation_tag: str):
    """Delete all router table entries for a generation tag."""
    conn = connect_to_db()
    cur = conn.cursor()
    cur.execute(
        f"DELETE FROM {CHATGPT_DATA_GENERATION_TABLE} WHERE type = %s;",
        (generation_tag,)
    )
    deleted_count = cur.rowcount
    conn.commit()
    logger.info(f"Deleted {deleted_count} router table entries for tag '{generation_tag}'")
    return deleted_count


def delete_generation_data(generation_tag: str, dry_run: bool = True):
    """
    Delete all data associated with a generation tag:
    1. All algorithms for this tag
    2. All codes associated with those algorithms
    3. Router table entries for this tag
    """

    print(f"\n{'='*80}")
    print(f"{'DRY RUN - ' if dry_run else ''}Deleting data for generation tag: {generation_tag}")
    print(f"{'='*80}\n")

    # Get all algorithm IDs for this tag
    algorithm_ids = get_ids_from_router_table(CHATGPT_DATA_GENERATION_TABLE, generation_tag)

    if not algorithm_ids:
        print(f"No algorithms found for tag: {generation_tag}")
        return

    print(f"Found {len(algorithm_ids)} algorithms to delete")

    # Collect all code IDs
    all_code_ids = []
    for algo_id in algorithm_ids:
        algo = get_algorithm_result(algo_id)
        if algo and algo.code_id_list:
            all_code_ids.extend(algo.code_id_list)

    print(f"Found {len(all_code_ids)} codes to delete")

    if dry_run:
        print("\n--- DRY RUN ---")
        print(f"Would delete:")
        print(f"  - {len(algorithm_ids)} algorithms")
        print(f"  - {len(all_code_ids)} codes")
        print(f"  - Router table entries for tag '{generation_tag}'")
        print(f"\nRun with --confirm to actually delete this data")
        return

    # Confirm before deletion
    print(f"\nThis will permanently delete:")
    print(f"  - {len(algorithm_ids)} algorithms")
    print(f"  - {len(all_code_ids)} codes")
    print(f"  - Router table entries for tag '{generation_tag}'")

    # Actually delete
    print("\nDeleting codes...")
    for i, code_id in enumerate(all_code_ids, 1):
        if i % 10 == 0 or i == len(all_code_ids):
            print(f"  Deleted {i}/{len(all_code_ids)} codes", end='\r')
        remove_code_result(code_id)
    print(f"  Deleted {len(all_code_ids)}/{len(all_code_ids)} codes ✓")

    print("\nDeleting algorithms...")
    for i, algo_id in enumerate(algorithm_ids, 1):
        if i % 5 == 0 or i == len(algorithm_ids):
            print(f"  Deleted {i}/{len(algorithm_ids)} algorithms", end='\r')
        remove_algorithm_result(algo_id)
    print(f"  Deleted {len(algorithm_ids)}/{len(algorithm_ids)} algorithms ✓")

    print("\nDeleting router table entries...")
    deleted_router = delete_router_entries_by_tag(generation_tag)
    print(f"  Deleted {deleted_router} router entries ✓")

    print(f"\n{'='*80}")
    print(f"Successfully deleted all data for tag: {generation_tag}")
    print(f"{'='*80}\n")


def delete_all_data(dry_run: bool = True):
    """Delete ALL data from algorithm_results, code_results, and router table."""

    if dry_run:
        print("\n--- DRY RUN ---")
        print("Would delete ALL data from:")
        print("  - algorithm_results table")
        print("  - code_results table")
        print(f"  - {CHATGPT_DATA_GENERATION_TABLE} router table")
        print("\nRun with --confirm to actually delete ALL data")
        return

    print(f"\n{'='*80}")
    print("⚠️  WARNING: This will delete ALL data from all tables!")
    print(f"{'='*80}\n")

    print("Deleting all data...")
    clear_tables()  # Clears algorithm_results and code_results
    clear_router_table(CHATGPT_DATA_GENERATION_TABLE)  # Clears router table

    print("✓ All data deleted successfully")


def list_generation_tags():
    """List all generation tags and their algorithm counts."""
    tags = get_all_generation_tags()

    if not tags:
        print("\nNo generation tags found in database.")
        return

    print(f"\n{'='*80}")
    print(f"Generation Tags in Database")
    print(f"{'='*80}\n")

    for tag in tags:
        algorithm_ids = get_ids_from_router_table(CHATGPT_DATA_GENERATION_TABLE, tag)
        print(f"  {tag}: {len(algorithm_ids)} algorithms")

    print(f"\n{'='*80}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Delete generation data from database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all generation tags
  python scripts/delete_generation_data.py --list

  # Dry run - see what would be deleted for a specific tag
  python scripts/delete_generation_data.py --tag kissatmab_experiment1

  # Actually delete data for a specific tag
  python scripts/delete_generation_data.py --tag kissatmab_experiment1 --confirm

  # Delete ALL data (dry run)
  python scripts/delete_generation_data.py --all

  # Delete ALL data (actually delete)
  python scripts/delete_generation_data.py --all --confirm
        """
    )
    parser.add_argument("--tag", type=str, help="Generation tag to delete")
    parser.add_argument("--all", action="store_true", help="Delete ALL data from all tables")
    parser.add_argument("--list", action="store_true", help="List all generation tags")
    parser.add_argument("--confirm", action="store_true", help="Actually perform deletion (otherwise dry run)")
    args = parser.parse_args()

    try:
        if args.list:
            list_generation_tags()
        elif args.all:
            delete_all_data(dry_run=not args.confirm)
        elif args.tag:
            delete_generation_data(args.tag, dry_run=not args.confirm)
        else:
            parser.print_help()
            print("\n⚠️  No action specified. Use --list, --tag, or --all")

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
