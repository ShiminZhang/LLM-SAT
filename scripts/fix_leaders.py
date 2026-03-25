#!/usr/bin/env python3
"""
Fix leaders in a generation tag by finding algorithms referenced as parents
and restoring their parent_id to None so they are recognized as leaders.

Usage:
    PYTHONPATH=src python scripts/fix_leaders.py --tag phase_decide_iter0
"""

import argparse
from llmsat.llmsat import CHATGPT_DATA_GENERATION_TABLE, Role
from llmsat.utils.aws import (
    get_ids_from_router_table,
    get_algorithm_result,
    update_algorithm_result,
    update_router_table,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    tag = args.tag
    ids = get_ids_from_router_table(CHATGPT_DATA_GENERATION_TABLE, tag)
    print(f"Found {len(ids)} algorithms in {tag}")

    # Find all unique parent IDs
    parent_ids = set()
    for i in ids:
        ar = get_algorithm_result(i)
        if ar and ar.parent_id:
            pid = ar.parent_id[0] if isinstance(ar.parent_id, list) else ar.parent_id
            parent_ids.add(pid)

    print(f"{len(parent_ids)} unique parent IDs referenced")

    # Show current state of each parent
    for pid in parent_ids:
        ar = get_algorithm_result(pid)
        if ar is None:
            print(f"  {pid[:16]}... NOT IN DB")
            continue
        print(f"  {pid[:16]}... parent_id={ar.parent_id} role={ar.role}")

    # Fix them
    fixed = 0
    for pid in parent_ids:
        ar = get_algorithm_result(pid)
        if ar is None:
            continue
        ar.parent_id = None
        ar.role = Role.LEADER
        update_algorithm_result(ar)
        if pid not in ids:
            update_router_table(CHATGPT_DATA_GENERATION_TABLE, pid, tag)
        fixed += 1
        print(f"  Fixed {pid[:16]}...")

    # Verify
    ids = get_ids_from_router_table(CHATGPT_DATA_GENERATION_TABLE, tag)
    leaders = sum(1 for i in ids if get_algorithm_result(i) and get_algorithm_result(i).parent_id is None)
    print(f"\nDone: fixed {fixed}. Now {len(ids)} total, {leaders} leaders")


if __name__ == "__main__":
    main()
