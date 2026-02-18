"""
inspect_generation.py

Inspect all leaders, members, and code results for a given generation tag and
produce two JSONL mapping files:

  1. leader_member_mapping.jsonl  — one record per leader, with a nested list
                                    of its members (both are natural-language
                                    algorithm descriptions).

  2. plan_code_mapping.jsonl      — one record per algorithm (leaders first,
                                    then members), with a nested list of all
                                    associated code implementations.

Usage:
    python src/llmsat/pipelines/inspect_generation.py \
        --generation_tag mike_gemini_flash_test \
        --output_dir outputs/mike_gemini_flash_test/inspection
"""

import argparse
import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from llmsat.llmsat import (
    CHATGPT_DATA_GENERATION_TABLE,
    AlgorithmResult,
    CodeResult,
    get_logger,
    setup_logging,
)
from llmsat.utils.aws import (
    get_algorithm_result,
    get_code_result,
    get_ids_from_router_table,
)

setup_logging(level=logging.INFO)
logger = get_logger(__name__)

def parse_algorithm_json(algorithm_json_str: str) -> Dict[str, str]:
    """Parse the algorithm JSON string stored in AlgorithmResult.algorithm.

    The JSON produced by the generation pipeline contains at minimum:
        - "name"      : short human-readable name
        - "algorithm" : step-by-step natural-language description
        - "reason"    : rationale / motivation

    Returns a dict with those three keys (empty strings when absent or when
    parsing fails).
    """
    try:
        spec = json.loads(algorithm_json_str)
        return {
            "name": spec.get("name", ""),
            "algorithm": spec.get("algorithm", ""),
            "reason": spec.get("reason", ""),
        }
    except (json.JSONDecodeError, AttributeError):
        return {"name": "", "algorithm": algorithm_json_str, "reason": ""}


def load_algorithms(
    generation_tag: str,
) -> Tuple[
    Dict[str, AlgorithmResult],
    Dict[str, AlgorithmResult],
    Dict[str, List[AlgorithmResult]],
]:
    """Fetch all AlgorithmResult objects for *generation_tag* from the DB.

    Returns:
        leaders           – algorithms with parent_id == None
        members           – algorithms with parent_id != None
        members_by_leader – members grouped by their leader's id
    """
    logger.info(f"Fetching algorithm IDs for generation_tag='{generation_tag}' …")
    all_ids: List[str] = get_ids_from_router_table(
        CHATGPT_DATA_GENERATION_TABLE, generation_tag
    )
    logger.info(f"  Found {len(all_ids)} algorithm ID(s).")

    leaders: Dict[str, AlgorithmResult] = {}
    members: Dict[str, AlgorithmResult] = {}

    for alg_id in all_ids:
        alg = get_algorithm_result(alg_id)
        if alg is None:
            logger.warning(f"  Algorithm {alg_id} not found in DB — skipping.")
            continue
        if alg.parent_id is None:
            leaders[alg_id] = alg
        else:
            members[alg_id] = alg

    members_by_leader: Dict[str, List[AlgorithmResult]] = defaultdict(list)
    for member in members.values():
        members_by_leader[member.parent_id].append(member)

    logger.info(
        f"  Leaders: {len(leaders)}, Members: {len(members)} "
        f"across {len(members_by_leader)} team(s)."
    )
    return leaders, members, members_by_leader


def fetch_codes(algorithm: AlgorithmResult) -> List[Dict[str, Any]]:
    """Retrieve all CodeResult objects listed in *algorithm.code_id_list*."""
    codes: List[Dict[str, Any]] = []
    for code_id in algorithm.code_id_list:
        code_result: Optional[CodeResult] = get_code_result(code_id)
        if code_result is None:
            logger.warning(f"  Code {code_id} not found in DB — skipping.")
            continue
        codes.append(
            {
                "code_id": code_result.id,
                "code": code_result.code,
                "status": code_result.status,
                "build_success": code_result.build_success,
                "par2": code_result.par2,
                "last_updated": code_result.last_updated,
            }
        )
    return codes

def build_leader_member_records(
    leaders: Dict[str, AlgorithmResult],
    members_by_leader: Dict[str, List[AlgorithmResult]],
) -> List[Dict[str, Any]]:
    """Build one record per leader for the leader-member mapping JSONL."""
    records: List[Dict[str, Any]] = []

    for leader_id, leader in leaders.items():
        leader_spec = parse_algorithm_json(leader.algorithm)
        team_members = members_by_leader.get(leader_id, [])

        member_records = []
        for member in team_members:
            member_spec = parse_algorithm_json(member.algorithm)
            member_records.append(
                {
                    "member_id": member.id,
                    "member_name": member_spec["name"],
                    "member_algorithm": member_spec["algorithm"],
                    "member_reason": member_spec["reason"],
                    "member_target_function": member.target_function,
                    "member_status": member.status,
                    "member_num_codes": len(member.code_id_list),
                }
            )

        records.append(
            {
                "leader_id": leader.id,
                "leader_name": leader_spec["name"],
                "leader_algorithm": leader_spec["algorithm"],
                "leader_reason": leader_spec["reason"],
                "leader_target_function": leader.target_function,
                "leader_status": leader.status,
                "leader_num_codes": len(leader.code_id_list),
                "num_members": len(team_members),
                "members": member_records,
            }
        )

    return records


def build_plan_code_records(
    leaders: Dict[str, AlgorithmResult],
    members: Dict[str, AlgorithmResult],
    members_by_leader: Dict[str, List[AlgorithmResult]],
) -> List[Dict[str, Any]]:
    """Build one record per algorithm for the plan-code mapping JSONL.

    Leaders are emitted first (sorted by id for determinism), followed by
    their members in the same order — keeping related algorithms adjacent.
    """
    records: List[Dict[str, Any]] = []

    def make_record(alg: AlgorithmResult, alg_type: str) -> Dict[str, Any]:
        spec = parse_algorithm_json(alg.algorithm)
        codes = fetch_codes(alg)
        return {
            "algorithm_id": alg.id,
            "algorithm_type": alg_type,
            "parent_id": alg.parent_id,
            "algorithm_name": spec["name"],
            "algorithm_text": spec["algorithm"],
            "algorithm_reason": spec["reason"],
            "target_function": alg.target_function,
            "algorithm_status": alg.status,
            "num_codes": len(codes),
            "codes": codes,
        }

    # Leaders first, then their members immediately after each leader
    for leader_id, leader in sorted(leaders.items()):
        records.append(make_record(leader, "leader"))
        for member in members_by_leader.get(leader_id, []):
            records.append(make_record(member, "member"))

    # Members whose leader is not in this generation tag (shouldn't happen,
    # but handle gracefully)
    orphan_members = [
        m for m in members.values() if m.parent_id not in leaders
    ]
    for member in orphan_members:
        logger.warning(
            f"  Member {member.id} has parent_id={member.parent_id} which is "
            "not a leader in this generation tag."
        )
        records.append(make_record(member, "member"))

    return records


def write_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    """Write *records* as a JSONL file (one JSON object per line)."""
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"  Written {len(records)} record(s) → {path}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a generation tag and output leader-member and "
            "plan-code mapping JSONL files."
        )
    )
    parser.add_argument(
        "--generation_tag",
        required=True,
        help="Generation tag to inspect (e.g. gemini_trial1).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where the two JSONL files will be written.",
    )
    args = parser.parse_args()

    generation_tag: str = args.generation_tag
    output_dir: str = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    leaders, members, members_by_leader = load_algorithms(generation_tag)

    if not leaders and not members:
        logger.error(
            f"No algorithms found for generation_tag='{generation_tag}'. "
            "Verify the tag is correct and the database is accessible."
        )
        return

    logger.info("Building leader-member mapping …")
    lm_records = build_leader_member_records(leaders, members_by_leader)
    lm_path = os.path.join(output_dir, "leader_member_mapping.jsonl")
    write_jsonl(lm_records, lm_path)

    logger.info("Building plan-code mapping (fetching code results) …")
    pc_records = build_plan_code_records(leaders, members, members_by_leader)
    pc_path = os.path.join(output_dir, "plan_code_mapping.jsonl")
    write_jsonl(pc_records, pc_path)

    total_codes = sum(r["num_codes"] for r in pc_records)
    print("\n" + "=" * 60)
    print(f"Generation tag : {generation_tag}")
    print(f"Leaders        : {len(leaders)}")
    print(f"Members        : {len(members)}")
    print(f"Total algorithms: {len(leaders) + len(members)}")
    print(f"Total codes    : {total_codes}")
    print("-" * 60)
    print(f"Leader-member mapping → {lm_path}")
    print(f"Plan-code mapping     → {pc_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
