#!/usr/bin/env python3
"""
Generate DPO training data from evaluated codes.

Creates preference pairs: <algorithm_description, preferred_code, rejected_code>
where preferred_code has lower PAR2 than rejected_code.

Usage:
  python scripts/generate_dpo_data.py --tag dpo1 --output data/dpo_training_data.json
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llmsat.llmsat import (
    CHATGPT_DATA_GENERATION_TABLE,
    setup_logging,
    get_logger,
)
from llmsat.utils.aws import (
    get_ids_from_router_table,
    get_algorithm_result,
    get_code_result,
)

setup_logging()
logger = get_logger(__name__)

# Configuration
MIN_PAR2_DIFF = 5.0  # Minimum PAR2 difference to create a preference pair
PENALTY_PAR2 = 300.0  # PAR2 value that indicates solver solved 0 instances


def extract_algorithm_description(raw_algorithm: str) -> Optional[str]:
    """Extract the actual algorithm description from the raw database field.

    The field may contain:
    1. The full ChatGPT batch API response JSON (nested structure)
    2. Just the algorithm JSON with name/algorithm fields
    3. Plain text description
    """
    if not raw_algorithm:
        return None

    def parse_algorithm_json(text: str) -> Optional[str]:
        """Parse algorithm JSON that may be wrapped in markdown code blocks."""
        # Strip markdown code block formatting
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            algo_json = json.loads(text)
            name = algo_json.get("name", "")
            algorithm = algo_json.get("algorithm", "")
            return f"Name: {name}\n\nAlgorithm: {algorithm}"
        except json.JSONDecodeError:
            return None

    try:
        # Try to parse as JSON
        data = json.loads(raw_algorithm)

        # Case 1: Full batch API response
        if "response" in data and "body" in data.get("response", {}):
            body = data["response"]["body"]
            if "output" in body and len(body["output"]) > 0:
                content = body["output"][0].get("content", [])
                if len(content) > 0:
                    text = content[0].get("text", "")
                    # Try to parse as algorithm JSON
                    result = parse_algorithm_json(text)
                    if result:
                        return result
                    return text

        # Case 2: Direct algorithm JSON with name/algorithm fields
        if "name" in data and "algorithm" in data:
            return f"Name: {data['name']}\n\nAlgorithm: {data['algorithm']}"

        # Case 3: Some other JSON structure, return as-is
        return raw_algorithm

    except json.JSONDecodeError:
        # Not JSON, try parsing as markdown-wrapped algorithm JSON
        result = parse_algorithm_json(raw_algorithm)
        if result:
            return result
        return raw_algorithm


@dataclass
class CodeInfo:
    code_id: str
    code: str
    par2: float


def get_valid_codes_for_algorithm(algorithm_id: str, algorithm_result) -> List[CodeInfo]:
    """Get all valid codes for an algorithm (excluding those with penalty PAR2)."""
    valid_codes = []

    code_ids = algorithm_result.code_id_list or []

    for code_id in code_ids:
        code_result = get_code_result(code_id)
        if code_result is None:
            continue

        # Skip codes without PAR2 scores
        if code_result.par2 is None:
            continue

        # Skip codes that received the penalty (solved 0 instances)
        if code_result.par2 >= PENALTY_PAR2:
            continue

        valid_codes.append(CodeInfo(
            code_id=code_id,
            code=code_result.code,
            par2=code_result.par2
        ))

    return valid_codes


def generate_pairs_for_algorithm(
    algorithm_description: str,
    codes: List[CodeInfo],
    min_par2_diff: float
) -> List[Dict]:
    """Generate all valid preference pairs for an algorithm."""
    pairs = []

    # Sort codes by PAR2 (ascending - lower is better)
    sorted_codes = sorted(codes, key=lambda x: x.par2)

    # Generate all pairs
    for i, preferred in enumerate(sorted_codes):
        for rejected in sorted_codes[i+1:]:
            par2_diff = rejected.par2 - preferred.par2

            # Only create pair if difference meets threshold
            if par2_diff >= min_par2_diff:
                pairs.append({
                    "algorithm_description": algorithm_description,
                    "preferred_code": preferred.code,
                    "rejected_code": rejected.code,
                    "par2_preferred": preferred.par2,
                    "par2_rejected": rejected.par2
                })

    return pairs


def generate_dpo_data(
    generation_tag: str,
    min_par2_diff: float = MIN_PAR2_DIFF,
    verbose: bool = False
) -> List[Dict]:
    """Generate DPO training data from all algorithms in a generation tag."""

    algorithm_ids = get_ids_from_router_table(CHATGPT_DATA_GENERATION_TABLE, generation_tag)

    if not algorithm_ids:
        logger.error(f"No algorithms found for tag: {generation_tag}")
        return []

    logger.info(f"Found {len(algorithm_ids)} algorithms for tag '{generation_tag}'")

    all_pairs = []
    stats = {
        "total_algorithms": len(algorithm_ids),
        "algorithms_with_valid_codes": 0,
        "total_valid_codes": 0,
        "total_pairs": 0,
        "codes_excluded_penalty": 0,
        "codes_excluded_no_par2": 0,
    }

    for algo_idx, algorithm_id in enumerate(algorithm_ids, 1):
        algorithm_result = get_algorithm_result(algorithm_id)
        if algorithm_result is None:
            logger.warning(f"Algorithm {algorithm_id} not found")
            continue

        # Get algorithm description (extract from nested JSON if needed)
        algorithm_description = extract_algorithm_description(algorithm_result.algorithm)
        if not algorithm_description:
            logger.warning(f"Algorithm {algorithm_id} has no description")
            continue

        # Get valid codes
        valid_codes = get_valid_codes_for_algorithm(algorithm_id, algorithm_result)

        if len(valid_codes) < 2:
            if verbose:
                logger.info(f"Algorithm {algo_idx}/{len(algorithm_ids)}: {len(valid_codes)} valid codes (need >= 2 for pairs)")
            continue

        stats["algorithms_with_valid_codes"] += 1
        stats["total_valid_codes"] += len(valid_codes)

        # Generate pairs
        pairs = generate_pairs_for_algorithm(
            algorithm_description=algorithm_description,
            codes=valid_codes,
            min_par2_diff=min_par2_diff
        )

        all_pairs.extend(pairs)
        stats["total_pairs"] += len(pairs)

        if verbose:
            logger.info(f"Algorithm {algo_idx}/{len(algorithm_ids)}: {len(valid_codes)} valid codes -> {len(pairs)} pairs")

    logger.info(f"\n{'='*60}")
    logger.info("DPO Data Generation Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Generation tag: {generation_tag}")
    logger.info(f"Min PAR2 difference: {min_par2_diff}")
    logger.info(f"Total algorithms: {stats['total_algorithms']}")
    logger.info(f"Algorithms with >= 2 valid codes: {stats['algorithms_with_valid_codes']}")
    logger.info(f"Total valid codes: {stats['total_valid_codes']}")
    logger.info(f"Total preference pairs: {stats['total_pairs']}")
    logger.info(f"{'='*60}\n")

    return all_pairs


def main():
    parser = argparse.ArgumentParser(
        description="Generate DPO training data from evaluated codes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate DPO data for dpo1 tag
  python scripts/generate_dpo_data.py --tag dpo1 --output data/dpo_training_data.json

  # Use different PAR2 difference threshold
  python scripts/generate_dpo_data.py --tag dpo1 --min-diff 10 --output data/dpo_data.json

  # Verbose output
  python scripts/generate_dpo_data.py --tag dpo1 --output data/dpo_data.json --verbose
        """
    )
    parser.add_argument("--tag", "-t", required=True, help="Generation tag")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file")
    parser.add_argument("--min-diff", type=float, default=MIN_PAR2_DIFF,
                       help=f"Minimum PAR2 difference for preference pairs (default: {MIN_PAR2_DIFF})")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Generate data
    pairs = generate_dpo_data(
        generation_tag=args.tag,
        min_par2_diff=args.min_diff,
        verbose=args.verbose
    )

    if not pairs:
        logger.error("No pairs generated!")
        sys.exit(1)

    # Save to file
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)

    with open(args.output, 'w') as f:
        json.dump(pairs, f, indent=2)

    logger.info(f"Saved {len(pairs)} preference pairs to {args.output}")

    # Print sample
    print(f"\nSample pair:")
    print(f"  Algorithm (truncated): {pairs[0]['algorithm_description'][:200]}...")
    print(f"  Preferred PAR2: {pairs[0]['par2_preferred']:.2f}")
    print(f"  Rejected PAR2: {pairs[0]['par2_rejected']:.2f}")
    print(f"  PAR2 difference: {pairs[0]['par2_rejected'] - pairs[0]['par2_preferred']:.2f}")


if __name__ == "__main__":
    main()