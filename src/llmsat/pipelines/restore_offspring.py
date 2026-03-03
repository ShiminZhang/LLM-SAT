"""
Re-store offspring from saved JSON files into the DB under a new output_tag.

Use this when offspring were already generated (offspring_codes_iterN.json +
crossover_results_iterN.json exist) but you want to re-store them under a
clean tag to avoid DB pollution from previous runs.

After running this, use --evaluate_only with the new output_tag.

Usage:
    python src/llmsat/pipelines/restore_offspring.py \
        --output_dir outputs/gemini_trial5_gen1 \
        --new_output_tag gemini_trial5_gen1_v2 \
        --iteration 1

Then evaluate:
    python src/llmsat/pipelines/genetic_evolution.py \
        --folder outputs/gemini_trial5 \
        --output_tag gemini_trial5_gen1_v2 \
        --evaluate_only \
        --evaluate \
        --max_iterations 1
"""

import argparse
import json
import os
from datetime import datetime

from llmsat.llmsat import (
    CHATGPT_DATA_GENERATION_TABLE,
    AlgorithmResult,
    AlgorithmStatus,
    CodeResult,
    CodeStatus,
    NOT_INITIALIZED,
    get_id,
    get_logger,
    setup_logging,
)
from llmsat.utils.aws import (
    update_algorithm_result,
    update_code_result,
    update_router_table,
)

setup_logging()
logger = get_logger(__name__)


def restore_offspring(output_dir: str, new_output_tag: str, iteration: int = 1) -> None:
    """
    Read offspring_codes_iterN.json + crossover_results_iterN.json from output_dir
    and re-store all entries in DB under new_output_tag_iterN.
    """
    codes_path = os.path.join(output_dir, f"offspring_codes_iter{iteration}.json")
    crossover_path = os.path.join(output_dir, f"crossover_results_iter{iteration}.json")

    if not os.path.exists(codes_path):
        logger.error(f"offspring_codes file not found: {codes_path}")
        return
    if not os.path.exists(crossover_path):
        logger.error(f"crossover_results file not found: {crossover_path}")
        return

    with open(codes_path, "r") as f:
        offspring_codes: dict = json.load(f)  # algorithm_id -> code string

    with open(crossover_path, "r") as f:
        crossover_results: list = json.load(f)  # list of offspring dicts

    # Build lookup: algorithm_id -> crossover entry
    crossover_map = {entry["algorithm_id"]: entry for entry in crossover_results}

    iter_output_tag = f"{new_output_tag}_iter{iteration}"
    logger.info(f"Re-storing {len(offspring_codes)} offspring under tag: {iter_output_tag}")

    stored = 0
    skipped = 0

    for algorithm_id, code_str in offspring_codes.items():
        crossover_entry = crossover_map.get(algorithm_id)
        if crossover_entry is None:
            logger.warning(f"No crossover entry for {algorithm_id[:16]}, skipping")
            skipped += 1
            continue

        code_id = get_id(code_str)

        algo_result = AlgorithmResult(
            id=algorithm_id,
            algorithm=crossover_entry["algorithm_json"],
            status=AlgorithmStatus.CodeGenerated,
            last_updated=datetime.now(),
            prompt="genetic_evolution_crossover",
            par2=NOT_INITIALIZED,
            error_rate=NOT_INITIALIZED,
            other_metrics={
                "parent_a": crossover_entry["parent_a_id"],
                "parent_b": crossover_entry["parent_b_id"],
                "evolution_method": "causal_crossover",
            },
            code_id_list=[code_id],
            target_function=crossover_entry.get("target_function", "kissat_restarting"),
            parent_id=crossover_entry["parent_a_id"],
        )
        update_algorithm_result(algo_result)
        update_router_table(CHATGPT_DATA_GENERATION_TABLE, algorithm_id, iter_output_tag)

        code_result = CodeResult(
            id=code_id,
            algorithm_id=algorithm_id,
            code=code_str,
            status=CodeStatus.Generated,
            par2=None,
            last_updated=datetime.now(),
            build_success=NOT_INITIALIZED,
        )
        update_code_result(code_result)

        stored += 1
        logger.info(f"  Stored {algorithm_id[:16]} -> code {code_id[:16]}")

    logger.info(f"Done: {stored} stored, {skipped} skipped")
    logger.info(f"Now run --evaluate_only --output_tag {new_output_tag} --max_iterations {iteration}")


def main():
    parser = argparse.ArgumentParser(description="Re-store offspring under a new output_tag")
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Path to output dir containing offspring_codes_iterN.json and crossover_results_iterN.json "
             "(e.g. outputs/gemini_trial5_gen1)"
    )
    parser.add_argument(
        "--new_output_tag", type=str, required=True,
        help="New output tag to store offspring under (e.g. gemini_trial5_gen1_v2)"
    )
    parser.add_argument(
        "--iteration", type=int, default=1,
        help="Which iteration's files to restore (default: 1)"
    )
    args = parser.parse_args()

    restore_offspring(
        output_dir=args.output_dir,
        new_output_tag=args.new_output_tag,
        iteration=args.iteration,
    )


if __name__ == "__main__":
    main()
