"""
Reset CodeStatus.Evaluating -> CodeStatus.Generated for all codes under a given
output_tag whose SLURM jobs were never actually submitted (e.g. due to a failed
sbatch call). This unblocks re-evaluation on the next run.

Usage:
    python src/llmsat/pipelines/reset_evaluating_status.py \
        --output_tag gemini_trial5_gen1_v2 \
        --iteration 1
"""

import argparse

from llmsat.llmsat import CHATGPT_DATA_GENERATION_TABLE, CodeStatus, get_logger, setup_logging
from llmsat.utils.aws import (
    get_algorithm_result,
    get_code_result,
    get_ids_from_router_table,
    update_code_result,
)

setup_logging()
logger = get_logger(__name__)


def reset_evaluating_status(output_tag: str, iteration: int = 1) -> None:
    iter_tag = f"{output_tag}_iter{iteration}"
    algorithm_ids = get_ids_from_router_table(CHATGPT_DATA_GENERATION_TABLE, iter_tag)
    logger.info(f"Found {len(algorithm_ids)} algorithms under tag '{iter_tag}'")

    reset = 0
    skipped = 0

    for algo_id in algorithm_ids:
        algo = get_algorithm_result(algo_id)
        if not algo or not algo.code_id_list:
            continue
        for code_id in algo.code_id_list:
            code_result = get_code_result(code_id)
            if code_result is None:
                continue
            if code_result.status == CodeStatus.Evaluating:
                code_result.status = CodeStatus.Generated
                update_code_result(code_result)
                logger.info(f"  Reset {code_id[:16]} -> Generated")
                reset += 1
            else:
                skipped += 1

    logger.info(f"Done: {reset} reset to Generated, {skipped} skipped (not Evaluating)")


def main():
    parser = argparse.ArgumentParser(description="Reset stuck Evaluating status in DB")
    parser.add_argument("--output_tag", type=str, required=True,
                        help="Output tag (e.g. gemini_trial5_gen1_v2)")
    parser.add_argument("--iteration", type=int, default=1,
                        help="Iteration number (default: 1)")
    args = parser.parse_args()
    reset_evaluating_status(args.output_tag, args.iteration)


if __name__ == "__main__":
    main()
