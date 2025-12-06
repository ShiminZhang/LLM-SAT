#!/usr/bin/env python3
"""
Generate codes for algorithms that don't have any code implementations yet.
Uses ChatGPT batch API.
"""

import sys
import os
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llmsat.llmsat import (
    CHATGPT_DATA_GENERATION_TABLE,
    AlgorithmStatus,
    CodeStatus,
    CodeResult,
    setup_logging,
    get_logger,
    get_id,
    NOT_INITIALIZED
)
from llmsat.utils.aws import (
    get_ids_from_router_table,
    get_algorithm_result,
    update_algorithm_result,
    update_code_result
)
from llmsat.pipelines.chatgpt_data_generation import (
    read_code_prompt_template,
    generate_code_prompt,
    create_batch_input_file,
    submit_batch_input,
    block_until_completion,
    download_batch_outputs,
    parse_code_response
)
from llmsat.utils.paths import get_batch_output_dir
from datetime import datetime
import json

setup_logging()
logger = get_logger(__name__)

def generate_codes_for_algorithms(generation_tag, code_prompt_template_path, n_codes=10, model="gpt-4o"):
    """
    Generate codes for algorithms in a generation tag that don't have any codes yet.
    """
    logger.info(f"Generating codes for tag: {generation_tag}")

    # Get all algorithm IDs for this tag
    algorithm_ids = get_ids_from_router_table(CHATGPT_DATA_GENERATION_TABLE, generation_tag)
    logger.info(f"Found {len(algorithm_ids)} algorithms in tag '{generation_tag}'")

    # Filter to only those with no codes
    algorithms_needing_codes = []
    for algo_id in algorithm_ids:
        algo = get_algorithm_result(algo_id)
        if algo and (not algo.code_id_list or len(algo.code_id_list) == 0):
            algorithms_needing_codes.append(algo)

    logger.info(f"Found {len(algorithms_needing_codes)} algorithms needing code generation")

    if not algorithms_needing_codes:
        logger.info("No algorithms need code generation. Exiting.")
        return

    # Read code prompt template
    code_prompt_template = read_code_prompt_template(code_prompt_template_path)

    # Create output directory
    output_dir = get_batch_output_dir(generation_tag, batch_id="code_regen")
    os.makedirs(output_dir, exist_ok=True)

    # Submit batch jobs for each algorithm
    waiting_batch_ids = []
    batch_id_to_algorithm_id = {}

    for algo in algorithms_needing_codes:
        logger.info(f"Submitting code generation batch for algorithm {algo.id[:16]}...")
        code_prompt = generate_code_prompt(code_prompt_template, algo.algorithm)
        code_batch_input_path = os.path.join(output_dir, f"code_batch_input_{algo.id}.txt")

        create_batch_input_file(code_prompt, code_batch_input_path, n_requests=n_codes, model=model)
        batch_id = submit_batch_input(code_batch_input_path)

        waiting_batch_ids.append(batch_id)
        batch_id_to_algorithm_id[batch_id] = algo.id
        logger.info(f"  Submitted batch {batch_id}")

    # Save batch mapping
    batch_map_path = os.path.join(output_dir, f"batch_id_map_{datetime.now().strftime('%Y%m%d%H%M%S')}.json")
    with open(batch_map_path, 'w') as f:
        json.dump({
            'code_batch_ids': waiting_batch_ids,
            'code_batch_map': batch_id_to_algorithm_id
        }, f, indent=2)
    logger.info(f"Saved batch mapping to {batch_map_path}")

    # Wait for all batches to complete and download results
    logger.info(f"Waiting for {len(waiting_batch_ids)} batches to complete...")

    for i, batch_id in enumerate(waiting_batch_ids, 1):
        logger.info(f"Processing batch {i}/{len(waiting_batch_ids)}: {batch_id}")

        # Wait for completion
        block_until_completion(batch_id)

        # Download outputs
        code_output_path = os.path.join(output_dir, f"code_output_{batch_id}.txt")
        download_batch_outputs(batch_id, code_output_path)

        # Get the algorithm for this batch
        mapped_algorithm_id = batch_id_to_algorithm_id.get(batch_id)
        if not mapped_algorithm_id:
            logger.warning(f"Unknown batch_id {batch_id}, skipping")
            continue

        algorithm_result = get_algorithm_result(mapped_algorithm_id)
        if not algorithm_result:
            logger.warning(f"Algorithm {mapped_algorithm_id} not found, skipping")
            continue

        # Parse codes from output
        if algorithm_result.code_id_list is None:
            algorithm_result.code_id_list = []

        existing_ids = set(algorithm_result.code_id_list)
        codes_added = 0

        with open(code_output_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    code_response = json.loads(line)
                    code_str = parse_code_response(code_response)

                    code_id = get_id(code_str)

                    # Create code result
                    code_result = CodeResult(
                        id=code_id,
                        algorithm_id=mapped_algorithm_id,
                        code=code_str,
                        status=CodeStatus.Generated,
                        last_updated=datetime.now(),
                        build_success=NOT_INITIALIZED,
                        par2=None
                    )
                    update_code_result(code_result)

                    # Add to algorithm's code list if not already there
                    if code_id not in existing_ids:
                        algorithm_result.code_id_list.append(code_id)
                        existing_ids.add(code_id)
                        codes_added += 1

                except Exception as e:
                    logger.warning(f"Error parsing code response: {e}")
                    continue

        # Update algorithm status
        algorithm_result.status = AlgorithmStatus.CodeGenerated
        update_algorithm_result(algorithm_result)

        logger.info(f"  Added {codes_added} codes for algorithm {mapped_algorithm_id[:16]}...")

    logger.info("Code generation complete!")

def main():
    parser = argparse.ArgumentParser(description="Generate codes for algorithms missing code implementations")
    parser.add_argument("--generation_tag", type=str, required=True, help="Generation tag to process")
    parser.add_argument("--code_prompt_template", type=str, default="./data/prompts/kissat_code.txt",
                       help="Path to code prompt template")
    parser.add_argument("--n_codes", type=int, default=10, help="Number of code variants per algorithm")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model to use (gpt-4o, gpt-5, etc.)")

    args = parser.parse_args()

    generate_codes_for_algorithms(
        generation_tag=args.generation_tag,
        code_prompt_template_path=args.code_prompt_template,
        n_codes=args.n_codes,
        model=args.model
    )

if __name__ == "__main__":
    main()