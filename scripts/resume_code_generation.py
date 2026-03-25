#!/usr/bin/env python3
"""
Resume code generation for algorithms that were created but didn't get code
(e.g., after a 503 crash during the code generation step).

Usage:
    PYTHONPATH=src python scripts/resume_code_generation.py --tag decide_iter0 --model gemini-3.1-pro-preview
"""

import argparse
from datetime import datetime

from llmsat.llmsat import (
    CHATGPT_DATA_GENERATION_TABLE,
    AlgorithmStatus,
    CodeResult,
    CodeStatus,
    NOT_INITIALIZED,
)
from llmsat.pipelines.gemini_data_generation import (
    generate_code_prompt,
    get_id,
    parse_code_response,
)
from llmsat.utils.aws import (
    get_algorithm_result,
    get_ids_from_router_table,
    update_algorithm_result,
    update_code_result,
)
from llmsat.utils.gemini_helper import get_response_from_gemini


def main():
    parser = argparse.ArgumentParser(description="Resume code generation after a crash")
    parser.add_argument("--tag", required=True, help="Generation tag (e.g. decide_iter0)")
    parser.add_argument("--model", default="gemini-3.1-pro-preview", help="Gemini model")
    parser.add_argument(
        "--code_prompt_path",
        default="data/prompts/coder_prompt_testing.txt",
        help="Path to coder prompt template",
    )
    args = parser.parse_args()

    code_template = open(args.code_prompt_path).read()

    ids = get_ids_from_router_table(CHATGPT_DATA_GENERATION_TABLE, args.tag)
    print(f"Found {len(ids)} algorithms in {args.tag}")

    generated = 0
    skipped = 0
    for i, alg_id in enumerate(ids):
        ar = get_algorithm_result(alg_id)
        if ar.code_id_list and len(ar.code_id_list) > 0:
            print(f"  [{i+1}/{len(ids)}] {alg_id[:16]}... already has code, skipping")
            skipped += 1
            continue

        print(f"  [{i+1}/{len(ids)}] {alg_id[:16]}... generating code")
        prompt = generate_code_prompt(code_template, ar.description, ar.function_name)
        raw = get_response_from_gemini(prompt, model=args.model)
        code_str = parse_code_response({"text": raw})
        code_id = get_id(code_str)

        update_code_result(
            CodeResult(
                id=code_id,
                algorithm_id=alg_id,
                code=code_str,
                status=CodeStatus.Generated,
                par2=None,
                last_updated=datetime.now(),
                build_success=NOT_INITIALIZED,
            )
        )

        if ar.code_id_list is None:
            ar.code_id_list = []
        ar.code_id_list.append(code_id)
        ar.status = AlgorithmStatus.CodeGenerated
        update_algorithm_result(ar)
        generated += 1

    print(f"\nDone: {generated} generated, {skipped} skipped (already had code)")


if __name__ == "__main__":
    main()
