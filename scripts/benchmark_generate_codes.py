#!/usr/bin/env python3
"""
Generate C code using base Qwen vs DPO fine-tuned model for benchmarking.

Takes algorithm descriptions from a generation tag and generates C code
implementations using both models for comparison.

Usage:
    # On GPU node:
    python scripts/benchmark_generate_codes.py --tag dpo_testing --output outputs/benchmark
"""

import os
import sys
import json
import argparse
import torch
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from llmsat.llmsat import CHATGPT_DATA_GENERATION_TABLE, setup_logging, get_logger
from llmsat.utils.aws import get_ids_from_router_table, get_algorithm_result

setup_logging()
logger = get_logger(__name__)

# Model paths
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
FINETUNED_MODEL_PATH = "outputs/dpo1/dpo_training/final_model"

# Generation config
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9


def extract_algorithm_description(raw_algorithm: str) -> Optional[str]:
    """Extract the actual algorithm description from the raw database field."""
    if not raw_algorithm:
        return None

    def parse_algorithm_json(text: str) -> Optional[str]:
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
        data = json.loads(raw_algorithm)
        if "response" in data and "body" in data.get("response", {}):
            body = data["response"]["body"]
            if "output" in body and len(body["output"]) > 0:
                content = body["output"][0].get("content", [])
                if len(content) > 0:
                    text = content[0].get("text", "")
                    result = parse_algorithm_json(text)
                    if result:
                        return result
                    return text
        if "name" in data and "algorithm" in data:
            return f"Name: {data['name']}\n\nAlgorithm: {data['algorithm']}"
        return raw_algorithm
    except json.JSONDecodeError:
        result = parse_algorithm_json(raw_algorithm)
        if result:
            return result
        return raw_algorithm


def load_model_and_tokenizer(model_path: str, is_finetuned: bool = False):
    """Load model with QLoRA quantization."""
    print(f"\nLoading model: {model_path}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if is_finetuned:
        # Load base model then apply PEFT adapter
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        print(f"  Loaded fine-tuned model with adapter from {model_path}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        print(f"  Loaded base model from {model_path}")

    model.eval()
    return model, tokenizer


def format_prompt(algorithm_description: str) -> str:
    """Format the prompt for code generation (same as DPO training)."""
    # Note: Keep this format consistent with DPO training data
    # The function signature MUST be: bool kissat_restarting(kissat *solver)
    return f"""Implement the following Kissat restart heuristic algorithm as a C function:

{algorithm_description}

Provide the complete kissat_restarting() function implementation.
The function signature must be: bool kissat_restarting(kissat *solver)
The solver parameter provides access to: solver->stable, solver->level, solver->unassigned,
solver->statistics.conflicts, solver->statistics.decisions, solver->limits.restart.conflicts,
AVERAGE(fast_glue), AVERAGE(slow_glue), AVERAGE(decision_rate), GET_OPTION(restart),
GET_OPTION(restartmargin), GET_OPTION(restartint), kissat_reluctant_triggered(&solver->reluctant)."""


def generate_code(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
) -> str:
    """Generate C code from the model."""

    # Format as chat message
    messages = [
        {"role": "user", "content": prompt}
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated part
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return generated_text.strip()


def extract_c_code(generated_text: str) -> str:
    """Extract C code from generated text (may be wrapped in markdown)."""
    text = generated_text.strip()

    # Try to extract from markdown code blocks
    if "```c" in text:
        start = text.find("```c") + 4
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()

    return text


@dataclass
class GeneratedCode:
    algorithm_id: str
    algorithm_description: str
    model_type: str  # "base" or "finetuned"
    generated_code: str
    raw_output: str


def get_algorithms_from_tag(generation_tag: str, limit: Optional[int] = None) -> List[Dict]:
    """Fetch algorithm descriptions from database by generation tag."""
    algorithm_ids = get_ids_from_router_table(CHATGPT_DATA_GENERATION_TABLE, generation_tag)

    if not algorithm_ids:
        logger.error(f"No algorithms found for tag: {generation_tag}")
        return []

    if limit:
        algorithm_ids = algorithm_ids[:limit]

    algorithms = []
    for algo_id in algorithm_ids:
        result = get_algorithm_result(algo_id)
        if result is None:
            continue

        description = extract_algorithm_description(result.algorithm)
        if description:
            algorithms.append({
                "id": algo_id,
                "description": description
            })

    logger.info(f"Loaded {len(algorithms)} algorithms from tag '{generation_tag}'")
    return algorithms


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark codes with base vs fine-tuned model")
    parser.add_argument("--tag", "-t", required=True, help="Generation tag for algorithm descriptions")
    parser.add_argument("--output", "-o", required=True, help="Output directory for generated codes")
    parser.add_argument("--limit", "-n", type=int, default=None, help="Limit number of algorithms (default: all)")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Max new tokens")
    parser.add_argument("--base-only", action="store_true", help="Only run base model")
    parser.add_argument("--finetuned-only", action="store_true", help="Only run fine-tuned model")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Fetch algorithms
    print(f"\n{'='*60}")
    print("Fetching algorithms from database...")
    print(f"{'='*60}")
    algorithms = get_algorithms_from_tag(args.tag, args.limit)

    if not algorithms:
        print("No algorithms found!")
        sys.exit(1)

    print(f"Found {len(algorithms)} algorithms to process")

    all_results = []

    # Generate with base model
    if not args.finetuned_only:
        print(f"\n{'='*60}")
        print("PHASE 1: Generating codes with BASE model")
        print(f"{'='*60}")

        model, tokenizer = load_model_and_tokenizer(BASE_MODEL, is_finetuned=False)

        for i, algo in enumerate(algorithms, 1):
            print(f"\n[Base Model] Algorithm {i}/{len(algorithms)}: {algo['id'][:16]}...")

            prompt = format_prompt(algo['description'])
            raw_output = generate_code(model, tokenizer, prompt,
                                       max_new_tokens=args.max_tokens,
                                       temperature=args.temperature)
            code = extract_c_code(raw_output)

            result = GeneratedCode(
                algorithm_id=algo['id'],
                algorithm_description=algo['description'],
                model_type="base",
                generated_code=code,
                raw_output=raw_output
            )
            all_results.append(asdict(result))

            print(f"  Generated {len(code)} chars of code")

        # Free memory
        del model
        torch.cuda.empty_cache()

    # Generate with fine-tuned model
    if not args.base_only:
        print(f"\n{'='*60}")
        print("PHASE 2: Generating codes with FINE-TUNED model")
        print(f"{'='*60}")

        model, tokenizer = load_model_and_tokenizer(FINETUNED_MODEL_PATH, is_finetuned=True)

        for i, algo in enumerate(algorithms, 1):
            print(f"\n[Fine-tuned Model] Algorithm {i}/{len(algorithms)}: {algo['id'][:16]}...")

            prompt = format_prompt(algo['description'])
            raw_output = generate_code(model, tokenizer, prompt,
                                       max_new_tokens=args.max_tokens,
                                       temperature=args.temperature)
            code = extract_c_code(raw_output)

            result = GeneratedCode(
                algorithm_id=algo['id'],
                algorithm_description=algo['description'],
                model_type="finetuned",
                generated_code=code,
                raw_output=raw_output
            )
            all_results.append(asdict(result))

            print(f"  Generated {len(code)} chars of code")

        del model
        torch.cuda.empty_cache()

    # Save results
    output_file = os.path.join(args.output, "generated_codes.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total codes generated: {len(all_results)}")
    print(f"Results saved to: {output_file}")

    # Summary statistics
    base_codes = [r for r in all_results if r['model_type'] == 'base']
    finetuned_codes = [r for r in all_results if r['model_type'] == 'finetuned']

    print(f"\nBase model: {len(base_codes)} codes")
    print(f"Fine-tuned model: {len(finetuned_codes)} codes")

    if base_codes:
        avg_len_base = sum(len(r['generated_code']) for r in base_codes) / len(base_codes)
        print(f"Base model avg code length: {avg_len_base:.0f} chars")

    if finetuned_codes:
        avg_len_ft = sum(len(r['generated_code']) for r in finetuned_codes) / len(finetuned_codes)
        print(f"Fine-tuned model avg code length: {avg_len_ft:.0f} chars")


if __name__ == "__main__":
    main()