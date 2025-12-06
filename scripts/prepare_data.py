#!/usr/bin/env python3
"""
Prepare DPO training data from generated preference pairs.

Converts JSON data from generate_dpo_data.py into HuggingFace Dataset format
suitable for DPO training with trl.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --input data/dpo_training_data.json --output data/dpo_formatted
"""
import json
import argparse
import os
from datasets import Dataset


def convert_to_dpo_format(raw_data_path, output_path):
    """
    Convert raw DPO data to HuggingFace Dataset format.

    Input format (JSON array):
    [
        {
            "algorithm_description": "...",
            "preferred_code": "...",
            "rejected_code": "...",
            "par2_preferred": float,
            "par2_rejected": float
        },
        ...
    ]

    Output format (HuggingFace Dataset with columns):
    - prompt: The algorithm description formatted as a prompt
    - chosen: The preferred (lower PAR2) code
    - rejected: The rejected (higher PAR2) code
    """

    dpo_data = []

    # Load JSON file (not JSONL)
    with open(raw_data_path, 'r') as f:
        raw_data = json.load(f)

    print(f"Loaded {len(raw_data)} preference pairs from {raw_data_path}")

    for item in raw_data:
        dpo_example = {
            "prompt": f"Implement the following Kissat restart heuristic algorithm as a C function:\n\n{item['algorithm_description']}\n\nProvide the complete kissat_restarting() function implementation:",
            "chosen": item['preferred_code'],
            "rejected": item['rejected_code']
        }
        dpo_data.append(dpo_example)

    dataset = Dataset.from_list(dpo_data)

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    dataset.save_to_disk(output_path)

    print(f"Converted {len(dpo_data)} examples to DPO format")
    print(f"Saved to {output_path}")

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare DPO training data")
    parser.add_argument("--input", type=str, default="data/dpo_training_data.json",
                        help="Path to input JSON file from generate_dpo_data.py")
    parser.add_argument("--output", type=str, default="data/dpo_formatted",
                        help="Output directory for HuggingFace Dataset")
    args = parser.parse_args()

    dataset = convert_to_dpo_format(args.input, args.output)

    print("\n" + "="*60)
    print("Dataset Statistics:")
    print("="*60)
    print(f"Total examples: {len(dataset)}")
    print(f"\nExample Entry:")
    print(f"  Prompt length: {len(dataset[0]['prompt'])} chars")
    print(f"  Chosen length: {len(dataset[0]['chosen'])} chars")
    print(f"  Rejected length: {len(dataset[0]['rejected'])} chars")
    print(f"\nPrompt preview:\n{dataset[0]['prompt'][:300]}...")
