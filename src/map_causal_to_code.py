"""
Map causal reports to their corresponding code_batch_input and code_output files.

Usage:
    python src/llmsat/utils/map_causal_to_code.py \
        --folder outputs/controlled_mutation \
        --causal_reports outputs/controlled_mutation_gen1/causal_reports.json
"""

import argparse
import json
import os
import glob


def build_mapping(folder: str, causal_reports_path: str) -> list[dict]:
    """
    For each algorithm_id in causal_reports.json, find:
      - code_batch_input_{algorithm_id}.txt  (the code generation prompt)
      - code_output_batch_{openai_batch_id}.txt  (the generated code)

    Returns list of dicts with algorithm_id, causal_report, input_file, output_file.
    """
    # Find batch dir
    batch_dir = None
    for entry in sorted(os.listdir(folder)):
        full = os.path.join(folder, entry)
        if entry.startswith("batch_") and os.path.isdir(full):
            batch_dir = full
            break
    if batch_dir is None:
        if os.path.exists(os.path.join(folder, "leaders_output.txt")):
            batch_dir = folder
        else:
            raise FileNotFoundError(f"No batch directory found in {folder}")

    # Load causal reports
    with open(causal_reports_path, "r") as f:
        causal_reports = json.load(f)

    # Load team_batch_id_map to get code_batch_map (openai_batch_id -> algorithm_id)
    map_files = sorted(glob.glob(os.path.join(batch_dir, "team_batch_id_map_*.json")))
    if not map_files:
        raise FileNotFoundError(f"No team_batch_id_map found in {batch_dir}")

    with open(map_files[-1], "r") as f:
        batch_map = json.load(f)

    # Invert code_batch_map: algorithm_id -> openai_batch_id
    code_batch_map = batch_map.get("code_batch_map", {})
    algo_to_openai_batch = {algo_id: batch_id for batch_id, algo_id in code_batch_map.items()}

    # Build the mapping
    results = []
    for algo_id, report in causal_reports.items():
        # code_batch_input file is named by algorithm_id
        input_file = os.path.join(batch_dir, f"code_batch_input_{algo_id}.txt")
        input_exists = os.path.exists(input_file)

        # code_output file is named by openai batch_id
        openai_batch_id = algo_to_openai_batch.get(algo_id)
        output_file = None
        output_exists = False
        if openai_batch_id:
            output_file = os.path.join(batch_dir, f"code_output_{openai_batch_id}.txt")
            output_exists = os.path.exists(output_file)

        results.append({
            "algorithm_id": algo_id,
            "causal_report": report,
            "code_input_file": input_file if input_exists else None,
            "code_output_file": output_file if output_exists else None,
            "openai_batch_id": openai_batch_id,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Map causal reports to code input/output files")
    parser.add_argument("--folder", required=True, help="Path to outputs folder (e.g. outputs/controlled_mutation)")
    parser.add_argument("--causal_reports", required=True, help="Path to causal_reports.json")
    parser.add_argument("--output", default=None, help="Save mapping to JSON file (optional)")
    args = parser.parse_args()

    mapping = build_mapping(args.folder, args.causal_reports)

    # Print summary
    found_input = sum(1 for m in mapping if m["code_input_file"])
    found_output = sum(1 for m in mapping if m["code_output_file"])
    print(f"Mapped {len(mapping)} algorithms:")
    print(f"  {found_input}/{len(mapping)} have code_batch_input files")
    print(f"  {found_output}/{len(mapping)} have code_output files")

    # Print each mapping
    for m in mapping:
        status = []
        if m["code_input_file"]:
            status.append(f"input={os.path.basename(m['code_input_file'])}")
        else:
            status.append("input=MISSING")
        if m["code_output_file"]:
            status.append(f"output={os.path.basename(m['code_output_file'])}")
        else:
            status.append("output=MISSING")
        print(f"  {m['algorithm_id'][:16]}... | {' | '.join(status)}")

    if args.output:
        # Strip causal report from saved mapping to keep it small
        save_data = []
        for m in mapping:
            save_data.append({
                "algorithm_id": m["algorithm_id"],
                "code_input_file": m["code_input_file"],
                "code_output_file": m["code_output_file"],
                "openai_batch_id": m["openai_batch_id"],
            })
        with open(args.output, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nSaved mapping to {args.output}")


if __name__ == "__main__":
    main()
