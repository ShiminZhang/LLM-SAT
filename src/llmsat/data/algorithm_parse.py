from __future__ import annotations
from dataclasses import dataclass
from typing import List, Any, Optional
import json
from llmsat.utils.aws import update_algorithm_result, get_algorithm_result
from llmsat.llmsat import *
from datetime import datetime
import argparse

@dataclass
class RestartPolicySpec:
    algorithm_name: str
    scoring_function_definition: str
    selection_procedure: List[str]
    tie_breaking_rules: List[str]

EXPECTED_KEYS = [
    "Algorithm Name",
    "Scoring Function Definition",
    "Selection Procedure",
    "Tie-breaking Rules",
]

def _has_required_keys(obj: dict) -> bool:
    return all(k in obj for k in EXPECTED_KEYS)

def _find_embedded_json_string(value: Any) -> Optional[str]:
    """
    Traverse dicts/lists to find a JSON string that looks like the algorithm spec.
    This is needed when the JSONL line wraps the payload (e.g., API response objects)
    and the spec is stored as a string in a nested 'text' field.
    """
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("{") and '"Algorithm Name"' in s:
            return s
        return None
    if isinstance(value, dict):
        for v in value.values():
            found = _find_embedded_json_string(v)
            if found is not None:
                return found
        return None
    if isinstance(value, list):
        for v in value:
            found = _find_embedded_json_string(v)
            if found is not None:
                return found
        return None
    return None

def parse_kissat_restart_policy_json(text: str) -> dict[str]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e

    # Case 1: Top-level object is already the spec
    if isinstance(obj, dict) and _has_required_keys(obj):
        spec_obj = obj
    else:
        # Case 2: The line is a wrapped response; try to find the embedded spec JSON string
        embedded = _find_embedded_json_string(obj)
        if embedded is None:
            raise ValueError("Could not locate algorithm spec JSON in input line.")
        try:
            spec_obj = json.loads(embedded)
        except json.JSONDecodeError as e:
            raise ValueError(f"Embedded algorithm spec is invalid JSON: {e}") from e

    if not isinstance(spec_obj, dict):
        raise ValueError("Algorithm spec must be a JSON object.")

    # Validate required keys; ignore any extra keys such as 'Reason'
    missing = [k for k in EXPECTED_KEYS if k not in spec_obj]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    algorithm_name = spec_obj["Algorithm Name"]
    scoring_fn = spec_obj["Scoring Function Definition"]
    selection = spec_obj["Selection Procedure"]
    tie_break = spec_obj["Tie-breaking Rules"]

    if not isinstance(algorithm_name, str) or not algorithm_name.strip():
        raise ValueError("Algorithm Name must be a non-empty string.")
    if not isinstance(scoring_fn, str) or not scoring_fn.strip():
        raise ValueError("Scoring Function Definition must be a non-empty string.")
    if not isinstance(selection, list) or not all(isinstance(x, str) and x.strip() for x in selection):
        raise ValueError("Selection Procedure must be an array of non-empty strings.")
    if not isinstance(tie_break, list) or not all(isinstance(x, str) and x.strip() for x in tie_break):
        raise ValueError("Tie-breaking Rules must be an array of non-empty strings.")

    return spec_obj

def main():
    # read file from argument
    parser = argparse.ArgumentParser(description="Parse Kissat restart policy JSON")
    parser.add_argument("--input", type=str, default="data/algorithm_generation_outputs/kissat_algorithm_outputs.jsonl", help="Input file path")
    parser.add_argument("--output", type=str, default="data/algorithm_response.jsonl", help="Output file path")
    parser.add_argument("--print_algorithms", type=int, default=0, help="Print the first n algorithms")
    args = parser.parse_args()
    with open(args.input, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            algorithm = parse_kissat_restart_policy_json(line)
            algorithm.pop("Reason")
            algorithm_result = AlgorithmResult(
                id=get_id(str(algorithm)),
                algorithm=str(algorithm),
                status=AlgorithmStatus.Generated,
                last_updated=datetime.now(),
                prompt="",
                par2=NOT_INITIALIZED, 
                error_rate=NOT_INITIALIZED,
                code_id_list=NOT_INITIALIZED,
                other_metrics=NOT_INITIALIZED
            )
            update_algorithm_result(algorithm_result)
            retrieved_algorithm_result = get_algorithm_result(algorithm_result.id)
            print(retrieved_algorithm_result.algorithm)


if __name__ == "__main__":
    main()