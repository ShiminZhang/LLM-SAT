from __future__ import annotations
from dataclasses import dataclass
from typing import List, Any, Optional, Tuple
import json
import re
from llmsat.utils.aws import update_algorithm_result, get_algorithm_result
from llmsat.llmsat import *  # includes Role, AlgorithmResult, AlgorithmStatus, get_id, NOT_INITIALIZED
from datetime import datetime
import argparse

@dataclass
class RestartPolicySpec:
    algorithm_name: str
    scoring_function_definition: str
    selection_procedure: List[str]
    tie_breaking_rules: List[str]

# New format keys (ae_prompt.txt style)
# target_function is optional for variant/member algorithms (they inherit from leader)
NEW_FORMAT_REQUIRED_KEYS = ["name", "algorithm"]
NEW_FORMAT_OPTIONAL_KEYS = ["reason", "target_function"]

# Legacy format keys (kissat.txt style)
LEGACY_EXPECTED_KEYS = [
    "Algorithm Name",
    "Scoring Function Definition",
    "Selection Procedure",
    "Tie-breaking Rules",
]

def _has_new_format_keys(obj: dict) -> bool:
    return all(k in obj for k in NEW_FORMAT_REQUIRED_KEYS)

def _has_legacy_keys(obj: dict) -> bool:
    return all(k in obj for k in LEGACY_EXPECTED_KEYS)

def _find_embedded_json_string(value: Any) -> Optional[str]:
    """
    Traverse dicts/lists to find a JSON string that looks like the algorithm spec.
    Supports both new format ("name") and legacy format ("Algorithm Name").
    """
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("{") and ('"name"' in s or '"Algorithm Name"' in s):
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

def parse_algorithm_spec_json(text: str) -> Tuple[dict, Optional[str]]:
    """
    Parse algorithm specification JSON from LLM output.
    
    Supports two formats:
    1. New format (ae_prompt.txt): {"name", "algorithm", "target_function", "reason"?}
    2. Legacy format (kissat.txt): {"Algorithm Name", "Scoring Function Definition", ...}
    
    Returns:
        Tuple of (spec_dict, target_function)
        - spec_dict: The parsed algorithm specification
        - target_function: The target function name (None for legacy format)
    """
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # LLMs sometimes produce invalid escape sequences like \e, \a, etc.
        # Replace invalid escapes with their literal characters (e.g., \e -> \\e)
        sanitized = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)
        try:
            obj = json.loads(sanitized)
        except json.JSONDecodeError as e2:
            raise ValueError(f"Invalid JSON: {e2}") from e2

    # Try to find the spec object
    spec_obj = None
    if isinstance(obj, dict):
        if _has_new_format_keys(obj) or _has_legacy_keys(obj):
            spec_obj = obj
    
    if spec_obj is None:
        # Try to find embedded JSON string
        embedded = _find_embedded_json_string(obj)
        if embedded is None:
            raise ValueError("Could not locate algorithm spec JSON in input.")
        try:
            spec_obj = json.loads(embedded)
        except json.JSONDecodeError as e:
            raise ValueError(f"Embedded algorithm spec is invalid JSON: {e}") from e

    if not isinstance(spec_obj, dict):
        raise ValueError("Algorithm spec must be a JSON object.")

    # Determine format and validate
    if _has_new_format_keys(spec_obj):
        return _validate_new_format(spec_obj)
    elif _has_legacy_keys(spec_obj):
        return _validate_legacy_format(spec_obj)
    else:
        raise ValueError(f"Missing required keys. Expected either {NEW_FORMAT_REQUIRED_KEYS} or {LEGACY_EXPECTED_KEYS}")

def _validate_new_format(spec_obj: dict) -> Tuple[dict, Optional[str]]:
    """Validate new format and return (spec, target_function).

    target_function is optional for variant/member algorithms (they inherit from leader).
    """
    name = spec_obj.get("name")
    algorithm = spec_obj.get("algorithm")
    target_function = spec_obj.get("target_function")

    if not isinstance(name, str) or not name.strip():
        raise ValueError("'name' must be a non-empty string.")
    if not isinstance(algorithm, str) or not algorithm.strip():
        raise ValueError("'algorithm' must be a non-empty string.")
    # target_function is optional (variants/members inherit from leader)
    if target_function is not None and (not isinstance(target_function, str) or not target_function.strip()):
        raise ValueError("'target_function' must be a non-empty string if provided.")

    return spec_obj, target_function

def _validate_legacy_format(spec_obj: dict) -> Tuple[dict, None]:
    """Validate legacy format and return (spec, None) since no target_function."""
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

    return spec_obj, None

# Backward compatibility alias
def parse_kissat_restart_policy_json(text: str) -> dict:
    """Legacy function name for backward compatibility. Returns only the spec dict."""
    spec, _ = parse_algorithm_spec_json(text)
    return spec

def generate_and_parse_algorithms(input_file: str, output_file: str) -> None:
    pass

def parse_algorithms(input_file: str, output_file: str) -> None:
    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            algorithm = parse_kissat_restart_policy_json(line)
            reason = algorithm.pop("Reason", None) or algorithm.pop("reason", None)
            name = algorithm.get("name", algorithm.get("Algorithm Name", ""))
            algo_text = algorithm.get("algorithm", str(algorithm))
            description = f"{name}: {algo_text}" if name else algo_text
            algorithm_result = AlgorithmResult(
                id=get_id(description),
                function_name="kissat_rescale_scores",
                description=description,
                role=Role.LEADER,
                status=AlgorithmStatus.Generated,
                last_updated=datetime.now(),
                code_id_list=[],
                analysis=reason,
                prompt="",
            )
            update_algorithm_result(algorithm_result)
            retrieved_algorithm_result = get_algorithm_result(algorithm_result.id)
            print(retrieved_algorithm_result.description)
            with open(output_file, "a") as f:
                f.write(retrieved_algorithm_result.description + "\n")

def main():
    # read file from argument
    parser = argparse.ArgumentParser(description="Parse Kissat restart policy JSON")
    parser.add_argument("--input", type=str, default="data/algorithm_generation_outputs/kissat_algorithm_outputs.jsonl", help="Input file path")
    parser.add_argument("--output", type=str, default="data/algorithm_response.jsonl", help="Output file path")
    parser.add_argument("--print_algorithms", type=int, default=0, help="Print the first n algorithms")
    parser.add_argument("--generate_and_parse", type=bool, default=False, help="Generate and parse the algorithms")
    args = parser.parse_args()
    if args.generate_and_parse:
        generate_and_parse_algorithms(args.input, args.output)
    else:
        parse_algorithms(args.input, args.output)


if __name__ == "__main__":
    main()