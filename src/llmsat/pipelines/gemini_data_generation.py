import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from llmsat.utils.gemini_helper import (
    build_gemini_batch_request,
    write_gemini_batch_jsonl,
    submit_batch_input as helper_submit_batch_input,
    block_until_completion as helper_block_until_completion,
    download_batch_outputs as helper_download_batch_outputs,
    wait_for_all_batches,
)
from llmsat.utils.paths import get_batch_output_dir, get_generation_output_dir
from llmsat.utils.aws import (
    get_ids_from_router_table,
    update_router_table,
    get_algorithm_result,
    get_code_result,
)
from llmsat.llmsat import (
    CHATGPT_DATA_GENERATION_TABLE,
    AlgorithmStatus,
    AlgorithmResult,
    CodeResult,
    CodeStatus,
    get_id,
    NOT_INITIALIZED,
    setup_logging,
    get_logger,
)
from llmsat.utils.aws import update_algorithm_result, update_code_result
from datetime import datetime
import json
import logging

setup_logging(level=logging.INFO)
logger = get_logger(__name__)

# Default model for Gemini
DEFAULT_MODEL = "gemini-3-pro-preview"


def read_prompt_file(path: str) -> str:
    """Read a prompt template from file."""
    with open(path, "r") as f:
        return f.read()


def create_batch_input_file(
    prompt: str,
    output_path: str,
    n_requests: int = 10,
    model: str = DEFAULT_MODEL,
):
    """Create a Gemini batch input JSONL file with identical prompts."""
    logger.info(f"Creating batch input file for {n_requests} requests")

    system_message = os.environ.get(
        "LLMSAT_SYSTEM_MESSAGE",
        "You are an AI researcher specialising in SAT solver heuristics.",
    )
    model = os.environ.get("GEMINI_MODEL", model)
    try:
        temperature = float(os.environ.get("GEMINI_TEMPERATURE", "0.7"))
    except Exception:
        temperature = 0.7

    requests = []
    for i in range(1, int(n_requests) + 1):
        custom_id = f"req-{i:04d}"
        requests.append(
            build_gemini_batch_request(
                prompt=prompt,
                system_message=system_message,
                model=model,
                temperature=temperature,
                custom_id=custom_id,
            )
        )

    write_gemini_batch_jsonl(requests, Path(output_path))


def count_steps(algorithm_text: str) -> int:
    """Count the number of steps in an algorithm based on 'Step N:' markers."""
    matches = re.findall(r"Step\s+\d+:", algorithm_text, re.IGNORECASE)
    return len(matches)


def create_batch_input_file_variant(
    prompts: List[str],
    output_path: str,
    model: str = DEFAULT_MODEL,
):
    """Create batch input file with different prompts for each request."""
    logger.info(f"Creating variant batch input file for {len(prompts)} requests")

    system_message = os.environ.get(
        "LLMSAT_SYSTEM_MESSAGE",
        "You are an AI researcher specialising in SAT solver heuristics.",
    )
    model = os.environ.get("GEMINI_MODEL", model)
    try:
        temperature = float(os.environ.get("GEMINI_TEMPERATURE", "0.7"))
    except Exception:
        temperature = 0.7

    requests = []
    for i, prompt in enumerate(prompts, start=1):
        custom_id = f"req-{i:04d}"
        requests.append(
            build_gemini_batch_request(
                prompt=prompt,
                system_message=system_message,
                model=model,
                temperature=temperature,
                custom_id=custom_id,
            )
        )

    write_gemini_batch_jsonl(requests, Path(output_path))


def submit_batch_input(
    file_path: str,
    model: str = DEFAULT_MODEL,
    block: bool = False,
    poll_interval_seconds: int = 60,
    timeout_seconds: int = 24 * 60 * 60,
) -> str:
    """Submit batch input file and return batch job name."""
    batch_name = helper_submit_batch_input(
        file_path,
        model=model,
        block=block,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
    )
    return batch_name


def block_until_completion(
    batch_name: str,
    poll_interval_seconds: int = 60,
    timeout_seconds: int = 24 * 60 * 60,
) -> str:
    """Wait for batch job to complete."""
    return helper_block_until_completion(
        batch_name,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
    )


def download_batch_outputs(batch_name: str, output_path: str) -> str:
    """Download batch outputs to specified path."""
    result_path = helper_download_batch_outputs(batch_name, Path(output_path))
    return str(result_path)


def generate_code_prompt(template: str, algorithm: str) -> str:
    """Substitute the algorithm into the coder prompt template."""
    return template.replace("ALGORITHM_PLACEHOLDER", algorithm)


def _strip_markdown_code_block(text: str) -> str:
    """Strip markdown code block wrappers (```json...```) from text."""
    pattern = r"^```(?:json)?\s*\n?(.*?)\n?```\s*$"
    match = re.match(pattern, text.strip(), re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_text_from_gemini_response(response: Dict[str, Any]) -> str:
    """
    Extract text content from a Gemini batch response.
    Handles the Gemini response structure: candidates -> content -> parts -> text
    """
    if not isinstance(response, dict):
        return str(response)

    # Check for error
    if "error" in response:
        logger.warning(f"Gemini response contains error: {response['error']}")
        return ""

    # Get the response object (may be nested under 'response' key)
    resp_obj = response.get("response", response)

    # Try direct text field
    if isinstance(resp_obj, dict) and "text" in resp_obj:
        return resp_obj["text"]

    # Navigate Gemini's candidate structure
    candidates = None
    if isinstance(resp_obj, dict):
        candidates = resp_obj.get("candidates")

    if candidates and isinstance(candidates, list):
        for candidate in candidates:
            content = candidate.get("content") if isinstance(candidate, dict) else None
            if content and isinstance(content, dict):
                parts = content.get("parts", [])
                for part in parts:
                    if isinstance(part, dict) and "text" in part:
                        return part["text"]

    # Fallback: try to find any text field recursively
    def find_text(obj):
        if isinstance(obj, dict):
            if "text" in obj and isinstance(obj["text"], str):
                return obj["text"]
            for v in obj.values():
                result = find_text(v)
                if result:
                    return result
        elif isinstance(obj, list):
            for item in obj:
                result = find_text(item)
                if result:
                    return result
        return None

    found = find_text(resp_obj)
    if found:
        return found

    return json.dumps(response, ensure_ascii=False)


def parse_algorithm_response(response: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """
    Parse algorithm response from Gemini batch API.

    Returns:
        Tuple of (algorithm_json_str, target_function)
        - algorithm_json_str: JSON string of the algorithm spec
        - target_function: Target function name (None if not specified)
    """
    from llmsat.data.algorithm_parse import parse_algorithm_spec_json

    raw_text = _extract_text_from_gemini_response(response)
    raw_text = _strip_markdown_code_block(raw_text)

    try:
        spec, target_function = parse_algorithm_spec_json(raw_text)
        # Remove optional fields not part of core spec
        if isinstance(spec, dict):
            spec.pop("Reason", None)
            spec.pop("reason", None)
        return json.dumps(spec, ensure_ascii=False), target_function
    except Exception as e:
        logger.warning(f"Failed to parse algorithm response: {e}")
        return raw_text, None


def parse_code_response(response: Dict[str, Any]) -> str:
    """
    Parse code response from Gemini batch API.
    Extracts code from <code>...</code> tags or returns full text.
    """
    full_text = _extract_text_from_gemini_response(response)

    # Try to extract <code>...</code> block
    start = full_text.find("<code>")
    end = full_text.find("</code>")
    if start != -1 and end != -1 and end > start:
        return full_text[start + len("<code>") : end].strip()

    # Try markdown code block
    if "```c" in full_text:
        start = full_text.find("```c") + 4
        end = full_text.find("```", start)
        if end > start:
            return full_text[start:end].strip()

    return full_text


def generate_team_data(
    designer_prompt_path: str,
    variant_prompt_path: str,
    code_prompt_template_path: str,
    generation_tag: str,
    n_leaders: int = 5,
    m_variants_per_leader: int = 3,
    model: str = DEFAULT_MODEL,
):
    """
    Generate team-based algorithm data and code using Gemini:
    1. Generate n_leaders Team Leader strategies
    2. For each leader, generate m_variants_per_leader Team Member variants
    3. Generate code for all algorithms

    Args:
        designer_prompt_path: Path to the designer prompt for Team Leaders
        variant_prompt_path: Path to variant prompt template (uses {leader_algorithm} placeholder)
        code_prompt_template_path: Path to code prompt template
        generation_tag: Tag for this generation run
        n_leaders: Number of Team Leader strategies to generate
        m_variants_per_leader: Number of Team Member variants per leader
        model: Gemini model to use
    """
    if generation_tag is None:
        logger.error("Generation tag is None")
        return

    designer_prompt = read_prompt_file(designer_prompt_path)
    variant_prompt_template = read_prompt_file(variant_prompt_path)

    # Step 1: Generate Team Leaders
    logger.info(f"Generating {n_leaders} Team Leaders with model {model}")
    leader_batch_input_path = os.path.join(
        get_generation_output_dir(generation_tag), "leader_batch_input.txt"
    )
    create_batch_input_file(
        designer_prompt, leader_batch_input_path, n_requests=n_leaders, model=model
    )

    leader_batch_name = submit_batch_input(leader_batch_input_path, model=model)
    block_until_completion(leader_batch_name)

    leaders_output_path = os.path.join(
        get_batch_output_dir(generation_tag, batch_id=leader_batch_name),
        "leaders_output.txt",
    )
    download_batch_outputs(leader_batch_name, leaders_output_path)

    # Parse and store Team Leaders
    leader_ids = []
    leader_target_functions = {}

    with open(leaders_output_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                leader_response = json.loads(line)
            except Exception:
                continue

            leader_str, target_function = parse_algorithm_response(leader_response)
            leader_id = get_id(leader_str)
            leader_ids.append(leader_id)
            leader_target_functions[leader_id] = target_function or "kissat_restarting"

            update_router_table(CHATGPT_DATA_GENERATION_TABLE, leader_id, generation_tag)
            leader_result = AlgorithmResult(
                id=leader_id,
                algorithm=leader_str,
                status=AlgorithmStatus.Generated,
                last_updated=datetime.now(),
                prompt=designer_prompt,
                par2=NOT_INITIALIZED,
                error_rate=NOT_INITIALIZED,
                other_metrics={},
                code_id_list=[],
                parent_id=None,
                target_function=target_function or "kissat_restarting",
            )
            update_algorithm_result(leader_result)

    logger.info(f"Generated {len(leader_ids)} Team Leaders")

    # Step 2: Generate Team Members for each leader
    logger.info(f"Generating {m_variants_per_leader} Team Members per leader")

    waiting_batch_names = []
    batch_name_to_leader_id = {}

    for leader_id in leader_ids:
        leader_result = get_algorithm_result(leader_id)
        leader_algorithm = leader_result.algorithm

        # Count steps in leader algorithm
        num_steps = count_steps(leader_algorithm)
        if num_steps == 0:
            raise ValueError(
                f"Leader {leader_id} has no step markers. Expected 'Step N:' format."
            )

        # Calculate step assignments
        step_assignments = []
        if num_steps < m_variants_per_leader:
            for i in range(m_variants_per_leader):
                if i < num_steps:
                    step_assignments.append(i + 1)
                else:
                    step_assignments.append(num_steps)
        else:
            start_step = num_steps - m_variants_per_leader + 1
            for i in range(m_variants_per_leader):
                step_assignments.append(start_step + i)

        logger.info(
            f"Leader {leader_id[:8]}... has {num_steps} steps, "
            f"assigning variants to steps: {step_assignments}"
        )

        # Build variant prompts
        variant_prompts = []
        for target_step in step_assignments:
            prompt = variant_prompt_template.replace(
                "{leader_algorithm}", leader_algorithm
            )
            prompt = prompt.replace("{target_step_num}", str(target_step))
            variant_prompts.append(prompt)

        member_batch_input_path = os.path.join(
            get_batch_output_dir(generation_tag, batch_id=leader_batch_name),
            f"member_batch_input_{leader_id}.txt",
        )
        create_batch_input_file_variant(variant_prompts, member_batch_input_path, model=model)

        batch_name = submit_batch_input(member_batch_input_path, model=model)
        waiting_batch_names.append(batch_name)
        batch_name_to_leader_id[batch_name] = leader_id

    # Save batch mapping
    batch_id_map = {
        "leader_batch_name": leader_batch_name,
        "member_batch_names": waiting_batch_names,
        "member_batch_map": batch_name_to_leader_id,
    }
    json.dump(
        batch_id_map,
        open(
            os.path.join(
                get_batch_output_dir(generation_tag, batch_id=leader_batch_name),
                f"team_batch_map_{datetime.now().strftime('%Y%m%d%H%M%S')}.json",
            ),
            "w",
        ),
    )

    # Process member batches
    member_ids = []

    for batch_name in waiting_batch_names:
        block_until_completion(batch_name)

        leader_id = batch_name_to_leader_id[batch_name]
        member_output_path = os.path.join(
            get_batch_output_dir(generation_tag, batch_id=leader_batch_name),
            f"member_output_{batch_name}.txt",
        )
        download_batch_outputs(batch_name, member_output_path)

        leader_target_function = leader_target_functions.get(
            leader_id, "kissat_restarting"
        )

        with open(member_output_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    member_response = json.loads(line)
                except Exception:
                    continue

                member_str, _ = parse_algorithm_response(member_response)
                member_id = get_id(member_str)
                member_ids.append(member_id)

                update_router_table(
                    CHATGPT_DATA_GENERATION_TABLE, member_id, generation_tag
                )
                member_result = AlgorithmResult(
                    id=member_id,
                    algorithm=member_str,
                    status=AlgorithmStatus.Generated,
                    last_updated=datetime.now(),
                    prompt=variant_prompt_template,
                    par2=NOT_INITIALIZED,
                    error_rate=NOT_INITIALIZED,
                    other_metrics={},
                    code_id_list=[],
                    parent_id=leader_id,
                    target_function=leader_target_function,
                )
                update_algorithm_result(member_result)

        logger.info(f"Generated members for leader {leader_id}")

    logger.info(
        f"Team generation complete: {len(leader_ids)} leaders, {len(member_ids)} members"
    )

    # Step 3: Generate code for all algorithms
    all_algorithm_ids = leader_ids + member_ids
    logger.info(f"Starting code generation for {len(all_algorithm_ids)} algorithms")

    code_prompt_template = read_prompt_file(code_prompt_template_path)

    code_batch_names = []
    code_batch_to_algorithm = {}

    for algorithm_id in all_algorithm_ids:
        algorithm_result = get_algorithm_result(algorithm_id)
        code_prompt = generate_code_prompt(code_prompt_template, algorithm_result.algorithm)

        code_batch_input_path = os.path.join(
            get_batch_output_dir(generation_tag, batch_id=leader_batch_name),
            f"code_batch_input_{algorithm_id}.txt",
        )
        create_batch_input_file(code_prompt, code_batch_input_path, n_requests=1, model=model)

        batch_name = submit_batch_input(code_batch_input_path, model=model)
        code_batch_names.append(batch_name)
        code_batch_to_algorithm[batch_name] = algorithm_id

    # Update batch map
    batch_id_map["code_batch_names"] = code_batch_names
    batch_id_map["code_batch_map"] = code_batch_to_algorithm
    json.dump(
        batch_id_map,
        open(
            os.path.join(
                get_batch_output_dir(generation_tag, batch_id=leader_batch_name),
                f"team_batch_map_{datetime.now().strftime('%Y%m%d%H%M%S')}.json",
            ),
            "w",
        ),
    )

    logger.info(f"Submitted {len(code_batch_names)} code generation batches")

    # Wait for ALL code batches to complete (parallel polling)
    logger.info("Waiting for all code batches to complete (parallel polling)...")
    wait_for_all_batches(code_batch_names)
    logger.info("All code batches completed, downloading results...")

    # Download and process all results
    for batch_name in code_batch_names:
        algorithm_id = code_batch_to_algorithm[batch_name]
        code_output_path = os.path.join(
            get_batch_output_dir(generation_tag, batch_id=leader_batch_name),
            f"code_output_{batch_name}.txt",
        )
        download_batch_outputs(batch_name, code_output_path)

        algorithm_result = get_algorithm_result(algorithm_id)

        with open(code_output_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    code_response = json.loads(line)
                except Exception:
                    continue

                code_str = parse_code_response(code_response)
                code_id = get_id(code_str)

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

                if algorithm_result.code_id_list is None:
                    algorithm_result.code_id_list = []
                algorithm_result.code_id_list.append(code_id)

        algorithm_result.status = AlgorithmStatus.CodeGenerated
        update_algorithm_result(algorithm_result)
        logger.info(f"Generated code for algorithm {algorithm_id}")

    logger.info(f"Code generation complete for {len(all_algorithm_ids)} algorithms")


def main():
    generate_team_data(
        generation_tag="gemini_test",
        designer_prompt_path="./data/prompts/leader_prompt_testing.txt",
        variant_prompt_path="./data/prompts/variant_prompt.txt",
        code_prompt_template_path="./data/prompts/coder_prompt_testing.txt",
        n_leaders=2,
        m_variants_per_leader=4,
        model="gemini-3-pro-preview",
    )


if __name__ == "__main__":
    main()
