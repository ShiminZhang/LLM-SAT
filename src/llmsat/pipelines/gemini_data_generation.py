import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from llmsat.utils.gemini_helper import (
    build_gemini_batch_request,
    write_gemini_batch_jsonl,
    submit_batch_input as helper_submit_batch_input,
    block_until_completion as helper_block_until_completion,
    download_batch_outputs as helper_download_batch_outputs,
    wait_for_all_batches,
    get_response_from_gemini,
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
    Role,
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

from llmsat.config import DEFAULT_MODEL


def read_prompt_file(path: str) -> str:
    """Read a prompt template from file."""
    with open(path, "r") as f:
        return f.read()


def create_batch_input_file(
    prompt: str,
    output_path: str,
    n_requests: int = 10,
    model: str = DEFAULT_MODEL,
    temperature_min: float = None,
    temperature_max: float = None,
):
    """Create a Gemini batch input JSONL file with identical prompts.

    If temperature_min and temperature_max are provided, temperature is linearly
    interpolated across requests. Otherwise uses GEMINI_TEMPERATURE env var or 0.7.
    """
    logger.info(f"Creating batch input file for {n_requests} requests")

    system_message = os.environ.get(
        "LLMSAT_SYSTEM_MESSAGE",
        "You are an AI researcher specialising in SAT solver heuristics.",
    )
    model = os.environ.get("GEMINI_MODEL", model)

    # Determine temperature strategy
    if temperature_min is not None and temperature_max is not None:
        use_range = True
    else:
        use_range = False
        try:
            temperature = float(os.environ.get("GEMINI_TEMPERATURE", "0.7"))
        except Exception:
            temperature = 0.7

    requests = []
    for i in range(1, int(n_requests) + 1):
        custom_id = f"req-{i:04d}"
        if use_range:
            # Linear interpolation from min to max
            t = (i - 1) / max(n_requests - 1, 1)
            temp = temperature_min + (temperature_max - temperature_min) * t
        else:
            temp = temperature
        requests.append(
            build_gemini_batch_request(
                prompt=prompt,
                system_message=system_message,
                model=model,
                temperature=temp,
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


def generate_code_prompt(template: str, algorithm: str, function_name: str = "restart_mab") -> str:
    """Substitute the algorithm and target function into the coder prompt template."""
    prompt = template.replace("ALGORITHM_PLACEHOLDER", algorithm)
    prompt = prompt.replace("FUNCTION_NAME_PLACEHOLDER", function_name)
    return prompt


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


def parse_algorithm_response(response: Dict[str, Any]) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Parse algorithm response from Gemini batch API.

    Returns:
        Tuple of (description, target_function, reason)
        - description: "Name: algorithm text" human-readable description
        - target_function: Target function name (None if not specified)
        - reason: LLM's reasoning for why this algorithm is good (None if absent)
    """
    from llmsat.data.algorithm_parse import parse_algorithm_spec_json

    raw_text = _extract_text_from_gemini_response(response)
    raw_text = _strip_markdown_code_block(raw_text)

    try:
        spec, target_function = parse_algorithm_spec_json(raw_text)
        reason = None
        if isinstance(spec, dict):
            reason = spec.pop("Reason", None) or spec.pop("reason", None)
            # Build description as "Name: algorithm text"
            name = spec.get("name", "")
            algo_text = spec.get("algorithm", "")
            description = f"{name}: {algo_text}" if name else algo_text
        else:
            description = str(spec)
        return description, target_function, reason
    except Exception as e:
        logger.warning(f"Failed to parse algorithm response: {e}")
        return raw_text, None, None


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


def _save_timing_log(timing: Dict[str, Any], output_dir: str, filename: str = "timing_log.json") -> None:
    """Append a timing record to the timing log JSON in output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    records = []
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                records = json.load(f)
            if not isinstance(records, list):
                records = [records]
        except Exception:
            records = []
    records.append(timing)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    logger.info(f"[TIMING] Saved timing log to {path}")


def _generate_team_data_sync(
    designer_prompt_path: str,
    variant_prompt_path: str,
    code_prompt_template_path: str,
    generation_tag: str,
    n_leaders: int = 5,
    m_variants_per_leader: int = 3,
    model: str = DEFAULT_MODEL,
):
    """Synchronous implementation of generate_team_data — no batch API, no waiting."""
    designer_prompt = read_prompt_file(designer_prompt_path)
    variant_prompt_template = read_prompt_file(variant_prompt_path)
    code_prompt_template = read_prompt_file(code_prompt_template_path)

    system_message = os.environ.get(
        "LLMSAT_SYSTEM_MESSAGE",
        "You are an AI researcher specialising in SAT solver heuristics.",
    )

    _t_total_start = time.time()
    _timing: Dict[str, Any] = {
        "generation_tag": generation_tag,
        "timestamp": datetime.now().isoformat(),
        "script": "gemini_data_generation",
    }

    # Step 1: Generate Team Leaders
    logger.info(f"[sync] Generating {n_leaders} Team Leaders")
    leader_ids = []
    leader_target_functions = {}
    leader_descriptions = {}

    temperatures = [
        0.5 + (1.0 - 0.5) * i / max(n_leaders - 1, 1) for i in range(n_leaders)
    ]

    _t0 = time.time()
    for i in range(n_leaders):
        logger.info(f"[sync] Leader {i+1}/{n_leaders} (temp={temperatures[i]:.2f})")
        raw_text = get_response_from_gemini(
            designer_prompt,
            system_message=system_message,
            model=model,
            temperature=temperatures[i],
        )
        description, target_function, reason = parse_algorithm_response({"text": raw_text})
        leader_id = get_id(description)
        fn = target_function or "kissat_restarting"

        leader_ids.append(leader_id)
        leader_target_functions[leader_id] = fn
        leader_descriptions[leader_id] = description

        update_router_table(CHATGPT_DATA_GENERATION_TABLE, leader_id, generation_tag)
        update_algorithm_result(AlgorithmResult(
            id=leader_id,
            function_name=fn,
            description=description,
            role=Role.LEADER,
            status=AlgorithmStatus.Generated,
            last_updated=datetime.now(),
            code_id_list=[],
            parent_id=None,
            analysis=reason,
            prompt=designer_prompt,
        ))

    _timing["leader_generation_s"] = round(time.time() - _t0, 2)
    logger.info(f"[TIMING] Leader generation: {_timing['leader_generation_s']}s")
    logger.info(f"[sync] Generated {len(leader_ids)} Team Leaders")

    # Step 2: Generate Team Members
    logger.info(f"[sync] Generating {m_variants_per_leader} Team Members per leader")
    member_ids = []

    _t0 = time.time()
    for leader_id in leader_ids:
        leader_algorithm = leader_descriptions[leader_id]
        num_steps = count_steps(leader_algorithm)
        if num_steps == 0:
            raise ValueError(
                f"Leader {leader_id} has no step markers. Expected 'Step N:' format."
            )

        if num_steps < m_variants_per_leader:
            step_assignments = [
                (i + 1) if i < num_steps else num_steps
                for i in range(m_variants_per_leader)
            ]
        else:
            start_step = num_steps - m_variants_per_leader + 1
            step_assignments = [start_step + i for i in range(m_variants_per_leader)]

        for j, target_step in enumerate(step_assignments):
            logger.info(
                f"[sync] Member {j+1}/{m_variants_per_leader} for leader {leader_id[:8]}... (step {target_step})"
            )
            prompt = variant_prompt_template.replace("{leader_algorithm}", leader_algorithm)
            prompt = prompt.replace("{target_step_num}", str(target_step))

            raw_text = get_response_from_gemini(
                prompt, system_message=system_message, model=model
            )
            member_desc, _, member_reason = parse_algorithm_response({"text": raw_text})
            member_id = get_id(member_desc)
            member_ids.append(member_id)

            update_router_table(CHATGPT_DATA_GENERATION_TABLE, member_id, generation_tag)
            update_algorithm_result(AlgorithmResult(
                id=member_id,
                function_name=leader_target_functions[leader_id],
                description=member_desc,
                role=Role.MEMBER,
                status=AlgorithmStatus.Generated,
                last_updated=datetime.now(),
                code_id_list=[],
                parent_id=[leader_id],
                parent_algorithm_description=[leader_descriptions.get(leader_id, "")],
                analysis=member_reason,
                prompt=variant_prompt_template,
            ))

    _timing["member_generation_s"] = round(time.time() - _t0, 2)
    logger.info(f"[TIMING] Member generation: {_timing['member_generation_s']}s")
    logger.info(f"[sync] Generated {len(member_ids)} Team Members")

    # Step 3: Generate code for all algorithms that don't have code yet
    all_algorithm_ids = leader_ids + member_ids
    codeless_ids = []
    for aid in all_algorithm_ids:
        ar = get_algorithm_result(aid)
        if ar and not (ar.code_id_list and len(ar.code_id_list) > 0):
            codeless_ids.append(aid)

    logger.info(f"[sync] Generating code for {len(codeless_ids)}/{len(all_algorithm_ids)} algorithms (skipping {len(all_algorithm_ids) - len(codeless_ids)} with existing code)")

    _t0 = time.time()
    for idx, algorithm_id in enumerate(codeless_ids):
        logger.info(f"[sync] Code {idx+1}/{len(codeless_ids)} for {algorithm_id[:16]}...")
        algorithm_result = get_algorithm_result(algorithm_id)
        code_prompt = generate_code_prompt(code_prompt_template, algorithm_result.description, algorithm_result.function_name)

        raw_text = get_response_from_gemini(
            code_prompt, system_message=system_message, model=model
        )
        code_str = parse_code_response({"text": raw_text})
        code_id = get_id(code_str)

        update_code_result(CodeResult(
            id=code_id,
            algorithm_id=algorithm_id,
            code=code_str,
            status=CodeStatus.Generated,
            par2=None,
            last_updated=datetime.now(),
            build_success=NOT_INITIALIZED,
        ))

        if algorithm_result.code_id_list is None:
            algorithm_result.code_id_list = []
        algorithm_result.code_id_list.append(code_id)
        algorithm_result.status = AlgorithmStatus.CodeGenerated
        update_algorithm_result(algorithm_result)

    _timing["code_generation_s"] = round(time.time() - _t0, 2)
    _timing["total_s"] = round(time.time() - _t_total_start, 2)
    logger.info(f"[TIMING] Code generation: {_timing['code_generation_s']}s")
    logger.info(f"[TIMING] Total: {_timing['total_s']}s")
    output_dir = get_generation_output_dir(generation_tag)
    _save_timing_log(_timing, output_dir)

    logger.info(f"[sync] Code generation complete for {len(codeless_ids)} algorithms")


def generate_team_data(
    designer_prompt_path: str,
    variant_prompt_path: str,
    code_prompt_template_path: str,
    generation_tag: str,
    n_leaders: int = 5,
    m_variants_per_leader: int = 3,
    model: str = DEFAULT_MODEL,
    sync: bool = False,
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
        sync: If True, use synchronous API calls instead of batch (faster for small runs)
    """
    if generation_tag is None:
        logger.error("Generation tag is None")
        return

    if sync:
        return _generate_team_data_sync(
            designer_prompt_path=designer_prompt_path,
            variant_prompt_path=variant_prompt_path,
            code_prompt_template_path=code_prompt_template_path,
            generation_tag=generation_tag,
            n_leaders=n_leaders,
            m_variants_per_leader=m_variants_per_leader,
            model=model,
        )

    designer_prompt = read_prompt_file(designer_prompt_path)
    variant_prompt_template = read_prompt_file(variant_prompt_path)

    # Step 1: Generate Team Leaders
    logger.info(f"Generating {n_leaders} Team Leaders with model {model}")
    leader_batch_input_path = os.path.join(
        get_generation_output_dir(generation_tag), "leader_batch_input.txt"
    )
    create_batch_input_file(
        designer_prompt,
        leader_batch_input_path,
        n_requests=n_leaders,
        model=model,
        temperature_min=0.5,
        temperature_max=1.0,
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
    leader_descriptions = {}

    with open(leaders_output_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                leader_response = json.loads(line)
            except Exception:
                continue

            description, target_function, reason = parse_algorithm_response(leader_response)
            leader_id = get_id(description)
            leader_ids.append(leader_id)
            fn = target_function or "kissat_restarting"
            leader_target_functions[leader_id] = fn
            leader_descriptions[leader_id] = description

            update_router_table(CHATGPT_DATA_GENERATION_TABLE, leader_id, generation_tag)
            leader_result = AlgorithmResult(
                id=leader_id,
                function_name=fn,
                description=description,
                role=Role.LEADER,
                status=AlgorithmStatus.Generated,
                last_updated=datetime.now(),
                code_id_list=[],
                parent_id=None,
                analysis=reason,
                prompt=designer_prompt,
            )
            update_algorithm_result(leader_result)

    logger.info(f"Generated {len(leader_ids)} Team Leaders")

    # Step 2: Generate Team Members for each leader
    logger.info(f"Generating {m_variants_per_leader} Team Members per leader")

    waiting_batch_names = []
    batch_name_to_leader_id = {}

    for leader_id in leader_ids:
        leader_result = get_algorithm_result(leader_id)
        leader_algorithm = leader_result.description

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

                member_desc, _, member_reason = parse_algorithm_response(member_response)
                member_id = get_id(member_desc)
                member_ids.append(member_id)

                update_router_table(
                    CHATGPT_DATA_GENERATION_TABLE, member_id, generation_tag
                )
                member_result = AlgorithmResult(
                    id=member_id,
                    function_name=leader_target_function,
                    description=member_desc,
                    role=Role.MEMBER,
                    status=AlgorithmStatus.Generated,
                    last_updated=datetime.now(),
                    code_id_list=[],
                    parent_id=[leader_id],
                    parent_algorithm_description=[leader_descriptions.get(leader_id, "")],
                    analysis=member_reason,
                    prompt=variant_prompt_template,
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
        code_prompt = generate_code_prompt(code_prompt_template, algorithm_result.description, algorithm_result.function_name)

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


def _generate_mutants_sync(
    leaders: Dict[str, AlgorithmResult],
    variant_prompt_template: str,
    code_prompt_template: str,
    generation_tag: str,
    m_variants_per_leader: int = 3,
    model: str = DEFAULT_MODEL,
):
    """Sync implementation: generate mutant variants + code for existing leaders.

    Resumable: skips leaders that already have enough members under this
    generation_tag, and skips code generation for algorithms that already
    have code.
    """
    system_message = os.environ.get(
        "LLMSAT_SYSTEM_MESSAGE",
        "You are an AI researcher specialising in SAT solver heuristics.",
    )

    # Build map of existing members per leader in this generation tag
    all_tag_ids = get_ids_from_router_table(CHATGPT_DATA_GENERATION_TABLE, generation_tag)
    existing_members_by_leader: Dict[str, int] = {}
    for aid in all_tag_ids:
        ar = get_algorithm_result(aid)
        if ar and ar.role == Role.MEMBER:
            parent = ar.parent_id[0] if isinstance(ar.parent_id, list) else ar.parent_id
            existing_members_by_leader[parent] = existing_members_by_leader.get(parent, 0) + 1

    member_ids = []

    for leader_id, leader_result in leaders.items():
        existing_count = existing_members_by_leader.get(leader_id, 0)
        if existing_count >= m_variants_per_leader:
            logger.info(
                f"Leader {leader_id[:8]}... already has {existing_count} members, skipping"
            )
            continue

        leader_algorithm = leader_result.description
        num_steps = count_steps(leader_algorithm)
        if num_steps == 0:
            logger.warning(
                f"Leader {leader_id[:8]}... has no step markers, skipping"
            )
            continue

        needed = m_variants_per_leader - existing_count
        if num_steps < m_variants_per_leader:
            step_assignments = [
                (i + 1) if i < num_steps else num_steps
                for i in range(m_variants_per_leader)
            ]
        else:
            start_step = num_steps - m_variants_per_leader + 1
            step_assignments = [start_step + i for i in range(m_variants_per_leader)]

        # Only generate the remaining variants (skip already-generated ones)
        step_assignments = step_assignments[existing_count:]

        logger.info(
            f"Leader {leader_id[:8]}... has {num_steps} steps, "
            f"{existing_count} existing members, generating {needed} more "
            f"(steps: {step_assignments})"
        )

        for j, target_step in enumerate(step_assignments):
            logger.info(
                f"[sync] Member {existing_count+j+1}/{m_variants_per_leader} for leader {leader_id[:8]}... (step {target_step})"
            )
            prompt = variant_prompt_template.replace("{leader_algorithm}", leader_algorithm)
            prompt = prompt.replace("{target_step_num}", str(target_step))

            raw_text = get_response_from_gemini(
                prompt, system_message=system_message, model=model
            )
            member_desc, _, member_reason = parse_algorithm_response({"text": raw_text})
            member_id = get_id(member_desc)
            member_ids.append(member_id)

            update_router_table(CHATGPT_DATA_GENERATION_TABLE, member_id, generation_tag)
            update_algorithm_result(AlgorithmResult(
                id=member_id,
                function_name=leader_result.function_name,
                description=member_desc,
                role=Role.MEMBER,
                status=AlgorithmStatus.Generated,
                last_updated=datetime.now(),
                code_id_list=[],
                parent_id=[leader_id],
                parent_algorithm_description=[leader_algorithm],
                analysis=member_reason,
                prompt=variant_prompt_template,
            ))

    logger.info(f"[sync] Generated {len(member_ids)} new mutants")

    # Generate code for all members (new + previously code-less) under this tag
    all_tag_ids = get_ids_from_router_table(CHATGPT_DATA_GENERATION_TABLE, generation_tag)
    codeless_ids = []
    for aid in all_tag_ids:
        ar = get_algorithm_result(aid)
        if ar and ar.role == Role.MEMBER and not (ar.code_id_list and len(ar.code_id_list) > 0):
            codeless_ids.append(aid)

    logger.info(f"[sync] Generating code for {len(codeless_ids)} codeless mutants")
    for idx, mid in enumerate(codeless_ids):
        logger.info(f"[sync] Code {idx+1}/{len(codeless_ids)} for {mid[:16]}...")
        algorithm_result = get_algorithm_result(mid)
        code_prompt = generate_code_prompt(code_prompt_template, algorithm_result.description, algorithm_result.function_name)

        raw_text = get_response_from_gemini(
            code_prompt, system_message=system_message, model=model
        )
        code_str = parse_code_response({"text": raw_text})
        code_id = get_id(code_str)

        update_code_result(CodeResult(
            id=code_id,
            algorithm_id=mid,
            code=code_str,
            status=CodeStatus.Generated,
            par2=None,
            last_updated=datetime.now(),
            build_success=NOT_INITIALIZED,
        ))

        if algorithm_result.code_id_list is None:
            algorithm_result.code_id_list = []
        algorithm_result.code_id_list.append(code_id)
        algorithm_result.status = AlgorithmStatus.CodeGenerated
        update_algorithm_result(algorithm_result)

    logger.info(f"[sync] Mutant generation complete: {len(member_ids)} new mutants, {len(codeless_ids)} code generated")
    return member_ids


def _generate_mutants_batch(
    leaders: Dict[str, AlgorithmResult],
    variant_prompt_template: str,
    code_prompt_template: str,
    generation_tag: str,
    m_variants_per_leader: int = 3,
    model: str = DEFAULT_MODEL,
):
    """Batch implementation: generate mutant variants + code for existing leaders."""

    # Submit variant generation batches
    waiting_batch_names = []
    batch_name_to_leader_id = {}

    for leader_id, leader_result in leaders.items():
        leader_algorithm = leader_result.description
        num_steps = count_steps(leader_algorithm)
        if num_steps == 0:
            logger.warning(f"Leader {leader_id[:8]}... has no step markers, skipping")
            continue

        step_assignments = []
        if num_steps < m_variants_per_leader:
            for i in range(m_variants_per_leader):
                step_assignments.append((i + 1) if i < num_steps else num_steps)
        else:
            start_step = num_steps - m_variants_per_leader + 1
            for i in range(m_variants_per_leader):
                step_assignments.append(start_step + i)

        logger.info(
            f"Leader {leader_id[:8]}... has {num_steps} steps, "
            f"assigning variants to steps: {step_assignments}"
        )

        variant_prompts = []
        for target_step in step_assignments:
            prompt = variant_prompt_template.replace("{leader_algorithm}", leader_algorithm)
            prompt = prompt.replace("{target_step_num}", str(target_step))
            variant_prompts.append(prompt)

        member_batch_input_path = os.path.join(
            get_generation_output_dir(generation_tag),
            f"member_batch_input_{leader_id}.txt",
        )
        create_batch_input_file_variant(variant_prompts, member_batch_input_path, model=model)

        batch_name = submit_batch_input(member_batch_input_path, model=model)
        waiting_batch_names.append(batch_name)
        batch_name_to_leader_id[batch_name] = leader_id

    # Save batch mapping
    batch_id_map = {
        "member_batch_names": waiting_batch_names,
        "member_batch_map": batch_name_to_leader_id,
    }
    json.dump(
        batch_id_map,
        open(
            os.path.join(
                get_generation_output_dir(generation_tag),
                f"mutant_batch_map_{datetime.now().strftime('%Y%m%d%H%M%S')}.json",
            ),
            "w",
        ),
    )

    # Process member batches
    member_ids = []

    for batch_name in waiting_batch_names:
        block_until_completion(batch_name)

        leader_id = batch_name_to_leader_id[batch_name]
        leader_result = leaders[leader_id]
        member_output_path = os.path.join(
            get_generation_output_dir(generation_tag),
            f"member_output_{batch_name}.txt",
        )
        download_batch_outputs(batch_name, member_output_path)

        with open(member_output_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    member_response = json.loads(line)
                except Exception:
                    continue

                member_desc, _, member_reason = parse_algorithm_response(member_response)
                member_id = get_id(member_desc)
                member_ids.append(member_id)

                update_router_table(
                    CHATGPT_DATA_GENERATION_TABLE, member_id, generation_tag
                )
                update_algorithm_result(AlgorithmResult(
                    id=member_id,
                    function_name=leader_result.function_name,
                    description=member_desc,
                    role=Role.MEMBER,
                    status=AlgorithmStatus.Generated,
                    last_updated=datetime.now(),
                    code_id_list=[],
                    parent_id=[leader_id],
                    parent_algorithm_description=[leader_result.description],
                    analysis=member_reason,
                    prompt=variant_prompt_template,
                ))

        logger.info(f"Generated members for leader {leader_id}")

    logger.info(f"Generated {len(member_ids)} mutants")

    # Generate code for mutants only
    logger.info(f"Starting code generation for {len(member_ids)} mutants")

    code_batch_names = []
    code_batch_to_algorithm = {}

    for member_id in member_ids:
        algorithm_result = get_algorithm_result(member_id)
        code_prompt = generate_code_prompt(code_prompt_template, algorithm_result.description, algorithm_result.function_name)

        code_batch_input_path = os.path.join(
            get_generation_output_dir(generation_tag),
            f"code_batch_input_{member_id}.txt",
        )
        create_batch_input_file(code_prompt, code_batch_input_path, n_requests=1, model=model)

        batch_name = submit_batch_input(code_batch_input_path, model=model)
        code_batch_names.append(batch_name)
        code_batch_to_algorithm[batch_name] = member_id

    # Update batch map
    batch_id_map["code_batch_names"] = code_batch_names
    batch_id_map["code_batch_map"] = code_batch_to_algorithm
    json.dump(
        batch_id_map,
        open(
            os.path.join(
                get_generation_output_dir(generation_tag),
                f"mutant_batch_map_{datetime.now().strftime('%Y%m%d%H%M%S')}.json",
            ),
            "w",
        ),
    )

    logger.info(f"Submitted {len(code_batch_names)} code generation batches")
    logger.info("Waiting for all code batches to complete...")
    wait_for_all_batches(code_batch_names)
    logger.info("All code batches completed, downloading results...")

    for batch_name in code_batch_names:
        member_id = code_batch_to_algorithm[batch_name]
        code_output_path = os.path.join(
            get_generation_output_dir(generation_tag),
            f"code_output_{batch_name}.txt",
        )
        download_batch_outputs(batch_name, code_output_path)

        algorithm_result = get_algorithm_result(member_id)

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

                update_code_result(CodeResult(
                    id=code_id,
                    algorithm_id=member_id,
                    code=code_str,
                    status=CodeStatus.Generated,
                    par2=None,
                    last_updated=datetime.now(),
                    build_success=NOT_INITIALIZED,
                ))

                if algorithm_result.code_id_list is None:
                    algorithm_result.code_id_list = []
                algorithm_result.code_id_list.append(code_id)

        algorithm_result.status = AlgorithmStatus.CodeGenerated
        update_algorithm_result(algorithm_result)
        logger.info(f"Generated code for mutant {member_id}")

    logger.info(f"Mutant generation complete: {len(member_ids)} mutants")
    return member_ids


def generate_mutants_for_leaders(
    variant_prompt_path: str,
    code_prompt_template_path: str,
    source_generation_tag: str,
    output_generation_tag: str,
    m_variants_per_leader: int = 3,
    model: str = DEFAULT_MODEL,
    sync: bool = False,
):
    """
    Generate mutant variants + code for existing leaders (no new leader generation).

    Loads leaders from source_generation_tag, registers them under output_generation_tag,
    then generates M mutant variants per leader with code.

    Args:
        variant_prompt_path: Path to variant prompt template
        code_prompt_template_path: Path to code prompt template
        source_generation_tag: Tag to load existing leaders from
        output_generation_tag: Tag for this iteration's output
        m_variants_per_leader: Number of mutant variants per leader
        model: Gemini model to use
        sync: If True, use synchronous API calls instead of batch
    """
    if output_generation_tag is None:
        logger.error("output_generation_tag is required")
        return

    # Load leaders from source tag
    logger.info(f"Loading leaders from source tag: {source_generation_tag}")
    all_ids = get_ids_from_router_table(CHATGPT_DATA_GENERATION_TABLE, source_generation_tag)
    logger.info(f"Found {len(all_ids)} algorithms under {source_generation_tag}")

    leaders = {}
    for algorithm_id in all_ids:
        result = get_algorithm_result(algorithm_id)
        if result is None:
            logger.warning(f"Algorithm {algorithm_id[:16]}... not found in DB, skipping")
            continue
        if result.role != Role.LEADER:
            continue  # Skip non-leaders
        leaders[algorithm_id] = result

    logger.info(f"Loaded {len(leaders)} leaders")
    if not leaders:
        logger.error("No leaders found, nothing to do")
        return

    # Read prompt templates
    variant_prompt_template = read_prompt_file(variant_prompt_path)
    code_prompt_template = read_prompt_file(code_prompt_template_path)

    if sync:
        result = _generate_mutants_sync(
            leaders=leaders,
            variant_prompt_template=variant_prompt_template,
            code_prompt_template=code_prompt_template,
            generation_tag=output_generation_tag,
            m_variants_per_leader=m_variants_per_leader,
            model=model,
        )
    else:
        result = _generate_mutants_batch(
            leaders=leaders,
            variant_prompt_template=variant_prompt_template,
            code_prompt_template=code_prompt_template,
            generation_tag=output_generation_tag,
            m_variants_per_leader=m_variants_per_leader,
            model=model,
        )

    # Register leaders under the new output tag only after successful mutation
    for leader_id in leaders:
        update_router_table(CHATGPT_DATA_GENERATION_TABLE, leader_id, output_generation_tag)
    logger.info(f"Registered {len(leaders)} leaders under {output_generation_tag}")

    return result


def resume_code_collection(generation_tag: str, batch_map_path: str):
    """
    Resume code collection from a saved batch map after interruption.

    Use this when the generation was interrupted during wait_for_all_batches()
    but the batches have since completed on Gemini's side.

    Args:
        generation_tag: The generation tag used in the original run
        batch_map_path: Path to the team_batch_map JSON file
    """
    logger.info(f"Resuming code collection for generation_tag={generation_tag}")

    with open(batch_map_path, "r") as f:
        batch_id_map = json.load(f)

    leader_batch_name = batch_id_map["leader_batch_name"]
    code_batch_names = batch_id_map.get("code_batch_names", [])
    code_batch_to_algorithm = batch_id_map.get("code_batch_map", {})

    if not code_batch_names:
        logger.error("No code_batch_names found in batch map")
        return

    logger.info(f"Found {len(code_batch_names)} code batches to collect")

    # Download and process all results
    for i, batch_name in enumerate(code_batch_names):
        algorithm_id = code_batch_to_algorithm.get(batch_name)
        if not algorithm_id:
            logger.warning(f"No algorithm_id found for batch {batch_name}")
            continue

        code_output_path = os.path.join(
            get_batch_output_dir(generation_tag, batch_id=leader_batch_name),
            "code_output_batches",
            f"{batch_name.split('/')[-1]}.txt",
        )

        # Skip if already downloaded
        if os.path.exists(code_output_path):
            logger.info(f"[{i+1}/{len(code_batch_names)}] Already downloaded: {batch_name}")
        else:
            logger.info(f"[{i+1}/{len(code_batch_names)}] Downloading: {batch_name}")
            try:
                download_batch_outputs(batch_name, code_output_path)
            except Exception as e:
                logger.error(f"Failed to download {batch_name}: {e}")
                continue

        algorithm_result = get_algorithm_result(algorithm_id)
        if algorithm_result is None:
            logger.warning(f"Algorithm not found: {algorithm_id}")
            continue

        # Check if already processed
        if algorithm_result.code_id_list and len(algorithm_result.code_id_list) > 0:
            logger.info(f"Algorithm {algorithm_id[:16]}... already has code, skipping")
            continue

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
        logger.info(f"Processed code for algorithm {algorithm_id[:16]}...")

    logger.info(f"Code collection complete for {len(code_batch_names)} algorithms")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="LLM-SAT data generation pipeline")
    parser.add_argument("--mutants-only", action="store_true",
                        help="Skip leader generation, generate mutants for existing leaders")
    parser.add_argument("--source_tag", type=str, default=None,
                        help="Source generation tag to load existing leaders from (used with --mutants-only)")
    parser.add_argument("--output_tag", type=str, default=None,
                        help="Output generation tag for new mutants (used with --mutants-only)")
    parser.add_argument("--generation_tag", type=str, default="pipeline_test",
                        help="Generation tag (used for full team generation)")
    parser.add_argument("--designer_prompt_path", type=str,
                        default="./data/prompts/leader_prompt_testing.txt",
                        help="Path to designer prompt for leader generation")
    parser.add_argument("--variant_prompt_path", type=str,
                        default="./data/prompts/variant_prompt.txt",
                        help="Path to variant prompt template")
    parser.add_argument("--code_prompt_path", type=str,
                        default="./data/prompts/coder_prompt_testing.txt",
                        help="Path to coder prompt template")
    parser.add_argument("--n_leaders", type=int, default=5,
                        help="Number of leaders to generate (full mode only)")
    parser.add_argument("--m_variants", type=int, default=3,
                        help="Number of mutant variants per leader")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Gemini model to use")
    parser.add_argument("--sync", action="store_true",
                        help="Use synchronous API calls instead of batch")

    args = parser.parse_args()

    if args.mutants_only:
        if not args.source_tag:
            parser.error("--source_tag is required with --mutants-only")
        if not args.output_tag:
            parser.error("--output_tag is required with --mutants-only")

        generate_mutants_for_leaders(
            variant_prompt_path=args.variant_prompt_path,
            code_prompt_template_path=args.code_prompt_path,
            source_generation_tag=args.source_tag,
            output_generation_tag=args.output_tag,
            m_variants_per_leader=args.m_variants,
            model=args.model,
            sync=args.sync,
        )
    else:
        generate_team_data(
            generation_tag=args.generation_tag,
            designer_prompt_path=args.designer_prompt_path,
            variant_prompt_path=args.variant_prompt_path,
            code_prompt_template_path=args.code_prompt_path,
            n_leaders=args.n_leaders,
            m_variants_per_leader=args.m_variants,
            model=args.model,
            sync=args.sync,
        )


if __name__ == "__main__":
    main()
