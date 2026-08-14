import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from llmsat.data.create_prompt import build_record, write_jsonl, render_custom_id
from llmsat.data.algorithm_parse import parse_algorithm_spec_json, parse_kissat_restart_policy_json
from llmsat.utils.chatgpt_helper import (
    submit_batch_input as helper_submit_batch_input,
    block_until_completion as helper_block_until_completion,
    download_batch_outputs as helper_download_batch_outputs,
)
from llmsat.utils.paths import get_batch_output_dir, get_generation_output_dir, get_solver_solving_times_path
from llmsat.utils.aws import get_ids_from_router_table, update_router_table, get_algorithm_result, get_code_result, clear_router_table, get_algorithms_by_prompt, remove_code_result, remove_algorithm_result  
from llmsat.llmsat import CHATGPT_DATA_GENERATION_TABLE, AlgorithmStatus, AlgorithmResult, ALGORITHM, CodeResult, CodeStatus, Role
from datetime import datetime
import json
from llmsat.llmsat import get_id
from llmsat.utils.aws import update_algorithm_result
from llmsat.llmsat import NOT_INITIALIZED
from llmsat.utils.aws import update_code_result, add_par2_to_code_results_table
from llmsat.llmsat import setup_logging, get_logger 
from llmsat.utils.paths import get_algorithm_dir
import logging

setup_logging(level=logging.INFO)
logger = get_logger(__name__)
def read_algorithm_prompt_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()

def create_batch_input_file(prompt: str, output_path: str, n_requests: int = 10, model: str = "gpt-4.1"):
    logger.info(f"Creating batch input file for {n_requests} requests")
    # Reuse JSONL record structure from llmsat.data.create_prompt
    system_message = os.environ.get("LLMSAT_SYSTEM_MESSAGE", "You are an AI researcher specialising in SAT solver heuristics.")
    model = os.environ.get("OPENAI_MODEL", "gpt-4.1")
    try:
        temperature = float(os.environ.get("OPENAI_TEMPERATURE", "0.7"))
    except Exception:
        temperature = 0.7
    method = "POST"
    url = "/v1/responses"

    records = []
    for i in range(1, int(n_requests) + 1):
        custom_id = render_custom_id("req-{index:04d}", i, stem=None)
        records.append(
            build_record(
                system_message=system_message,
                user_prompt=prompt,
                model=model,
                temperature=temperature,
                method=method,
                url=url,
                custom_id=custom_id,
            )
        )
    write_jsonl(records, Path(output_path))


def count_steps(algorithm_text: str) -> int:
    """Count the number of steps in an algorithm based on 'Step N:' markers."""
    matches = re.findall(r'Step\s+\d+:', algorithm_text, re.IGNORECASE)
    return len(matches)


def create_batch_input_file_variant(prompts: List[str], output_path: str, model: str = "gpt-4.1"):
    """Create batch input file with different prompts for each request."""
    logger.info(f"Creating variant batch input file for {len(prompts)} requests")
    system_message = os.environ.get("LLMSAT_SYSTEM_MESSAGE", "You are an AI researcher specialising in SAT solver heuristics.")
    model = os.environ.get("OPENAI_MODEL", model)
    try:
        temperature = float(os.environ.get("OPENAI_TEMPERATURE", "0.7"))
    except Exception:
        temperature = 0.7
    method = "POST"
    url = "/v1/responses"

    records = []
    for i, prompt in enumerate(prompts, start=1):
        custom_id = render_custom_id("req-{index:04d}", i, stem=None)
        records.append(
            build_record(
                system_message=system_message,
                user_prompt=prompt,
                model=model,
                temperature=temperature,
                method=method,
                url=url,
                custom_id=custom_id,
            )
        )
    write_jsonl(records, Path(output_path))


def submit_batch_input(file_path: str, block: bool = False, poll_interval_seconds: int = 60, timeout_seconds: int = 24 * 60 * 60) -> str:
    batch_id = helper_submit_batch_input(file_path, block=block, poll_interval_seconds=poll_interval_seconds, timeout_seconds=timeout_seconds)
    return batch_id

def block_until_completion(batch_id: str, poll_interval_seconds: int = 60, timeout_seconds: int = 24 * 60 * 60) -> str:
    return helper_block_until_completion(batch_id, poll_interval_seconds=poll_interval_seconds, timeout_seconds=timeout_seconds)

def download_batch_outputs(batch_id: str, output_path: str) -> str:
    result_path = helper_download_batch_outputs(batch_id, Path(output_path))
    return str(result_path)

def read_code_prompt_template(path: str) -> str:
    return read_algorithm_prompt_file(path)

def generate_code_prompt(
    template: str,
    algorithm: str,
    parent_a_code: Optional[str] = None,
    parent_b_code: Optional[str] = None,
) -> str:
    """Substitute the algorithm into the coder prompt template.

    Optionally appends parent C code as reference implementations so the LLM
    can use them as structural anchors (valid struct fields, includes, etc.)
    without being forced to copy them.
    """
    prompt = template.replace("ALGORITHM_PLACEHOLDER", algorithm)

    parent_section = ""
    if parent_a_code and parent_b_code:
        parent_section = f"""

---

### Parent Reference Implementations (both compiled successfully)

Use these as structural references for valid struct fields, includes, and
coding conventions.  Do NOT copy them verbatim — design a NEW implementation
that follows the offspring algorithm description above.

#### Parent A:
<code>
{parent_a_code}
</code>

#### Parent B:
<code>
{parent_b_code}
</code>"""
    elif parent_a_code:
        parent_section = f"""

---

### Parent Reference Implementation (compiled successfully)

Use this as a structural reference for valid struct fields, includes, and
coding conventions.  Do NOT copy it verbatim — design a NEW implementation
that follows the offspring algorithm description above.

<code>
{parent_a_code}
</code>"""

    return prompt + parent_section

def parse_code_response(response: Dict[str, Any]) -> str:
    # Try to extract assistant text from OpenAI batch Responses output
    if not isinstance(response, dict):
        full_text = str(response)
        # Attempt to extract <code>...</code>
        start = full_text.find("<code>")
        end = full_text.find("</code>")
        if start != -1 and end != -1 and end > start:
            return full_text[start + len("<code>"):end].strip()
        return full_text
    resp_obj = response.get("response") or response
    text = resp_obj.get("output_text") if isinstance(resp_obj, dict) else None
    if text:
        full_text = text
        start = full_text.find("<code>")
        end = full_text.find("</code>")
        if start != -1 and end != -1 and end > start:
            return full_text[start + len("<code>"):end].strip()
        return full_text
    outputs = resp_obj.get("output") or resp_obj.get("outputs") if isinstance(resp_obj, dict) else None
    if isinstance(outputs, list):
        for item in outputs:
            content = item.get("content") if isinstance(item, dict) else None
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        maybe = part.get("text")
                        if maybe:
                            full_text = maybe
                            start = full_text.find("<code>")
                            end = full_text.find("</code>")
                            if start != -1 and end != -1 and end > start:
                                return full_text[start + len("<code>"):end].strip()
                            return full_text
    # Fallbacks
    if isinstance(resp_obj, dict) and "content" in resp_obj:
        full_text = str(resp_obj["content"])
        start = full_text.find("<code>")
        end = full_text.find("</code>")
        if start != -1 and end != -1 and end > start:
            return full_text[start + len("<code>"):end].strip()
        return full_text
    full_text = json.dumps(response, ensure_ascii=False)
    start = full_text.find("<code>")
    end = full_text.find("</code>")
    if start != -1 and end != -1 and end > start:
        return full_text[start + len("<code>"):end].strip()
    return full_text

def _strip_markdown_code_block(text: str) -> str:
    """Strip markdown code block wrappers (```json...```) from text."""
    import re
    # Match ```json or ``` at start, and ``` at end
    pattern = r'^```(?:json)?\s*\n?(.*?)\n?```\s*$'
    match = re.match(pattern, text.strip(), re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def parse_algorithm_response(response: Dict[str, Any]) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Parse algorithm response from LLM API.

    Returns:
        Tuple of (description, target_function, reason)
        - description: "Name: algorithm text" human-readable description
        - target_function: Target function name (None if not specified or legacy format)
        - reason: LLM's reasoning for why this algorithm is good (None if absent)
    """
    # Extract text from the batch API response structure
    if not isinstance(response, dict):
        raw_text = str(response)
    else:
        resp_obj = response.get("response") or response
        text = None

        # Try direct output_text field first
        if isinstance(resp_obj, dict):
            text = resp_obj.get("output_text")

        if not text and isinstance(resp_obj, dict):
            # Navigate into body.output for batch API responses
            body = resp_obj.get("body", {})
            if isinstance(body, dict):
                outputs = body.get("output") or body.get("outputs")
            else:
                outputs = resp_obj.get("output") or resp_obj.get("outputs")

            if isinstance(outputs, list):
                for item in outputs:
                    content = item.get("content") if isinstance(item, dict) else None
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "output_text":
                                text = part.get("text")
                                if text:
                                    break
                    if text:
                        break

        if text:
            raw_text = text
        elif isinstance(resp_obj, dict) and "content" in resp_obj:
            raw_text = str(resp_obj["content"])
        else:
            raw_text = json.dumps(response, ensure_ascii=False)

    # Strip markdown code block wrapper if present
    raw_text = _strip_markdown_code_block(raw_text)

    # Parse into validated algorithm spec dict, then build description
    try:
        spec, target_function = parse_algorithm_spec_json(raw_text)
        reason = None
        if isinstance(spec, dict):
            reason = spec.pop("Reason", None) or spec.pop("reason", None)
            name = spec.get("name", "")
            algo_text = spec.get("algorithm", "")
            description = f"{name}: {algo_text}" if name else algo_text
        else:
            description = str(spec)
        return description, target_function, reason
    except Exception as e:
        logger.warning(f"Failed to parse algorithm response: {e}")
        return raw_text, None, None

# given algorithm batch id, generate data for the algorithm batch, everything is already generated, but we need to reupdate the algorithm and code results to the database
def fake_generate_data(designer_prompt_path: str, code_prompt_template_path: str, generation_tag: str, n_algorithms: int = 500, n_codes: int = 10, algorithm_batch_id: str = None):
    # Reconstruct DB state from local files under outputs/{generation_tag}/batch_{algorithm_batch_id}/
    if generation_tag is None or algorithm_batch_id is None:
        logger.error("generation_tag and algorithm_batch_id are required to reconstruct data from local files")
        return
    base_dir = os.path.join(get_batch_output_dir(generation_tag, batch_id=algorithm_batch_id))
    algorithms_output_path = os.path.join(base_dir, "algorithms_output.txt")
    if not os.path.exists(algorithms_output_path):
        logger.error(f"algorithms_output.txt not found at {algorithms_output_path}")
        return
    try:
        designer_prompt = read_algorithm_prompt_file(designer_prompt_path) if designer_prompt_path else ""
    except Exception:
        designer_prompt = ""
    # 1) Recreate algorithms
    logger.info(f"Rebuilding algorithms from {algorithms_output_path}")
    with open(algorithms_output_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                algorithm_response = json.loads(line)
            except Exception:
                # Skip malformed lines
                continue
            description, target_function, reason = parse_algorithm_response(algorithm_response)
            algorithm_id = get_id(description)
            update_router_table(CHATGPT_DATA_GENERATION_TABLE, algorithm_id, generation_tag)
            algorithm_result = AlgorithmResult(
                id=algorithm_id,
                function_name=target_function or "kissat_restarting",
                description=description,
                role=Role.LEADER,
                status=AlgorithmStatus.Generated,
                last_updated=datetime.now(),
                code_id_list=[],
                analysis=reason,
                prompt=designer_prompt,
            )
            update_algorithm_result(algorithm_result)
    # 2) Recreate codes by pairing input files with output batch files using mtime
    #    Inputs:  code_batch_input_{algorithm_id}.txt
    #    Outputs: code_output_batch_{batch_id}.txt
    try:
        code_input_files = [fn for fn in os.listdir(base_dir) if fn.startswith("code_batch_input_") and fn.endswith(".txt")]
        code_output_files = [fn for fn in os.listdir(base_dir) if fn.startswith("code_output_batch_") and fn.endswith(".txt")]
    except Exception as e:
        logger.error(f"Failed to list directory {base_dir}: {e}")
        return
    if len(code_input_files) == 0 or len(code_output_files) == 0:
        logger.warning(f"No code input/output files found in {base_dir}")
        return
    # Build sortable (path, mtime) lists
    def _path(name: str) -> str:
        return os.path.join(base_dir, name)
    input_by_time = sorted(
        [(_path(fn), os.path.getmtime(_path(fn))) for fn in code_input_files],
        key=lambda x: x[1],
    )
    output_by_time = sorted(
        [(_path(fn), os.path.getmtime(_path(fn))) for fn in code_output_files],
        key=lambda x: x[1],
    )
    # Pair in chronological order
    num_pairs = min(len(input_by_time), len(output_by_time))
    if len(input_by_time) != len(output_by_time):
        logger.warning(f"Number of code inputs ({len(input_by_time)}) != outputs ({len(output_by_time)}); pairing by earliest {num_pairs}")
    logger.info(f"Rebuilding {num_pairs} algorithm->code batches by mtime pairing")
    for i in range(num_pairs):
        input_path = input_by_time[i][0]
        output_path = output_by_time[i][0]
        # Extract algorithm_id from input filename: code_batch_input_{algorithm_id}.txt
        base_name = os.path.basename(input_path)
        try:
            algorithm_id = base_name.removeprefix("code_batch_input_").removesuffix(".txt")
        except Exception:
            # Fallback if Python <3.9 or unexpected pattern
            if base_name.startswith("code_batch_input_") and base_name.endswith(".txt"):
                algorithm_id = base_name[len("code_batch_input_"):-len(".txt")]
            else:
                logger.warning(f"Unexpected input filename format: {base_name}; skipping pair")
                continue
        logger.info(f"Rebuilding codes for algorithm_id={algorithm_id} from {os.path.basename(output_path)}")
        # Fetch current algorithm result (created above), ensure list is mutable
        algorithm_result = get_algorithm_result(algorithm_id)
        if algorithm_result is None:
            logger.warning(f"Algorithm {algorithm_id} not found in DB; skipping its codes")
            continue
        # Parse all lines in the output file and create code results
        new_code_ids = []
        try:
            with open(output_path, "r") as fo:
                for line in fo:
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
                    new_code_ids.append(code_id)
        except Exception as e:
            logger.warning(f"Failed to process code outputs {output_path}: {e}")
            continue
        # Update algorithm with associated codes
        if algorithm_result.code_id_list is None:
            algorithm_result.code_id_list = []
        # Avoid duplicates
        existing = set(algorithm_result.code_id_list)
        for cid in new_code_ids:
            if cid not in existing:
                algorithm_result.code_id_list.append(cid)
        update_algorithm_result(algorithm_result)

def generate_data(designer_prompt_path: str, code_prompt_template_path: str, generation_tag: str, n_algorithms: int = 500, n_codes: int = 10, use_cache=False, model: str = "gpt-4o"):
    if generation_tag is None:
        logger.error("Generation tag is None")
        return
    designer_prompt = read_algorithm_prompt_file(designer_prompt_path)
    code_prompt = read_algorithm_prompt_file(code_prompt_template_path)
    batch_input_path = os.path.join(get_generation_output_dir(generation_tag), "designer_batch_input.txt")
    batch_id_map = {}
    batch_id_map["algorithm_batch_id"] = None
    algorithms_output_path = None
    if use_cache and algorithms_output_path and os.path.exists(algorithms_output_path):
        pass
    else:
        # algorithm_batch_id = "batch_6921363d7c8481909540f10ab2723deb"
        create_batch_input_file(designer_prompt, batch_input_path, n_requests=n_algorithms, model=model)
        algorithm_batch_id = submit_batch_input(batch_input_path)
        batch_id_map["algorithm_batch_id"] = algorithm_batch_id
        block_until_completion(algorithm_batch_id)
        algorithms_output_path = os.path.join(get_batch_output_dir(generation_tag, batch_id=algorithm_batch_id), f"algorithms_output.txt")
        download_batch_outputs(algorithm_batch_id, algorithms_output_path)
    # Collect only the algorithm ids generated in THIS run
    algorithm_ids_local = []
    with open(algorithms_output_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            algorithm_response = json.loads(line)
            description, target_function, reason = parse_algorithm_response(algorithm_response)
            algorithm_id = get_id(description)
            algorithm_ids_local.append(algorithm_id)
            update_router_table(CHATGPT_DATA_GENERATION_TABLE, algorithm_id, generation_tag)
            algorithm_result = AlgorithmResult(
                id=algorithm_id,
                function_name=target_function or "kissat_restarting",
                description=description,
                role=Role.LEADER,
                status=AlgorithmStatus.Generated,
                last_updated=datetime.now(),
                code_id_list=[],
                analysis=reason,
                prompt=designer_prompt,
            )
            update_algorithm_result(algorithm_result)
    # Use only algorithms from this batch/run instead of all-time router table
    algorithm_ids = algorithm_ids_local
    code_prompt_template = read_code_prompt_template(code_prompt_template_path)
    waiting_batch_ids = []
    batch_id_to_algorithm_id = {}

    for algorithm_id in algorithm_ids:
        algorithm_result = get_algorithm_result(algorithm_id)
        code_prompt = generate_code_prompt(code_prompt_template, algorithm_result.description)
        code_batch_input_path = os.path.join(get_batch_output_dir(generation_tag, batch_id=algorithm_batch_id), f"code_batch_input_{algorithm_id}.txt")
        create_batch_input_file(code_prompt, code_batch_input_path, n_requests=n_codes)
        batch_id = submit_batch_input(code_batch_input_path)
        waiting_batch_ids.append(batch_id)
        batch_id_to_algorithm_id[batch_id] = algorithm_id
    batch_id_map["code_batch_ids"] = waiting_batch_ids
    batch_id_map["code_batch_map"] = batch_id_to_algorithm_id
    json.dump(batch_id_map, open(os.path.join(get_batch_output_dir(generation_tag, batch_id=algorithm_batch_id), f"batch_id_map_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"), "w"))
    while len(waiting_batch_ids) > 0:
        # Iterate without mutating the list during traversal
        next_waiting_batch_ids = []
        for batch_id in list(waiting_batch_ids):
            block_until_completion(batch_id)
            code_output_path = os.path.join(get_batch_output_dir(generation_tag, batch_id=algorithm_batch_id), f"code_output_{batch_id}.txt")
            download_batch_outputs(batch_id, code_output_path)
            # Resolve the correct algorithm for this batch
            mapped_algorithm_id = batch_id_to_algorithm_id.get(batch_id)
            if mapped_algorithm_id is None:
                logger.warning(f"Unknown batch_id {batch_id} -> algorithm mapping not found; skipping")
                continue
            algorithm_result = get_algorithm_result(mapped_algorithm_id)
            if algorithm_result is None:
                logger.warning(f"Algorithm {mapped_algorithm_id} not found when processing batch {batch_id}; skipping")
                continue
            if algorithm_result.code_id_list is None:
                algorithm_result.code_id_list = []
            existing_ids = set(algorithm_result.code_id_list)
            with open(code_output_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    code_response = json.loads(line)
                    code_str = parse_code_response(code_response)
                    code_result = CodeResult(
                        id=get_id(code_str),
                        algorithm_id=mapped_algorithm_id,
                        code=code_str,
                        status=CodeStatus.Generated,
                        par2=None,
                        last_updated=datetime.now(),
                        build_success=NOT_INITIALIZED,
                    )
                    update_code_result(code_result)
                    if code_result.id not in existing_ids:
                        algorithm_result.code_id_list.append(code_result.id)
                        existing_ids.add(code_result.id)
                    update_algorithm_result(algorithm_result)
        # All processed in this pass as we block per batch; nothing remains
        waiting_batch_ids = next_waiting_batch_ids

# def fix_algorithm_code_mapping(algorithm_ids: List[str], algorithm_batch_id: str):
#     for algorithm_id in algorithm_ids:
#         algorithm_result = get_algorithm_result(algorithm_id)
#         code_ids = algorithm_result.code_id_list
#         for code_id in code_ids:
#             code_result = get_code_result(code_id)
#             print(code_result.algorithm_id)

def print_generation_result(generation_tag: str):
    algorithm_ids = get_ids_from_router_table(CHATGPT_DATA_GENERATION_TABLE, generation_tag)
    print(f"Number of algorithms: {len(algorithm_ids)}")
    for algorithm_id in algorithm_ids:
        algorithm_result = get_algorithm_result(algorithm_id)
        # print(algorithm_result)
        code_ids = algorithm_result.code_id_list
        print(f"algorithm_id: {algorithm_id}, Number of codes: {len(code_ids)}")
        for code_id in code_ids:
            code_result = get_code_result(code_id)
            print(f"code_id: {code_id}, algorithm_id: {code_result.algorithm_id}, status: {code_result.status}, build_success: {code_result.build_success}, par2: {code_result.par2}")
            if code_result.build_success:
                result_path = get_solver_solving_times_path(algorithm_id, code_id)
                with open(result_path, "r") as f:
                    data = json.load(f)
                    print(f"data: {data}")

def main():
    generate_team_data(
        generation_tag="chatgpt_1",
        designer_prompt_path="./data/prompts/leader_prompt_testing.txt",
        variant_prompt_path="./data/prompts/variant_prompt.txt",
        code_prompt_template_path="./data/prompts/coder_prompt_testing.txt",
        n_leaders=2,
        m_variants_per_leader=5,
        model="gpt-5.2",
    )

def generate_team_data(
    designer_prompt_path: str,
    variant_prompt_path: str,
    code_prompt_template_path: str,
    generation_tag: str,
    n_leaders: int = 5,
    m_variants_per_leader: int = 3,
    model: str = "gpt-4o",
):
    """
    Generate team-based algorithm data and code:
    1. Generate n_leaders Team Leader strategies (separate API calls)
    2. For each leader, generate m_variants_per_leader Team Member variants
    3. Generate code for all n_leaders + n_leaders*m_variants_per_leader algorithms

    Args:
        designer_prompt_path: Path to the designer prompt for Team Leaders
        variant_prompt_path: Path to variant prompt template (uses {leader_algorithm} placeholder)
        code_prompt_template_path: Path to code prompt template (uses {algorithm} placeholder)
        generation_tag: Tag for this generation run
        n_leaders: Number of Team Leader strategies to generate
        m_variants_per_leader: Number of Team Member variants per leader
        model: OpenAI model to use for both algorithm and code generation
    """
    if generation_tag is None:
        logger.error("Generation tag is None")
        return
    
    designer_prompt = read_algorithm_prompt_file(designer_prompt_path)
    variant_prompt_template = read_algorithm_prompt_file(variant_prompt_path)
    
    # Step 1: Generate Team Leaders (n separate batch requests, each with 1 algorithm)
    logger.info(f"Generating {n_leaders} Team Leaders")
    leader_batch_input_path = os.path.join(get_generation_output_dir(generation_tag), "leader_batch_input.txt")
    create_batch_input_file(designer_prompt, leader_batch_input_path, n_requests=n_leaders, model=model)
    
    leader_batch_id = submit_batch_input(leader_batch_input_path)
    block_until_completion(leader_batch_id)
    
    leaders_output_path = os.path.join(get_batch_output_dir(generation_tag, batch_id=leader_batch_id), "leaders_output.txt")
    download_batch_outputs(leader_batch_id, leaders_output_path)
    
    # Parse and store Team Leaders
    leader_ids = []
    leader_target_functions = {}  # Map leader_id -> target_function for member inheritance
    leader_descriptions = {}  # Map leader_id -> description for member parent_algorithm_description
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
    
    # Step 2: Generate Team Members for each leader with controlled mutation
    logger.info(f"Generating {m_variants_per_leader} Team Members per leader")

    waiting_batch_ids = []
    batch_id_to_leader_id = {}

    for leader_id in leader_ids:
        leader_result = get_algorithm_result(leader_id)
        leader_algorithm = leader_result.description

        # Count steps in leader algorithm
        num_steps = count_steps(leader_algorithm)
        if num_steps == 0:
            raise ValueError(f"Leader {leader_id} has no step markers. Expected 'Step N:' format.")

        # Calculate step assignments for each variant
        # x = num_steps, m = m_variants_per_leader
        # If x < m: variants 1..x target steps 1..x, variants x+1..m target step x
        # If x >= m: variants 1..m target steps (x-m+1)..x (last m steps)
        step_assignments = []
        if num_steps < m_variants_per_leader:
            for i in range(m_variants_per_leader):
                if i < num_steps:
                    step_assignments.append(i + 1)  # 1-indexed
                else:
                    step_assignments.append(num_steps)  # Target last step
        else:
            start_step = num_steps - m_variants_per_leader + 1
            for i in range(m_variants_per_leader):
                step_assignments.append(start_step + i)

        logger.info(f"Leader {leader_id[:8]}... has {num_steps} steps, assigning variants to steps: {step_assignments}")

        # Build variant prompts with step targets
        variant_prompts = []
        for target_step in step_assignments:
            prompt = variant_prompt_template.replace("{leader_algorithm}", leader_algorithm)
            prompt = prompt.replace("{target_step_num}", str(target_step))
            variant_prompts.append(prompt)

        member_batch_input_path = os.path.join(
            get_batch_output_dir(generation_tag, batch_id=leader_batch_id),
            f"member_batch_input_{leader_id}.txt"
        )
        create_batch_input_file_variant(variant_prompts, member_batch_input_path, model=model)

        batch_id = submit_batch_input(member_batch_input_path)
        waiting_batch_ids.append(batch_id)
        batch_id_to_leader_id[batch_id] = leader_id
    
    # Save batch mapping for recovery
    batch_id_map = {
        "leader_batch_id": leader_batch_id,
        "member_batch_ids": waiting_batch_ids,
        "member_batch_map": batch_id_to_leader_id,
    }
    json.dump(batch_id_map, open(
        os.path.join(get_batch_output_dir(generation_tag, batch_id=leader_batch_id), 
                     f"team_batch_id_map_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"), "w"))
    
    # Wait for all member batches and process
    member_ids = []  # Track all member IDs for code generation
    for batch_id in waiting_batch_ids:
        block_until_completion(batch_id)

        leader_id = batch_id_to_leader_id[batch_id]
        member_output_path = os.path.join(
            get_batch_output_dir(generation_tag, batch_id=leader_batch_id),
            f"member_output_{batch_id}.txt"
        )
        download_batch_outputs(batch_id, member_output_path)

        # Get leader's target_function for inheritance
        leader_target_function = leader_target_functions.get(leader_id, "kissat_restarting")

        # Parse and store Team Members
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

                update_router_table(CHATGPT_DATA_GENERATION_TABLE, member_id, generation_tag)
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

    logger.info(f"Team generation complete: {len(leader_ids)} leaders, {len(member_ids)} members")

    # Step 3: Generate code for all algorithms (leaders + members)
    all_algorithm_ids = leader_ids + member_ids
    logger.info(f"Starting code generation for {len(all_algorithm_ids)} algorithms")

    code_prompt_template = read_code_prompt_template(code_prompt_template_path)

    # Submit code generation batches (one per algorithm, one code per algorithm)
    code_batch_ids = []
    code_batch_to_algorithm = {}

    for algorithm_id in all_algorithm_ids:
        algorithm_result = get_algorithm_result(algorithm_id)
        code_prompt = generate_code_prompt(code_prompt_template, algorithm_result.description)

        code_batch_input_path = os.path.join(
            get_batch_output_dir(generation_tag, batch_id=leader_batch_id),
            f"code_batch_input_{algorithm_id}.txt"
        )
        create_batch_input_file(code_prompt, code_batch_input_path, n_requests=1, model=model)

        batch_id = submit_batch_input(code_batch_input_path)
        code_batch_ids.append(batch_id)
        code_batch_to_algorithm[batch_id] = algorithm_id

    # Update batch map with code batch info
    batch_id_map["code_batch_ids"] = code_batch_ids
    batch_id_map["code_batch_map"] = code_batch_to_algorithm
    json.dump(batch_id_map, open(
        os.path.join(get_batch_output_dir(generation_tag, batch_id=leader_batch_id),
                     f"team_batch_id_map_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"), "w"))

    logger.info(f"Submitted {len(code_batch_ids)} code generation batches")

    # Wait for all code batches and store CodeResults
    for batch_id in code_batch_ids:
        block_until_completion(batch_id)

        algorithm_id = code_batch_to_algorithm[batch_id]
        code_output_path = os.path.join(
            get_batch_output_dir(generation_tag, batch_id=leader_batch_id),
            f"code_output_{batch_id}.txt"
        )
        download_batch_outputs(batch_id, code_output_path)

        # Parse and store CodeResult
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

                # Update algorithm with code reference
                if algorithm_result.code_id_list is None:
                    algorithm_result.code_id_list = []
                algorithm_result.code_id_list.append(code_id)

        # Update algorithm status to CodeGenerated
        algorithm_result.status = AlgorithmStatus.CodeGenerated
        update_algorithm_result(algorithm_result)
        logger.info(f"Generated code for algorithm {algorithm_id}")

    logger.info(f"Code generation complete for {len(all_algorithm_ids)} algorithms")

def test():
    # print_generation_result("chatgpt_data_generation_gpt4o_2")
    # print_generation_result("chatgpt_data_generation_gpt5_2")
    print_generation_result(ALGORITHM)
    # # clear_router_table(CHATGPT_DATA_GENERATION_TABLE)
    # fake_generate_data("./data/prompts/kissat.txt", "./data/prompts/kissat_code.txt", "chatgpt_data_generation", algorithm_batch_id="batch_6921363d7c8481909540f10ab2723deb")
    # print(get_ids_from_router_table(CHATGPT_DATA_GENERATION_TABLE, None))
    # # algorithm_prompt = read_algorithm_prompt_file("./data/prompts/kissat.txt")
    # # algorithms = get_algorithms_by_prompt(algorithm_prompt)
    # # algorithm_ids = [algorithm.id for algorithm in algorithms]
    # algorithm_ids = get_ids_from_router_table(CHATGPT_DATA_GENERATION_TABLE, ALGORITHM)
    # print(f"Number of algorithms: {len(algorithm_ids)}")
    # for algorithm_id in algorithm_ids:
    #     algorithm_result = get_algorithm_result(algorithm_id)
    #     print(algorithm_result)
    #     code_ids = algorithm_result.code_id_list
    #     print(f"Number of codes: {len(code_ids)}")
    #     for code_id in code_ids:
    #         code_result = get_code_result(code_id)
    #         print(code_result.algorithm_id)
    #         remove_code_result(code_id)
    #     remove_algorithm_result(algorithm_id)
        # for code_id in code_ids:
        #     code_result = get_code_result(code_id)
            # print(code_result)
    # for algorithm in algorithms:
    #     print(algorithm.id)
    #     print(algorithm.algorithm)
    #     print(algorithm.prompt)
    #     print(algorithm.par2)
    #     print(algorithm.error_rate)
    #     print(algorithm.other_metrics)
    #     print(algorithm.code_id_list)
    # print(algorithms)
    # generate_data(
    #     generation_tag="chatgpt_data_generation_gpt4o_2",
    #     designer_prompt_path="./data/prompts/kissat.txt",
    #     code_prompt_template_path="./data/prompts/kissat_code.txt",
    #     n_algorithms=5, n_codes=10,
    #     model="gpt-4o",
    #     )
    # generate_data(
    #     generation_tag="chatgpt_data_generation_gpt5_2",
    #     designer_prompt_path="./data/prompts/kissat.txt",
    #     code_prompt_template_path="./data/prompts/kissat_code.txt",
    #     n_algorithms=5, n_codes=10,
    #     model="gpt-5",
    #     )
    

if __name__ == "__main__":
    main()
