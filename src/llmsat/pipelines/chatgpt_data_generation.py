import os
from pathlib import Path
from typing import Any, Dict
from llmsat.data.create_prompt import build_record, write_jsonl, render_custom_id
from llmsat.data.algorithm_parse import parse_kissat_restart_policy_json
from llmsat.utils.chatgpt_helper import (
    submit_batch_input as helper_submit_batch_input,
    block_until_completion as helper_block_until_completion,
    download_batch_outputs as helper_download_batch_outputs,
)
from llmsat.utils.paths import get_batch_output_dir, get_generation_output_dir, get_solver_solving_times_path
from llmsat.utils.aws import get_ids_from_router_table, update_router_table, get_algorithm_result, get_code_result, clear_router_table, get_algorithms_by_prompt, remove_code_result, remove_algorithm_result  
from llmsat.llmsat import CHATGPT_DATA_GENERATION_TABLE, AlgorithmStatus, AlgorithmResult, ALGORITHM, CodeResult, CodeStatus
from datetime import datetime
import json
from llmsat.llmsat import get_id
from llmsat.utils.aws import update_algorithm_result
from llmsat.llmsat import NOT_INITIALIZED
from llmsat.utils.aws import update_code_result, add_par2_to_code_results_table
from llmsat.llmsat import setup_logging, get_logger 
from llmsat.utils.paths import get_algorithm_dir
import logging
from statistics import mean
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

def generate_code_prompt(template: str, algorithm: str) -> str:
    try:
        return template.format(algorithm=algorithm)
    except Exception:
        return f"{template}\n\nAlgorithm:\n{algorithm}"

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

def parse_algorithm_response(response: Dict[str, Any]) -> str:
    # Extract text similarly to parse_code_response
    if not isinstance(response, dict):
        raw_text = str(response)
    else:
        resp_obj = response.get("response") or response
        text = resp_obj.get("output_text") if isinstance(resp_obj, dict) else None
        if text:
            raw_text = text
        else:
            outputs = resp_obj.get("output") or resp_obj.get("outputs") if isinstance(resp_obj, dict) else None
            extracted = None
            if isinstance(outputs, list):
                for item in outputs:
                    content = item.get("content") if isinstance(item, dict) else None
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "output_text":
                                maybe = part.get("text")
                                if maybe:
                                    extracted = maybe
                                    break
                    if extracted:
                        break
            raw_text = extracted if extracted is not None else (str(resp_obj.get("content")) if isinstance(resp_obj, dict) and "content" in resp_obj else json.dumps(response, ensure_ascii=False))
    # Parse into validated algorithm spec dict, then normalize to JSON string
    try:
        spec = parse_kissat_restart_policy_json(raw_text)
        # Drop optional fields that aren't part of the core spec if present
        if isinstance(spec, dict) and "Reason" in spec:
            spec.pop("Reason")
        return json.dumps(spec, ensure_ascii=False)
    except Exception:
        # If parsing fails, return the raw text for debugging/upstream handling
        return raw_text

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
            algorithm_str = parse_algorithm_response(algorithm_response)
            algorithm_id = get_id(algorithm_str)
            update_router_table(CHATGPT_DATA_GENERATION_TABLE, algorithm_id, generation_tag)
            algorithm_result = AlgorithmResult(
                id=algorithm_id,
                algorithm=algorithm_str,
                status=AlgorithmStatus.Generated,
                last_updated=datetime.now(),
                prompt=designer_prompt,
                par2=NOT_INITIALIZED,
                error_rate=NOT_INITIALIZED,
                other_metrics=NOT_INITIALIZED,
                code_id_list=[],
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
    if use_cache and os.path.exists(algorithms_output_path):
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
            algorithm_str = parse_algorithm_response(algorithm_response)
            algorithm_id = get_id(algorithm_str)
            algorithm_ids_local.append(algorithm_id)
            update_router_table(CHATGPT_DATA_GENERATION_TABLE, algorithm_id, generation_tag)
            algorithm_result = AlgorithmResult(
                id=algorithm_id,
                algorithm=algorithm_str,
                status=AlgorithmStatus.Generated,
                last_updated=datetime.now(),
                prompt=designer_prompt,
                par2=NOT_INITIALIZED,
                error_rate=NOT_INITIALIZED,
                other_metrics=NOT_INITIALIZED,
                code_id_list=[],)
            update_algorithm_result(algorithm_result)
    # Use only algorithms from this batch/run instead of all-time router table
    algorithm_ids = algorithm_ids_local
    code_prompt_template = read_code_prompt_template(code_prompt_template_path)
    waiting_batch_ids = []
    batch_id_to_algorithm_id = {}

    for algorithm_id in algorithm_ids:
        algorithm_result = get_algorithm_result(algorithm_id)
        code_prompt = generate_code_prompt(code_prompt_template, algorithm_result.algorithm)
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
            # continue
            # if code_result.build_success:
            #     result_path = get_solver_solving_times_path(algorithm_id, code_id)

            #     with open(result_path, "r") as f:
            #         data = json.load(f) 
            #         par2 = mean(data.values())
            #         print(f"par2: {par2}")
            #         if par2 > 0:
            #             code_result.par2 = par2
            #             update_code_result(code_result)

def main():
    generate_data(
        generation_tag="chatgpt_data_generation_gpt4o_algselfiteration_3",
        designer_prompt_path="./data/prompts/kissat_self_iteration.txt",
        code_prompt_template_path="./data/prompts/kissat_code.txt",
        n_algorithms=40, n_codes=10,
        model="gpt-4o",
    )

def test():
    # print_generation_result("chatgpt_data_generation")
    # print_generation_result("chatgpt_data_generation_gpt4o_algsingleshot_example")

    print_generation_result("chatgpt_data_generation_gpt4o_algselfiteration_2")
    # # clear_router_table(CHATGPT_DATA_GENERATION_TABLE)
    # fake_generate_data("./data/prompts/kissat_self_iteration.txt", "./data/prompts/kissat_code.txt", "chatgpt_data_generation_gpt4o_algsingleshot_example", algorithm_batch_id="batch_692a6496897c8190b4dd2dea58e3c634")
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
    # fake_generate_data("./data/prompts/kissat.txt", "./data/prompts/kissat_code.txt", "chatgpt_data_generation_gpt4o_3", n_algorithms=50, n_codes=10, algorithm_batch_id="batch_6924b9189824819099023e6efabcae98")
    # test()