from __future__ import annotations

import os
import shutil
import json
import subprocess
import re
from dataclasses import dataclass
from typing import Dict, List, Optional
import argparse
from datetime import datetime
from llmsat.llmsat import NOT_INITIALIZED, CHATGPT_DATA_GENERATION_TABLE, ALGORITHM
from llmsat.utils.aws import (
    get_algorithm_result,
    get_algorithm_result_of_status,
    get_ids_from_router_table,
    get_code_result,
    get_code_result_of_status,
    update_code_result,
    update_algorithm_result,
    get_all_algorithm_ids,
    get_all_algorithm_results,
    ToAlgorithmResult,
    ToCodeResult,
)
from llmsat.llmsat import (
    CodeResult,
    CodeStatus,
    AlgorithmResult,
    BASE_SOLVER_PATH,
    SAT2025_BENCHMARK_PATH,
    get_logger,
    setup_logging,
    AlgorithmStatus,
    RECOVERED_ALGORITHM,
)
from llmsat.utils.paths import get_solver_dir, get_solver_solving_times_path, get_algorithm_dir,get_solver_result_dir
from llmsat.utils.utils import wrap_command_to_slurm

logger = get_logger(__name__)


@dataclass
class SolverEvaluationResult:
    solver_path: str
    solver_id: str
    build_success: bool
    par2_score: float
    other_metrics: Dict[str, float]

    def get_reward(self):
        # compute the final scalar / vector reward
        pass

def parse_solving_time(content: str) -> Optional[float]:
    # parse the solving time from the content
    for line in content.split("\n"):
        if "solve time" in line:
            try:
                value = float(line.split(" ")[-1])
                logger.debug(f"Parsed solving time: {value}")
                return value
            except Exception as e:
                logger.debug(f"Failed parsing solving time from line: {line} ({e})")
                return None
    return None

def _compute_average(values: List[float]) -> Optional[float]:
    non_none = [v for v in values if v is not None]
    if not non_none:
        logger.debug("No values to average (all None or empty).")
        return None
    avg = sum(non_none) / len(non_none)
    logger.debug(f"Computed average over {len(non_none)} values: {avg}")
    return avg

def _get_activation_cmd() -> str:
    # Use user-requested env activation instead of conda
    logger.debug("Using activation command: source ../../general/bin/activate")
    return "source ../../general/bin/activate"

@dataclass
class EvaluationPipeline:
    """Unified evaluation entry point for designer and coder models."""
    def __init__(self):
        pass

    # def clean_solving_logs(self, algorithm_id: str, code_id: str) -> None:
    #     # clean the solving logs
    #     solver_dir = get_solver_dir(algorithm_id, code_id)
    #     logger.info(f"Cleaning solving logs in {solver_dir}")
    #     if not os.path.isdir(solver_dir):
    #         logger.debug(f"Solver dir does not exist: {solver_dir}")
    #         return
    #     removed = 0
    #     for file in os.listdir(solver_dir):
    #         if file.endswith(".solving.log") or file.endswith(".slurm.log"):
    #             try:
    #                 os.remove(f"{solver_dir}/{file}")
    #                 removed += 1
    #                 logger.debug(f"Removed log file: {file}")
    #             except FileNotFoundError:
    #                 pass
    #     logger.info(f"Removed {removed} log files from {solver_dir}")

    def test(self) -> None:
        # test the evaluation pipeline
        slurm_ids = self.slurm_run_evaluate("solvers/algorithm_1/code_1", SAT2025_BENCHMARK_PATH)
        # self.slurm_collect_result([slurm_ids], "1")
        # time = self.parse_solving_time("test/test.log")
        # print(time)
        # print(slurm_ids)
        pass

    def clean_solver(self, algorithm_id: str, code_id: str) -> None:
        # clean the solver
        solver_dir = get_solver_dir(algorithm_id, code_id)
        logger.info(f"Removing solver directory {solver_dir}")
        shutil.rmtree(solver_dir, ignore_errors=True)
        logger.debug(f"Removed solver directory {solver_dir} (ignore_errors=True)")

    def parse_solving_time(self, file_path: str) -> Optional[float]:
        lines = open(file_path, "r").readlines()
        for line in reversed(lines): 
            if "process-time" in line:
                match = re.search(r'(\d+\.?\d*)\s+seconds', line)
                if match:
                    time = float(match.group(1))
                    return time
            if "error" in line or "=== make stderr ===" in line:
                logger.warning(f"Error in solving time file {file_path}")
                print(file_path)
        
        # logger.warning(f"Failed to parse solving time from")
        # print(file_path)
        return 5000

    def collect_results(self, algorithm_id: str, code_id: str, force_recollect: bool = False) -> None:
        # collect the results from the solver
        solver_dir = get_solver_result_dir(algorithm_id, code_id)
        if not os.path.isdir(solver_dir):
            logger.warning(f"Solver directory missing: {solver_dir}")
            return
        result_path = get_solver_solving_times_path(algorithm_id, code_id)
        if os.path.exists(result_path) and not force_recollect:
            logger.warning(f"Results already collected for algorithm {algorithm_id}, code {code_id}")
            return
        print(f"Collecting results from {solver_dir}")
        solving_times: Dict[str, float] = {}
        removed_files = []
        if os.path.isdir(solver_dir):
            files = os.listdir(solver_dir)
            if len(files) == 0:
                logger.warning(f"No files in solver directory: {solver_dir}")
                return
            for file in files:
                if file.endswith(".solving.log"):
                    instance_name = file.split(".")[0]
                    instance_time = self.parse_solving_time(f"{solver_dir}/{file}") 
                    if instance_time is not None:
                        solving_times[instance_name] = instance_time
                        logger.debug(f"Parsed {file} -> {instance_time}") 
                        removed_files.append(file)
        for file in removed_files:
            os.remove(f"{solver_dir}/{file}")
        par2 = _compute_average(list(solving_times.values()))
        logger.info(f"Computed PAR2 for algorithm {algorithm_id}, code {code_id}: {par2}")

        # update the code result and algorithm result
        code_result = get_code_result(code_id)
        if code_result is not None:
            code_result.par2 = str(par2)
            code_result.build_success = True
            code_result.status = CodeStatus.Evaluated
            update_code_result(code_result)
            logger.debug(f"Updated code result par2={code_result.par2} for code_id={code_id}")
        with open(result_path, "w") as f:
            json.dump(solving_times, f)
        print(f"Wrote solving times to {result_path}")
        if len(solving_times) != 400:
            logger.warning(f"Expected 400 instances, but got {len(solving_times)}")
        self.clean_solver(algorithm_id, code_id)

        # remove all the solvers
        return par2

    def slurm_collect_result(self, slurm_ids: List[int], code_id: str) -> None:
        activate_python_path = _get_activation_cmd()
        logger.info(f"Collecting SLURM results for code_id={code_id}, {len(slurm_ids)} jobs")
        logger.info(f"Submitting job to collect results for code_id={code_id}")
        code_result = get_code_result(code_id)
        algorithm_id = code_result.algorithm_id
        logger.info(f"Algorithm ID: {algorithm_id}, Code ID: {code_id}")
        result_dir = get_solver_result_dir(algorithm_id, code_id)
        cmd = f"{activate_python_path} && python src/llmsat/pipelines/evaluation.py --algorithm_id {algorithm_id} --code_id {code_id} --collect_result"
        output_file = f"./logs/evaluation/collect_result_{code_id}.log"
        slurm_cmd = wrap_command_to_slurm(cmd, output_file=output_file, job_name=f"collect_result_{code_id}", dependencies=[str(slurm_id) for slurm_id in slurm_ids])
        slurm_id = os.popen(slurm_cmd).read()
        slurm_id = int(slurm_id.split()[-1])
        logger.info(f"Submitted collect result job {slurm_id}, dependencies: {','.join([str(slurm_id) for slurm_id in slurm_ids])}")
        logger.info(f"CMD: {slurm_cmd}")


    def filter_code(self, code: str) -> str:
        def normalize_escaped_whitespace(text: str) -> str:
            # Heuristic: if we see many literal '\n' and very few real newlines, unescape once
            literal_newlines = text.count("\\n")
            real_newlines = text.count("\n")
            if literal_newlines >= 3 and real_newlines < 3:
                try:
                    # Attempt a single unicode escape decode
                    decoded = bytes(text, "utf-8").decode("unicode_escape")
                    return decoded
                except Exception:
                    # Fallback to simple replacements
                    text = text.replace("\\r\\n", "\n")
                    text = text.replace("\\n", "\n")
                    text = text.replace("\\t", "\t")
                    text = text.replace('\\"', '"')
                    return text
            # Also normalize Windows line endings if present
            return text.replace("\r\n", "\n")

        def extract_function(text: str, func_name: str) -> Optional[str]:
            # Try to find a reasonable C function header for 'bool kissat_restarting'
            # Allow optional qualifiers like 'static' or 'inline'
            header_regex = rf"(?:static\s+)?(?:inline\s+)?bool\s+{func_name}\b[^\{{]*\{{"
            m = re.search(header_regex, text, flags=re.DOTALL)
            if m:
                start = m.start()
                open_brace = m.end() - 1  # points at '{'
                brace = 0
                i = open_brace
                # Count braces to find the matching closing brace of the function
                while i < len(text):
                    c = text[i]
                    if c == "{":
                        brace += 1
                    elif c == "}":
                        brace -= 1
                        if brace == 0:
                            # include the closing brace
                            end = i + 1
                            return text[start:end]
                    i += 1
                return None
            # Fallback: find function name and reconstruct a 'bool' header
            name_idx = text.find(func_name)
            if name_idx == -1:
                return None
            after_name = text[name_idx + len(func_name):]
            brace_idx = after_name.find("{")
            if brace_idx == -1:
                return None
            # Compose a normalized header starting at 'bool kissat_restarting'
            header_prefix = f"bool {func_name}"
            header_suffix = after_name[: brace_idx + 1]
            header = f"{header_prefix}{header_suffix}"
            # Now find the matching closing brace from this opening
            rest = after_name[brace_idx + 1 :]
            brace = 1
            i = 0
            while i < len(rest):
                c = rest[i]
                if c == "{":
                    brace += 1
                elif c == "}":
                    brace -= 1
                    if brace == 0:
                        body_end = i + 1
                        return header + rest[:body_end]
                i += 1
            return None

        # Extract content within <code>...</code> if present
        if "<code>" in code:
            code = code.split("<code>")[1].split("</code>")[0]
        else:
            logger.warning("No <code> tag found in code")

        # Normalize escaped whitespace (e.g., literal '\n') before parsing/writing
        code = normalize_escaped_whitespace(code)

        if "kissat_restarting" not in code:
            logger.error("No kissat_restarting function found in code")
            return None

        extracted = extract_function(code, "kissat_restarting")
        if not extracted:
            logger.error("Failed to parse kissat_restarting function body")
            return None
        return extracted

    def build_solver(self, code_result: CodeResult) -> str: # if success, return the solver path, otherwise return None
        logger.info(f"Building solver for code_result={code_result}")
        code = code_result.code
        code = self.filter_code(code) # 
        if code is None:
            logger.error("Failed to find kissat_restarting function in code")
            return None
        
        # copy original solver to a new folder
        new_solver_path = get_solver_dir(code_result.algorithm_id, code_result.id)
        logger.info(f"Building solver at {new_solver_path} for algorithm={code_result.algorithm_id}, code={code_result.id}")
        if os.path.exists(new_solver_path):
            shutil.rmtree(new_solver_path)
        shutil.copytree(BASE_SOLVER_PATH, new_solver_path)
        # replace the code in the new solver
        restart_file = f"{new_solver_path}/src/restart.c"
        # First read the file to find where to insert the code
        with open(restart_file, "r") as f:
            lines = f.readlines()
        
        # Find LLMSAT start/end markers and replace the block between them
        start_idx = None
        end_idx = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if start_idx is None and (line.startswith("//LLMSAT start") or stripped.startswith("// LLMSAT: start")):
                start_idx = i
                continue
            if start_idx is not None and (line.startswith("//LLMSAT end") or stripped.startswith("// LLMSAT: end")):
                end_idx = i
                break
        
        if start_idx is None:
            raise ValueError("Could not find '//LLMSAT start' marker in restart.c")
        if end_idx is None:
            raise ValueError("Could not find '//LLMSAT end' marker in restart.c")
        logger.debug(f"Replacing lines ({start_idx+1}, {end_idx}) in restart.c between LLMSAT markers")

        # Write the modified content: keep start marker line, replace inner block, keep end marker and rest
        with open(restart_file, "w") as f:
            # Up to and including start marker
            f.writelines(lines[: start_idx + 1])
            # New code block (ensure trailing newline)
            f.write(code.rstrip() + "\n")
            # From end marker to end (preserve end marker)
            f.writelines(lines[end_idx:])
        logger.debug(f"Injected code into {restart_file}")

        # try compile the solver
        try:
            logger.info(f"Compiling solver at {new_solver_path}")
            configure_proc = subprocess.run(
                ["./configure"],
                cwd=new_solver_path,
                capture_output=True,
                text=True,
            )
            make_proc = None
            if configure_proc.returncode == 0:
                make_proc = subprocess.run(
                    ["make", "-j1"],
                    cwd=new_solver_path,
                    capture_output=True,
                    text=True,
                )
                build_success = make_proc.returncode == 0
            else:
                build_success = False
            # print(build_success)
            # exit()
            # Aggregate logs from both phases (always record)
            logs = []
            logs.append("=== ./configure stdout ===\n" + (configure_proc.stdout or ""))
            logs.append("=== ./configure stderr ===\n" + (configure_proc.stderr or ""))
            if make_proc is not None:
                logs.append("=== make stdout ===\n" + (make_proc.stdout or ""))
                logs.append("=== make stderr ===\n" + (make_proc.stderr or ""))
            output = "\n".join(logs)

            # Always write a full build log
            algorithm_dir = get_algorithm_dir(code_result.algorithm_id)
            build_log_path = f"{algorithm_dir}/code_{code_result.id}.build.log"
            with open(build_log_path, "w") as f:
                f.write(output)
            logger.info(f"Wrote build log to {build_log_path}")
            # also copy the restart.c to the algorithm directory
            shutil.copy2(restart_file, f"{algorithm_dir}/code_{code_result.id}.restart.c")
            logger.info(f"Copied restart.c to {algorithm_dir}/code_{code_result.id}.restart.c")

            if not build_success:
                algorithm_dir = get_algorithm_dir(code_result.algorithm_id)
                failed_log_path = f"{algorithm_dir}/code_{code_result.id}.build_failed.log"
                with open(failed_log_path, "w") as f:
                    f.write(output)
                logger.warning(f"Build failed for solver at {new_solver_path}, output saved to {failed_log_path}")
            # if build_success remains True, proceed below
        except Exception as e:
            build_success = False
        if build_success:
            new_solver_bin_path = f"{new_solver_path}/build/kissat"
            os.makedirs(new_solver_path, exist_ok=True)
            try:
                shutil.copy2(new_solver_bin_path, f"{new_solver_path}/kissat")
                logger.info(f"Build succeeded, binary copied to {new_solver_path}/kissat")
            except Exception:
                return new_solver_path
            return new_solver_path
        else:
            logger.warning(f"Build failed for solver at {new_solver_path}")
            return None
        return 

    def slurm_run_evaluate(self, solver_path: str, benchmark_path: str, result_dir: str) -> List[int]:
        # run the solver on the benchmark
        # activate_python_path = _get_activation_cmd()
        logger.info(f"Submitting SLURM jobs for solver {solver_path} on benchmarks {benchmark_path}")
        slurm_ids = []
        for benchmark_file in os.listdir(benchmark_path):
            if benchmark_file.endswith(".cnf"):
                command = f"{solver_path}/build/kissat {benchmark_path}/{benchmark_file} > {result_dir}/{benchmark_file}.solving.log"
                slurm_log = f"{result_dir}/{benchmark_file}.slurm.log"
                if os.path.exists(f"{result_dir}/{benchmark_file}.solving.log"):
                    continue
                slurm_cmd = wrap_command_to_slurm(command, output_file=slurm_log, job_name=f"solve_{benchmark_file}", time="00:00:5000")
                logger.info(f"Submitting job with command: {slurm_cmd}")
                slurm_id = os.popen(slurm_cmd).read()
                slurm_id = int(slurm_id.split()[-1])
                logger.info(f"Submitted job {slurm_id} for {benchmark_file}")
                slurm_ids.append(slurm_id)
        return slurm_ids

    def run_single_solver(self, code_id: str) -> None:  # process single code
        """Run evaluation for configured components."""
        code_result = get_code_result(code_id)
        if code_result is None:
            logger.error(f"Code result not found for code_id={code_id}")
            return
        if code_result.status == CodeStatus.BuildFailed:
            logger.warning(f"Code result {code_id} is already build failed, skip?")
            return
        if code_result.status == CodeStatus.Evaluating:
            logger.warning(f"Code result {code_id} is already evaluating, skip")
            return
        assert code_result is not None, "Code result not found"
        logger.info(f"Running single solver for code_id={code_id}, algorithm_id={code_result.algorithm_id}")
        solver_path = self.build_solver(code_result) # build in evaluation
        # solver_path = f"solvers/algorithm_{code_result.algorithm_id}/code_{code_result.id}"
        if solver_path is not None: # build successful
            logger.info(f"Solver built successfully: {solver_path}")
            result_dir = get_solver_result_dir(code_result.algorithm_id, code_result.id)
            slurm_ids = self.slurm_run_evaluate(solver_path, SAT2025_BENCHMARK_PATH, result_dir)
            # slurm_ids = []
            code_result.status = CodeStatus.Evaluating
            update_code_result(code_result) 
            self.slurm_collect_result(slurm_ids, code_id)
        else: # build failed
            code_result.status = CodeStatus.BuildFailed
            update_code_result(code_result) 
            logger.error("Solver build failed")
    
    def read_current_progress(self) -> None:
        # read the current progress from the progress file
        pass

    def read_algorithm(self, algorithm_id: str) -> AlgorithmResult:
        logger.debug(f"Reading algorithm result for id={algorithm_id}")
        result = get_algorithm_result(algorithm_id)
        logger.info(f"Read algorithm {result.id} with {len(result.code_id_list or [])} code ids")
        return result

    def run_all_solvers(self, algorithm_id: str) -> None:  # pragma: no cover - declaration only
        """Run evaluation for all configured components."""
        logger.info(f"Running evaluation for algorithm {algorithm_id}")
        algorithm = self.read_algorithm(algorithm_id)
        os.makedirs(f"solvers/algorithm_{algorithm_id}", exist_ok=True)
        code_id_list = self.generate_or_read_code(algorithm) # actually should only read here, generation should be done in a separate process
        logger.info(f"Found {len(code_id_list)} code ids to evaluate for algorithm {algorithm_id}")
        for code_id in code_id_list:
            logger.info(f"Starting evaluation for code_id={code_id}")
            self.run_single_solver(code_id)
        algorithm.status = AlgorithmStatus.Evaluating
        update_algorithm_result(algorithm)

        # for code_id in code_id_list:
        #     logger.debug(f"Starting evaluation for code_id={code_id}")
        #     self.run_single_solver(code_id)
    def fix_algorithm_code_mapping(self, algorithm: AlgorithmResult, all_code_id_list: List[str]) -> None:
        # fix the algorithm code mapping
        if algorithm.code_id_list[0].strip('"') == "NOT_INITIALIZED":
            return
        logger.info(f"Fixing algorithm {algorithm.id} code mapping to {algorithm.code_id_list} code ids")
        code_registered = algorithm.code_id_list 
        for code_id in code_registered:
            if code_id in all_code_id_list:
                print(code_id)
                exit()

    def generate_or_read_code(self, algorithm: AlgorithmResult) -> List[str]:
        # Return list of code ids to evaluate
        ids = algorithm.code_id_list or []
        logger.debug(f"generate_or_read_code returning {len(ids)} code ids")
        return ids
        
def find_codes(code_ids: List[str]) -> Dict[str, str]:
    """
    Locally find where the code directories are when algorithm IDs are missing.
    Searches through all algorithm directories to find matching code directories.
    
    Args:
        code_ids: List of code IDs to search for
    
    Returns:
        Dictionary mapping code_id -> directory path for found code directories
    """
    directories = {}
    solvers_base = "solvers"
    
    if not os.path.exists(solvers_base):
        logger.warning(f"Solvers base directory does not exist: {solvers_base}")
        return directories
    
    # Iterate through all algorithm directories
    for algo_dir in os.listdir(solvers_base):
        algo_path = os.path.join(solvers_base, algo_dir)
        
        # Skip if not a directory or doesn't match algorithm directory pattern
        if not os.path.isdir(algo_path) or not algo_dir.startswith("algorithm_"):
            continue
        
        # Check each code_id in this algorithm directory
        for code_id in code_ids:
            code_dir_name = f"code_{code_id}"
            code_path = os.path.join(algo_path, code_dir_name)
            
            # If the code directory exists, add it to results
            if os.path.isdir(code_path):
                directories[code_id] = code_path
                logger.debug(f"Found code directory: {code_path} for code_id: {code_id}")
    
    logger.info(f"Found {len(directories)} code directories for {len(code_ids)} code IDs")
    return directories
def restore_codes(algorithm_ids: List[str]) -> None:
    # restore the codes
    for algorithm_id in algorithm_ids:
        algorithm = get_algorithm_result(algorithm_id)
        code_ids = algorithm.code_id_list
        for code_id in code_ids:
            code = get_code_result(code_id)
            if code is None:
                logger.error(f"Code result not found for code_id={code_id}")
                continue
            code.algorithm_id = algorithm_id
            update_code_result(code)
            logger.info(f"Restored code {code_id} for algorithm {algorithm_id}")

def restore_algorithms(algorithm_ids: List[str]) -> List[AlgorithmResult]:
    # restore the algorithms
    for algorithm_id in algorithm_ids:
        # algorithm = get_algorithm_result(algorithm_id)
        algorithm = AlgorithmResult(
            id=algorithm_id,
            code_id_list=NOT_INITIALIZED,
            status=AlgorithmStatus.Evaluating,
            last_updated=datetime.now(),
            prompt="",
            par2=NOT_INITIALIZED,
            error_rate=NOT_INITIALIZED,
            algorithm=RECOVERED_ALGORITHM,
            other_metrics=NOT_INITIALIZED,
        )
        # find code ids in the algorithm directory
        algorithm_dir = get_algorithm_dir(algorithm_id)
        code_ids = []
        # find directories in the algorithm directory to restore the code ids
        for code_dir in os.listdir(algorithm_dir):
            if code_dir.startswith("code_") and os.path.isdir(os.path.join(algorithm_dir, code_dir)):
                code_ids.append(code_dir.split("code_")[1])
        algorithm.code_id_list = code_ids
        update_algorithm_result(algorithm)
        for code_id in code_ids:
            code = get_code_result(code_id)
            if code is None:
                logger.error(f"Code result not found for code_id={code_id}")
                continue
            code.algorithm_id = algorithm_id
            update_code_result(code)
            logger.info(f"Restored code {code_id} for algorithm {algorithm_id}")
        logger.info(f"Restored algorithm {algorithm_id} code mapping to {len(code_ids)} code ids")

def rescue_data():
    code_results = get_code_result_of_status(CodeStatus.Evaluated)
    code_ids = []
    for code_result in code_results:
        if code_result.algorithm_id is None:
            code_ids.append(code_result.id)
    code_ids = [code_result.id for code_result in code_results]
    code_dirs = find_codes(code_ids)
    algorithm_ids = set()
    for code_id, code_dir in code_dirs.items():
        algorithm_id = code_dir.split("/")[-2].split("algorithm_")[1]
        algorithm_ids.add(algorithm_id)
        # algorithm = get_algorithm_result(algorithm_id)
        # algorithm.code_id_list = NOT_INITIALIZED
        # update_algorithm_result(algorithm)
    logger.info(f"algorithm ids: {algorithm_ids}")
    restore_algorithms(list(algorithm_ids))

def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm_id", type=str, default=None)
    parser.add_argument("--code_id", type=str, default=None)
    parser.add_argument("--first_n", type=int, default=None)
    parser.add_argument("--run_all", action="store_true", default=False)
    parser.add_argument("--collect_result", action="store_true", default=False)
    parser.add_argument("--collect_all_results", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--generation_tag", type=str, default="chatgpt_data_generation_gpt4o_2")
    args = parser.parse_args()
    evaluation_pipeline = EvaluationPipeline()
    # evaluation_pipeline.run_all_solvers("1")

    if args.run_all:
        assert args.algorithm_id is None, "Cannot specify both --algorithm_id and --run_all"
        # algorithms = get_algorithm_result_of_status(AlgorithmStatus.Generated)
        algorithm_ids = get_ids_from_router_table(CHATGPT_DATA_GENERATION_TABLE, args.generation_tag)
        algorithms = [get_algorithm_result(algorithm_id) for algorithm_id in algorithm_ids]
        # algorithms = get_algorithm_result_of_status(AlgorithmStatus.CodeGenerated)
        logger.info(f"Found {len(algorithms)} algorithms to evaluate")
        if args.first_n is not None:
            algorithms = algorithms[:args.first_n]
    elif args.algorithm_id is not None:
        assert args.first_n is None, "Cannot specify both --algorithm_id and --first_n"
        algorithms = [get_algorithm_result(args.algorithm_id)]
        if args.collect_result:
            evaluation_pipeline.collect_results(args.algorithm_id, args.code_id, force_recollect=True)
            return
    elif args.collect_all_results:
        assert args.algorithm_id is None, "Cannot specify both --algorithm_id and --collect_all_results"
        # code_results = get_code_result_of_status(CodeStatus.Evaluated)
        # code_ids = [code_result.id for code_result in code_results]
        # code_dirs = find_codes(code_ids)
        # algorithm_id_of_code_id = {}
        # for code_id, code_dir in code_dirs.items():
        #     algorithm_id = code_dir.split("/")[-2].split("algorithm_")[1]
        #     algorithm_id_of_code_id[code_id] = algorithm_id
        # for code_id in code_ids:
        #     algorithm_id = algorithm_id_of_code_id[code_id]
        #     evaluation_pipeline.collect_results(algorithm_id, code_id, force_recollect=True)
        #     # return
        algorithm_ids = get_ids_from_router_table(CHATGPT_DATA_GENERATION_TABLE, "chatgpt_data_generation_gpt4o_algsingleshot_example")
        for algorithm_id in algorithm_ids:
            algorithm_result = get_algorithm_result(algorithm_id)
            code_ids = algorithm_result.code_id_list
            for code_id in code_ids:
                evaluation_pipeline.collect_results(algorithm_id, code_id, force_recollect=True)
                # return
        # restore_codes(list(algorithm_ids))
        # exit()
        # logger.info(f"code dirs: {code_dirs}")
        # logger.info(f"all code ids: {code_ids}")
        # algorithms = get_all_algorithm_results()
        # logger.info(f"all algorithms: {len(algorithms)}")
        # for algorithm in algorithms:
        #     evaluation_pipeline.fix_algorithm_code_mapping(algorithm, code_ids)
        return
    elif args.test:
        evaluation_pipeline.test()
        return
    else:
        assert False, "Must specify either --run_all or --algorithm_id"
    logger.info(f"Running evaluation for {len(algorithms)} algorithms")
    for algorithm in algorithms:
        # print(algorithm.algorithm)
        logger.info(algorithm.id)
        evaluation_pipeline.run_all_solvers(algorithm.id)

def test():
    setup_logging()

    algorithms = get_algorithm_result_of_status(AlgorithmStatus.CodeGenerated)
    # algorithms = get_all_algorithm_results()
    logger.info(f"Found {len(algorithms)} algorithms to evaluate")
    for algorithm in algorithms:
        logger.info(f"{algorithm.id}, {algorithm.status}")

def should_stat(generation_tag: str) -> bool:
    setup_logging()
    algorithm_ids = get_ids_from_router_table(CHATGPT_DATA_GENERATION_TABLE, generation_tag)
    print(f"Number of algorithms: {len(algorithm_ids)} for generation tag: {generation_tag}")
    for algorithm_id in algorithm_ids:
        algorithm = get_algorithm_result(algorithm_id)
        logger.info(f"{algorithm.id}, {algorithm.status}")
        code_ids = algorithm.code_id_list
        for code_id in code_ids:
            code = get_code_result(code_id)
            logger.info(f"{code.id}, {code.status}, {code.par2}")

def clean_up_evaluation_results(generation_tag: str):
    algorithm_ids = get_ids_from_router_table(CHATGPT_DATA_GENERATION_TABLE, generation_tag)
    for algorithm_id in algorithm_ids:
        logger.info(f"Cleaning up evaluation results for algorithm {algorithm_id}")
        solver_dir = get_algorithm_dir(algorithm_id)
        # remove any file or directory starting with "code_"
        for file in os.listdir(solver_dir):
            if file.startswith("code_"):
                os.remove(f"{solver_dir}/{file}")
        logger.info(f"Removed all files starting with 'code_' in {solver_dir}")

if __name__ == "__main__":
    # test()
    # clean_up_evaluation_results("chatgpt_data_generation_gpt4o_4")
    main()
    # evaluation_pipeline = EvaluationPipeline()
    # evaluation_pipeline.run_single_solver("f7ca9f12b4c6b5fa877a000117237e551a6e565ba6c377a48b1917524c8943cc")
    # should_stat("chatgpt_data_generation_gpt4o_5")
    # algorithm_ids = get_ids_from_router_table(CHATGPT_DATA_GENERATION_TABLE, "chatgpt_data_generation_gpt4o_algsingleshot_example")
    # print(f"Number of algorithms: {len(algorithm_ids)}")
    # all_solver_dirs = "./solvers"
    # for solver_dir in os.listdir(all_solver_dirs):
    #     if solver_dir.startswith("algorithm_"):
    #         algorithm_id = solver_dir.split("_")[1]
    #         if algorithm_id not in algorithm_ids:
    #             print(f"Removing solver directory {solver_dir}")
    #             shutil.rmtree(f"{all_solver_dirs}/{solver_dir}")