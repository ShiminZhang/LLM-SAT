from __future__ import annotations

import os
import shutil
import json
import subprocess
import re
from dataclasses import dataclass
from typing import Dict, List, Optional
import argparse
from llmsat.utils.aws import (
    get_algorithm_result,
    get_algorithm_result_of_status,
    get_code_result,
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
        return None

    def collect_results(self, algorithm_id: str, code_id: str) -> None:
        # collect the results from the solver
        solver_dir = get_solver_dir(algorithm_id, code_id)
        logger.info(f"Collecting results from {solver_dir}")
        solving_times: Dict[str, float] = {}
        if os.path.isdir(solver_dir):
            for file in os.listdir(solver_dir):
                if file.endswith(".solving.log"):
                    instance_time = self.parse_solving_time(f"{solver_dir}/{file}")
                    if instance_time is not None:
                        solving_times[file] = instance_time
                        logger.debug(f"Parsed {file} -> {instance_time}")
        else:
            logger.warning(f"Solver directory missing: {solver_dir}")
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
        result_dir = get_solver_result_dir(algorithm_id, code_id)
        with open(f"{result_dir}/solving_times.json", "w") as f:
            json.dump(solving_times, f)
        logger.info(f"Wrote solving times to {result_dir}/solving_times.json")

        # remove all the solvers
        return par2

    def slurm_collect_result(self, slurm_ids: List[int], code_id: str) -> None:
        activate_python_path = _get_activation_cmd()
        logger.info(f"Collecting SLURM results for code_id={code_id}, {len(slurm_ids)} jobs")
        logger.info(f"Submitting job to collect results for code_id={code_id}")
        code_result = get_code_result(code_id)
        algorithm_id = code_result.algorithm_id
        result_dir = get_solver_result_dir(algorithm_id, code_id)
        cmd = f"{activate_python_path} && python src/llmsat/pipelines/evaluation.py --algorithm_id {algorithm_id} --code_id {code_id} --collect_result"
        output_file = f"{result_dir}/collect_result.log"
        slurm_cmd = wrap_command_to_slurm(cmd, output_file=output_file, job_name=f"collect_result_{code_id}", dependencies=[str(slurm_id) for slurm_id in slurm_ids])
        slurm_id = os.popen(slurm_cmd).read()
        slurm_id = int(slurm_id.split()[-1])
        logger.info(f"Submitted collect result job {slurm_id}, dependencies: {','.join([str(slurm_id) for slurm_id in slurm_ids])}")
        logger.info(f"CMD: {slurm_cmd}")


    def filter_code(self, code: str) -> str:
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

        # Find the start and end markers to replace content between them
        start_idx = None
        end_idx = None
        for i, line in enumerate(lines):
            # Support both markers: "//LLMSAT start" and "// LLMSAT: start"
            if line.startswith("//LLMSAT start") or line.strip().startswith("// LLMSAT: start"):
                start_idx = i + 1  # Insert after this line
            elif line.startswith("//LLMSAT end") or line.strip().startswith("// LLMSAT: end"):
                end_idx = i  # End before this line
                break

        if start_idx is None:
            raise ValueError("Could not find '//LLMSAT start' marker in restart.c")
        if end_idx is None:
            raise ValueError("Could not find '//LLMSAT end' marker in restart.c")
        logger.debug(f"Found markers: start at line {start_idx}, end at line {end_idx} in restart.c")

        # Write the modified content: before start, new code, from end marker onwards
        with open(restart_file, "w") as f:
            # Write lines before insertion point (including start marker)
            f.writelines(lines[:start_idx])
            # Write the new code
            f.write(code)
            f.write("\n")
            # Write the remaining lines (from end marker onwards)
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
                slurm_cmd = wrap_command_to_slurm(command, output_file=slurm_log, job_name=f"solve_{benchmark_file}")
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
    
    def generate_or_read_code(self, algorithm: AlgorithmResult) -> List[str]:
        # Return list of code ids to evaluate
        ids = algorithm.code_id_list or []
        logger.debug(f"generate_or_read_code returning {len(ids)} code ids")
        return ids
        

def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm_id", type=str, default=None)
    parser.add_argument("--code_id", type=str, default=None)
    parser.add_argument("--first_n", type=int, default=None)
    parser.add_argument("--run_all", action="store_true", default=False)
    parser.add_argument("--collect_result", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    args = parser.parse_args()
    evaluation_pipeline = EvaluationPipeline()
    # evaluation_pipeline.run_all_solvers("1")

    if args.run_all:
        assert args.algorithm_id is None, "Cannot specify both --algorithm_id and --run_all"
        algorithms = get_algorithm_result_of_status(AlgorithmStatus.CodeGenerated)
        logger.info(f"Found {len(algorithms)} algorithms to evaluate")
        if args.first_n is not None:
            algorithms = algorithms[:args.first_n]
    elif args.algorithm_id is not None:
        assert args.first_n is None, "Cannot specify both --algorithm_id and --first_n"
        algorithms = [get_algorithm_result(args.algorithm_id)]
        if args.collect_result:
            evaluation_pipeline.collect_results(args.algorithm_id, args.code_id)
            return
    elif args.test:
        evaluation_pipeline.test()
        return
    else:
        assert False, "Must specify either --run_all or --algorithm_id"
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
if __name__ == "__main__":
    # test()
    main()