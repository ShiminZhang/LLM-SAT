from __future__ import annotations

import os
import shutil
import json
import subprocess
import re
from dataclasses import dataclass
from typing import Dict, List, Optional
import argparse

from llmsat.llmsat import (
    CHATGPT_DATA_GENERATION_TABLE,
    CodeResult,
    CodeStatus,
    AlgorithmStatus,
    BASE_SOLVER_PATH,
    SAT2025_BENCHMARK_PATH,
    get_logger,
    setup_logging,
)
from llmsat.utils.aws import (
    get_algorithm_result,
    get_ids_from_router_table,
    get_code_result,
    update_code_result,
    update_algorithm_result,
)
from llmsat.utils.paths import (
    get_solver_dir,
    get_solver_solving_times_path,
    get_algorithm_dir,
    get_solver_result_dir,
)
from llmsat.utils.utils import wrap_command_to_slurm, wrap_command_to_slurm_array
from llmsat.code_injection import FunctionRegistry, FunctionInjector
from llmsat.debugging import CompilerDebugger

logger = get_logger(__name__)

DEFAULT_REGISTRY_PATH = "solvers/base/function_registry.yaml"


def _compute_average(values: List[float]) -> Optional[float]:
    """Compute average of non-None values."""
    non_none = [v for v in values if v is not None]
    if not non_none:
        return None
    return sum(non_none) / len(non_none)


def _get_activation_cmd() -> str:
    """Return shell command to activate Python environment."""
    return "source ~/general/bin/activate"


@dataclass
class EvaluationPipeline:
    """Evaluation pipeline for building and evaluating SAT solver variants."""

    def __init__(self, registry_path: str = DEFAULT_REGISTRY_PATH):
        self.registry = FunctionRegistry(registry_path)
        self.injector = FunctionInjector(self.registry, BASE_SOLVER_PATH)
        logger.info(f"Initialized FunctionRegistry with {len(self.registry)} functions: {self.registry.list_functions()}")

    def parse_solving_time(self, file_path: str) -> Optional[float]:
        """Parse solving time from a log file. Returns 10000 on timeout/error."""
        try:
            lines = open(file_path, "r").readlines()
        except Exception as e:
            logger.warning(f"Failed to read log file {file_path}: {e}")
            return 10000

        if not lines:
            logger.warning(f"Empty log file (likely timeout or crash): {file_path}")
            return 10000

        for line in reversed(lines):
            if "process-time" in line:
                match = re.search(r'(\d+\.?\d*)\s+seconds', line)
                if match:
                    return float(match.group(1))
            if "error" in line.lower():
                logger.warning(f"Error found in solving log: {file_path}")
                return 10000
            if "CANCELLED" in line or "TIMEOUT" in line or "TIME LIMIT" in line:
                logger.warning(f"SLURM timeout/cancellation detected: {file_path}")
                return 10000

        logger.warning(f"No process-time found in log (incomplete run): {file_path}")
        return 10000

    def collect_results(self, algorithm_id: str, code_id: str, force_recollect: bool = False) -> Optional[float]:
        """Collect evaluation results from solver logs and compute PAR2 score."""
        solver_dir = get_solver_result_dir(algorithm_id, code_id)
        result_path = get_solver_solving_times_path(algorithm_id, code_id)

        if os.path.exists(result_path) and not force_recollect:
            logger.warning(f"Results already collected for algorithm {algorithm_id}, code {code_id}")
            return None

        logger.info(f"Collecting results from {solver_dir}")
        solving_times: Dict[str, float] = {}
        timeouts_or_errors: List[str] = []

        if os.path.isdir(solver_dir):
            for file in os.listdir(solver_dir):
                if file.endswith(".solving.log"):
                    instance_name = file.split(".")[0]
                    instance_time = self.parse_solving_time(f"{solver_dir}/{file}")
                    if instance_time is not None:
                        solving_times[instance_name] = instance_time
                        if instance_time >= 10000:
                            timeouts_or_errors.append(instance_name)
        else:
            logger.warning(f"Solver directory missing: {solver_dir}")

        expected_benchmark_count = 400
        if len(solving_times) < expected_benchmark_count:
            missing_count = expected_benchmark_count - len(solving_times)
            logger.warning(f"Missing results for {missing_count} instances out of {expected_benchmark_count}")

        par2 = _compute_average(list(solving_times.values()))
        logger.info(f"Computed PAR2 for algorithm {algorithm_id}, code {code_id}: {par2}")

        if timeouts_or_errors:
            logger.warning(f"Found {len(timeouts_or_errors)} instances that timed out or had errors")

        # Update code result in database
        code_result = get_code_result(code_id)
        if code_result is not None:
            code_result.par2 = par2
            code_result.build_success = True
            code_result.status = CodeStatus.Evaluated
            update_code_result(code_result)

        with open(result_path, "w") as f:
            json.dump(solving_times, f)
        logger.info(f"Wrote solving times to {result_path}")

        return par2

    def slurm_collect_result(self, slurm_ids: List[int], code_id: str) -> None:
        """Submit SLURM job to collect results after evaluation jobs complete."""
        activate_cmd = _get_activation_cmd()
        code_result = get_code_result(code_id)
        algorithm_id = code_result.algorithm_id
        result_dir = get_solver_result_dir(algorithm_id, code_id)

        cmd = f"{activate_cmd} && python src/llmsat/pipelines/evaluation.py --algorithm_id {algorithm_id} --code_id {code_id} --collect_result"
        output_file = f"{result_dir}/00000000_collect_result.log"

        slurm_cmd = wrap_command_to_slurm(
            cmd,
            output_file=output_file,
            job_name=f"collect_result_{code_id[:8]}",
            dependencies=[str(sid) for sid in slurm_ids],
            dependency_type="afterany"
        )

        slurm_output = os.popen(slurm_cmd).read()
        slurm_id = int(slurm_output.split()[-1])
        logger.info(f"Submitted collect result job {slurm_id}")

    def _compile_with_debugging(
        self,
        solver_path: str,
        current_code: str,
        target_function: str,
        max_debug_rounds: int = 3,
        debug_model: str = "gpt-5.2",
    ) -> tuple:
        """
        Compile solver with optional LLM debugging on failure.

        Returns:
            Tuple of (success: bool, final_code: str or None, all_logs: list)
        """
        debugger = CompilerDebugger(model=debug_model)
        func_info = self.registry[target_function]
        modified_file_path = f"{solver_path}/{func_info.file}"

        try:
            with open(modified_file_path, "r") as f:
                original_file_content = f.read()
        except FileNotFoundError:
            logger.error(f"Target file not found: {modified_file_path}")
            return (False, None, [f"[ERROR] Target file not found: {modified_file_path}"])

        all_logs = []
        code_to_try = current_code

        for attempt in range(max_debug_rounds + 1):
            all_logs.append(f"\n{'='*60}\n=== ATTEMPT {attempt + 1} of {max_debug_rounds + 1} ===\n{'='*60}\n")

            # Restore original file content before injection
            try:
                with open(modified_file_path, "w") as f:
                    f.write(original_file_content)
            except Exception as e:
                logger.error(f"Failed to restore file on attempt {attempt + 1}: {e}")
                return (False, None, all_logs)

            # Inject the code
            try:
                self.injector.replace_function(solver_path, target_function, code_to_try)
                all_logs.append(f"[INFO] Injected {target_function} into solver")
            except Exception as e:
                logger.error(f"Failed to inject code on attempt {attempt + 1}: {e}")
                all_logs.append(f"[ERROR] Failed to inject code: {e}")
                return (False, None, all_logs)

            # Run configure on first attempt only
            if attempt == 0:
                all_logs.append("\n--- ./configure ---")
                configure_proc = subprocess.run(
                    ["./configure"],
                    cwd=solver_path,
                    capture_output=True,
                    text=True,
                )
                all_logs.append(f"[stdout]\n{configure_proc.stdout or '(empty)'}")
                all_logs.append(f"[stderr]\n{configure_proc.stderr or '(empty)'}")

                if configure_proc.returncode != 0:
                    logger.error(f"Configure failed")
                    all_logs.append(f"[FAILED] Configure returned {configure_proc.returncode}")
                    return (False, None, all_logs)
                all_logs.append("[OK] Configure succeeded")

            # Run make
            all_logs.append("\n--- make -j1 ---")
            make_proc = subprocess.run(
                ["make", "-j1"],
                cwd=solver_path,
                capture_output=True,
                text=True,
            )
            all_logs.append(f"[stdout]\n{make_proc.stdout or '(empty)'}")
            all_logs.append(f"[stderr]\n{make_proc.stderr or '(empty)'}")

            if make_proc.returncode == 0:
                logger.info(f"Build succeeded on attempt {attempt + 1}")
                all_logs.append("[SUCCESS] Build completed successfully!")
                return (True, code_to_try, all_logs)

            all_logs.append(f"[FAILED] Make returned {make_proc.returncode}")

            # No more debugging rounds available
            if attempt >= max_debug_rounds:
                logger.warning(f"Build failed after {attempt + 1} attempts")
                all_logs.append("\n[FINAL] No more debugging rounds available")
                break

            # Try LLM debugging
            try:
                with open(modified_file_path, "r") as f:
                    current_file_content = f.read()
            except FileNotFoundError:
                current_file_content = ""

            all_logs.append("\n--- LLM Debugging ---")
            logger.info(f"Build failed on attempt {attempt + 1}, requesting LLM fix...")

            fixed_code = debugger.suggest_fix(
                failing_code=code_to_try,
                compiler_stderr=make_proc.stderr or "",
                current_file_content=current_file_content,
                function_name=target_function,
                function_signature=func_info.signature,
            )

            if fixed_code is None:
                logger.warning(f"Debugger could not suggest a fix")
                all_logs.append("[FAILED] LLM could not suggest a fix")
                break

            all_logs.append(f"[OK] LLM suggested fix")
            code_to_try = fixed_code

        return (False, None, all_logs)

    def build_solver(self, code_result: CodeResult) -> Optional[str]:
        """
        Build a solver with the generated code.

        Returns:
            Path to the built solver directory, or None if build failed.
        """
        logger.info(f"Building solver for code_id={code_result.id}")

        # Get target function from algorithm
        algorithm = get_algorithm_result(code_result.algorithm_id)
        if algorithm is None:
            logger.error(f"Algorithm not found: {code_result.algorithm_id}")
            return None

        target_function = algorithm.target_function
        logger.info(f"Target function: {target_function}")

        # Validate target function is in registry
        if target_function not in self.registry:
            logger.error(f"Target function '{target_function}' not in registry. Available: {self.registry.list_functions()}")
            return None

        # Parse LLM output to extract the function code
        try:
            parsed = self.injector.parse_llm_output(code_result.code, expected_function=target_function)
            new_code = parsed.code
        except ValueError as e:
            logger.error(f"Failed to parse LLM output: {e}")
            return None

        # Copy base solver to new location
        new_solver_path = get_solver_dir(code_result.algorithm_id, code_result.id)
        algorithm_dir = get_algorithm_dir(code_result.algorithm_id)
        os.makedirs(algorithm_dir, exist_ok=True)

        logger.info(f"Building solver at {new_solver_path}")
        if os.path.exists(new_solver_path):
            shutil.rmtree(new_solver_path)
        shutil.copytree(BASE_SOLVER_PATH, new_solver_path)

        # Compile with debugging support
        func_info = self.registry[target_function]
        modified_file = f"{new_solver_path}/{func_info.file}"

        try:
            build_success, final_code, all_logs = self._compile_with_debugging(
                solver_path=new_solver_path,
                current_code=new_code,
                target_function=target_function,
                max_debug_rounds=1,
                debug_model="gpt-5.2",
            )

            # Write build log
            output = "\n".join(all_logs)
            build_log_path = f"{algorithm_dir}/code_{code_result.id}.build.log"
            with open(build_log_path, "w") as f:
                f.write(output)
            logger.info(f"Wrote build log to {build_log_path}")

            # Copy the modified source file to the algorithm directory
            modified_file_name = os.path.basename(func_info.file)
            shutil.copy2(modified_file, f"{algorithm_dir}/code_{code_result.id}.{modified_file_name}")

        except Exception as e:
            logger.error(f"Compilation with debugging failed: {e}")
            return None

        if build_success:
            new_solver_bin_path = f"{new_solver_path}/build/kissat"
            try:
                shutil.copy2(new_solver_bin_path, f"{new_solver_path}/kissat")
                logger.info(f"Build succeeded, binary copied to {new_solver_path}/kissat")
            except Exception:
                pass
            return new_solver_path
        else:
            logger.warning(f"Build failed for solver at {new_solver_path}")
            return None

    def slurm_run_evaluate(self, solver_path: str, benchmark_path: str, result_dir: str, max_jobs: int = 200) -> List[int]:
        """
        Submit solver evaluation using a SLURM job array.

        Returns:
            List containing the SLURM job array ID, or empty list on failure.
        """
        logger.info(f"Submitting SLURM job array for solver {solver_path}")
        os.makedirs(result_dir, exist_ok=True)

        # Collect CNF files to evaluate (skip already completed ones)
        cnf_files = []
        jobs_skipped = 0
        for benchmark_file in sorted(os.listdir(benchmark_path)):
            if benchmark_file.endswith(".cnf"):
                if os.path.exists(f"{result_dir}/{benchmark_file}.solving.log"):
                    jobs_skipped += 1
                    continue
                cnf_files.append(benchmark_file)

        if not cnf_files:
            logger.info(f"All {jobs_skipped} benchmarks already evaluated")
            return []

        if len(cnf_files) > max_jobs:
            logger.warning(f"Limiting evaluation to {max_jobs} benchmarks (out of {len(cnf_files)} remaining)")
            cnf_files = cnf_files[:max_jobs]

        # Write CNF file list
        cnf_list_path = f"{result_dir}/cnf_file_list.txt"
        with open(cnf_list_path, "w") as f:
            for cnf_file in cnf_files:
                f.write(f"{cnf_file}\n")
        logger.info(f"Wrote {len(cnf_files)} CNF files to {cnf_list_path}")

        # Create wrapper script for job array
        script_path = f"{result_dir}/run_solver_array.sh"
        script_content = f"""#!/bin/bash
CNF_LIST="{cnf_list_path}"
SOLVER="{solver_path}/build/kissat"
BENCHMARK_PATH="{benchmark_path}"
RESULT_DIR="{result_dir}"

CNF_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CNF_LIST")

if [ -z "$CNF_FILE" ]; then
    echo "ERROR: No CNF file found for array task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Running solver on $CNF_FILE (array task $SLURM_ARRAY_TASK_ID)"
"$SOLVER" "$BENCHMARK_PATH/$CNF_FILE" > "$RESULT_DIR/$CNF_FILE.solving.log" 2>&1
EXIT_CODE=$?
echo "Solver finished with exit code $EXIT_CODE"
exit $EXIT_CODE
"""
        with open(script_path, "w") as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)

        # Submit job array
        array_range = f"0-{len(cnf_files) - 1}"
        slurm_cmd = wrap_command_to_slurm_array(
            script_path=script_path,
            array_range=array_range,
            mem="8G",
            time="01:23:20",
            job_name=f"solve_array",
            output_file=f"{result_dir}/slurm_array_%a.log",
            max_concurrent=100,
        )
        logger.info(f"Submitting job array: {slurm_cmd}")

        try:
            slurm_output = os.popen(slurm_cmd).read().strip()
            if not slurm_output or "error" in slurm_output.lower():
                logger.error(f"Failed to submit job array: {slurm_output}")
                return []
            slurm_id = int(slurm_output.split()[-1])
            logger.info(f"Submitted job array {slurm_id} with {len(cnf_files)} tasks ({jobs_skipped} skipped)")
            return [slurm_id]
        except (ValueError, IndexError) as e:
            logger.error(f"Failed to parse SLURM job ID: {e}")
            return []

    def run_single_solver(self, code_id: str, build_only: bool = False) -> None:
        """Build and evaluate a single code result."""
        code_result = get_code_result(code_id)
        if code_result is None:
            logger.error(f"Code result not found for code_id={code_id}")
            return

        if not build_only and code_result.status == CodeStatus.Evaluating:
            logger.warning(f"Code result {code_id} is already evaluating, skipping")
            return

        logger.info(f"Building solver for code_id={code_id}, algorithm_id={code_result.algorithm_id}")

        solver_path = self.build_solver(code_result)
        if solver_path is not None:
            logger.info(f"Solver built successfully: {solver_path}")
            if build_only:
                logger.info(f"Build-only mode: skipping SLURM evaluation")
                return
            result_dir = get_solver_result_dir(code_result.algorithm_id, code_result.id)
            slurm_ids = self.slurm_run_evaluate(solver_path, SAT2025_BENCHMARK_PATH, result_dir)
            code_result.status = CodeStatus.Evaluating
            update_code_result(code_result)
            if slurm_ids:
                self.slurm_collect_result(slurm_ids, code_id)
        else:
            code_result.status = CodeStatus.BuildFailed
            update_code_result(code_result)
            logger.error("Solver build failed")

    def run_all_solvers(self, algorithm_id: str, build_only: bool = False) -> None:
        """Build and evaluate all code results for an algorithm."""
        logger.info(f"Running evaluation for algorithm {algorithm_id}")
        algorithm = get_algorithm_result(algorithm_id)
        if algorithm is None:
            logger.error(f"Algorithm not found: {algorithm_id}")
            return

        os.makedirs(f"solvers/algorithm_{algorithm_id}", exist_ok=True)

        code_id_list = algorithm.code_id_list or []
        logger.info(f"Found {len(code_id_list)} code ids to evaluate for algorithm {algorithm_id}")

        for code_id in code_id_list:
            logger.info(f"Starting build for code_id={code_id}")
            self.run_single_solver(code_id, build_only=build_only)

        if not build_only:
            algorithm.status = AlgorithmStatus.Evaluating
            update_algorithm_result(algorithm)


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Evaluate SAT solver variants")
    parser.add_argument("--algorithm_id", type=str, help="Single algorithm ID to evaluate")
    parser.add_argument("--code_id", type=str, help="Single code ID (used with --collect_result)")
    parser.add_argument("--first_n", type=int, help="Only evaluate first N algorithms")
    parser.add_argument("--run_all", action="store_true", help="Evaluate all algorithms in generation tag")
    parser.add_argument("--collect_result", action="store_true", help="Collect results for a single algorithm/code")
    parser.add_argument("--collect_all_results", action="store_true", help="Collect results for all algorithms in generation tag")
    parser.add_argument("--generation_tag", type=str, help="Generation tag to evaluate")
    parser.add_argument("--build-only", action="store_true", help="Build solvers but skip SLURM evaluation")
    args = parser.parse_args()

    evaluation_pipeline = EvaluationPipeline()

    if args.collect_result:
        if not args.algorithm_id or not args.code_id:
            logger.error("--collect_result requires both --algorithm_id and --code_id")
            return
        evaluation_pipeline.collect_results(args.algorithm_id, args.code_id, force_recollect=True)
        return

    if args.collect_all_results:
        if not args.generation_tag:
            logger.error("--collect_all_results requires --generation_tag")
            return
        algorithm_ids = get_ids_from_router_table(CHATGPT_DATA_GENERATION_TABLE, args.generation_tag)
        logger.info(f"Collecting results for {len(algorithm_ids)} algorithms")
        for algorithm_id in algorithm_ids:
            algorithm_result = get_algorithm_result(algorithm_id)
            if algorithm_result and algorithm_result.code_id_list:
                for code_id in algorithm_result.code_id_list:
                    evaluation_pipeline.collect_results(algorithm_id, code_id, force_recollect=True)
        return

    if args.run_all:
        if not args.generation_tag:
            logger.error("--run_all requires --generation_tag")
            return
        algorithm_ids = get_ids_from_router_table(CHATGPT_DATA_GENERATION_TABLE, args.generation_tag)
        algorithms = [get_algorithm_result(aid) for aid in algorithm_ids]
        algorithms = [a for a in algorithms if a is not None]
        logger.info(f"Found {len(algorithms)} algorithms to evaluate")
        if args.first_n:
            algorithms = algorithms[:args.first_n]
    elif args.algorithm_id:
        algorithms = [get_algorithm_result(args.algorithm_id)]
        algorithms = [a for a in algorithms if a is not None]
    else:
        logger.error("Must specify --run_all with --generation_tag, or --algorithm_id")
        return

    logger.info(f"Running {'build' if args.build_only else 'evaluation'} for {len(algorithms)} algorithms")
    for algorithm in algorithms:
        logger.info(f"Processing algorithm: {algorithm.id}")
        evaluation_pipeline.run_all_solvers(algorithm.id, build_only=args.build_only)


if __name__ == "__main__":
    main()
