from __future__ import annotations

import os
import shutil
import json
import subprocess
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
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

# SLURM Configuration for Compute Canada
SLURM_ACCOUNT = "def-vganesh"
SLURM_TIMEOUT_SECONDS = 5000           # 83 min 20 sec per CNF
SLURM_WALL_TIME = "01:30:00"           # 90 min (timeout + buffer)
SLURM_MEMORY = "4G"
SLURM_MAX_CONCURRENT = 100
SLURM_MAX_ARRAY_SIZE = 1000
PAR2_PENALTY = 10000                   # 2× timeout for unsolved


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

    def __init__(self, registry_path: str = DEFAULT_REGISTRY_PATH, generation_tag: str = None):
        self.registry = FunctionRegistry(registry_path)
        self.injector = FunctionInjector(self.registry, BASE_SOLVER_PATH)
        self.generation_tag = generation_tag
        logger.info(f"Initialized FunctionRegistry with {len(self.registry)} functions: {self.registry.list_functions()}")

    def parse_solving_time(self, file_path: str) -> Optional[float]:
        """Parse solving time from a log file. Returns PAR2_PENALTY on timeout/error."""
        try:
            lines = open(file_path, "r").readlines()
        except Exception as e:
            logger.warning(f"Failed to read log file {file_path}: {e}")
            return PAR2_PENALTY

        if not lines:
            logger.warning(f"Empty log file (likely timeout or crash): {file_path}")
            return PAR2_PENALTY

        for line in reversed(lines):
            if "process-time" in line:
                match = re.search(r'(\d+\.?\d*)\s+seconds', line)
                if match:
                    return float(match.group(1))
            if "error" in line.lower():
                logger.warning(f"Error found in solving log: {file_path}")
                return PAR2_PENALTY
            # Detect timeout from our wrapper script or SLURM
            if "TIMEOUT" in line or "timeout" in line.lower():
                logger.warning(f"Timeout detected: {file_path}")
                return PAR2_PENALTY
            if "CANCELLED" in line or "TIME LIMIT" in line:
                logger.warning(f"SLURM timeout/cancellation detected: {file_path}")
                return PAR2_PENALTY

        logger.warning(f"No process-time found in log (incomplete run): {file_path}")
        return PAR2_PENALTY

    def collect_results(self, algorithm_id: str, code_id: str, force_recollect: bool = False) -> Optional[float]:
        """Collect evaluation results from solver logs and compute PAR2 score."""
        algorithm = get_algorithm_result(algorithm_id)
        parent_id = algorithm.parent_id if algorithm else None
        solver_dir = get_solver_result_dir(algorithm_id, code_id,
                                           generation_tag=self.generation_tag, parent_id=parent_id)
        result_path = get_solver_solving_times_path(algorithm_id, code_id,
                                                    generation_tag=self.generation_tag, parent_id=parent_id)

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
                        if instance_time >= PAR2_PENALTY:
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
        algorithm = get_algorithm_result(algorithm_id)
        parent_id = algorithm.parent_id if algorithm else None
        result_dir = get_solver_result_dir(algorithm_id, code_id,
                                           generation_tag=self.generation_tag, parent_id=parent_id)

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
        new_solver_path = get_solver_dir(code_result.algorithm_id, code_result.id,
                                         generation_tag=self.generation_tag, parent_id=algorithm.parent_id)
        algorithm_dir = get_algorithm_dir(code_result.algorithm_id,
                                          generation_tag=self.generation_tag, parent_id=algorithm.parent_id)
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
            build_log_path = f"{algorithm_dir}code_{code_result.id}.build.log"
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

    def slurm_run_evaluate(
        self,
        solver_path: str,
        benchmark_path: str,
        result_dir: str,
        max_jobs: int = 400,
        dry_run: bool = False,
        timeout: int = SLURM_TIMEOUT_SECONDS,
    ) -> List[int]:
        """
        Submit solver evaluation using a SLURM job array.

        Args:
            solver_path: Path to the built solver directory
            benchmark_path: Path to the benchmark CNF files
            result_dir: Directory to store evaluation results
            max_jobs: Maximum number of CNF files to evaluate
            dry_run: If True, print SLURM commands without submitting
            timeout: Timeout per CNF in seconds (default: SLURM_TIMEOUT_SECONDS)

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

        # Create wrapper script for job array with timeout
        script_path = f"{result_dir}/run_solver_array.sh"
        script_content = f"""#!/bin/bash
CNF_LIST="{cnf_list_path}"
SOLVER="{solver_path}/build/kissat"
BENCHMARK_PATH="{benchmark_path}"
RESULT_DIR="{result_dir}"
TIMEOUT={timeout}

CNF_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CNF_LIST")
OUTPUT_FILE="${{RESULT_DIR}}/${{CNF_FILE}}.solving.log"

if [ -z "$CNF_FILE" ]; then
    echo "ERROR: No CNF file found for array task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Skip if already done
if [ -f "$OUTPUT_FILE" ]; then
    echo "Already completed: $CNF_FILE"
    exit 0
fi

echo "Running solver on $CNF_FILE (array task $SLURM_ARRAY_TASK_ID)"

# Run solver with timeout
timeout ${{TIMEOUT}}s "$SOLVER" "$BENCHMARK_PATH/$CNF_FILE" > "$OUTPUT_FILE" 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "TIMEOUT after ${{TIMEOUT}}s" >> "$OUTPUT_FILE"
fi

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
            account=SLURM_ACCOUNT,
            mem=SLURM_MEMORY,
            time=SLURM_WALL_TIME,
            job_name=f"solve_array",
            output_file=f"{result_dir}/slurm_array_%a.log",
            max_concurrent=SLURM_MAX_CONCURRENT,
        )
        logger.info(f"SLURM command: {slurm_cmd}")

        if dry_run:
            logger.info(f"[DRY-RUN] Would submit job array with {len(cnf_files)} tasks ({jobs_skipped} skipped)")
            print(f"[DRY-RUN] SLURM command:\n{slurm_cmd}")
            return []

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

    def slurm_run_evaluate_batch(
        self,
        solver_tasks: List[Tuple[str, str, str]],  # (solver_path, result_dir, code_id)
        benchmark_path: str,
        cnf_files: List[str],
        dry_run: bool = False,
        timeout: int = SLURM_TIMEOUT_SECONDS,
    ) -> List[int]:
        """
        Submit ALL (solver × CNF) pairs as unified job arrays.
        More efficient than one array per solver.

        Args:
            solver_tasks: List of (solver_path, result_dir, code_id) tuples
            benchmark_path: Path to the benchmark CNF files
            cnf_files: List of CNF file names to evaluate
            dry_run: If True, print SLURM commands without submitting
            timeout: Timeout per CNF in seconds

        Returns:
            List of SLURM job array IDs
        """
        if not solver_tasks or not cnf_files:
            logger.warning("No solver tasks or CNF files to evaluate")
            return []

        # Build flat list of (solver_path, result_dir, code_id, cnf_file) tasks
        all_tasks = []
        for solver_path, result_dir, code_id in solver_tasks:
            os.makedirs(result_dir, exist_ok=True)
            for cnf_file in cnf_files:
                # Skip already completed ones
                if os.path.exists(f"{result_dir}/{cnf_file}.solving.log"):
                    continue
                all_tasks.append((solver_path, result_dir, code_id, cnf_file))

        if not all_tasks:
            logger.info("All tasks already completed")
            return []

        logger.info(f"Total tasks to submit: {len(all_tasks)} (from {len(solver_tasks)} solvers × {len(cnf_files)} CNFs)")

        # Split into chunks of SLURM_MAX_ARRAY_SIZE
        job_ids = []
        for chunk_start in range(0, len(all_tasks), SLURM_MAX_ARRAY_SIZE):
            chunk_end = min(chunk_start + SLURM_MAX_ARRAY_SIZE, len(all_tasks))
            chunk_tasks = all_tasks[chunk_start:chunk_end]
            chunk_num = chunk_start // SLURM_MAX_ARRAY_SIZE

            # Create a directory for this batch
            batch_dir = f"/tmp/slurm_batch_{chunk_num}"
            os.makedirs(batch_dir, exist_ok=True)

            # Write task list file
            task_list_path = f"{batch_dir}/task_list.txt"
            with open(task_list_path, "w") as f:
                for solver_path, result_dir, code_id, cnf_file in chunk_tasks:
                    f.write(f"{solver_path}\t{result_dir}\t{cnf_file}\n")

            # Create wrapper script
            script_path = f"{batch_dir}/run_batch_array.sh"
            script_content = f"""#!/bin/bash
TASK_LIST="{task_list_path}"
BENCHMARK_PATH="{benchmark_path}"
TIMEOUT={timeout}

# Read task info for this array index
TASK_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_LIST")
SOLVER_PATH=$(echo "$TASK_LINE" | cut -f1)
RESULT_DIR=$(echo "$TASK_LINE" | cut -f2)
CNF_FILE=$(echo "$TASK_LINE" | cut -f3)

SOLVER="${{SOLVER_PATH}}/build/kissat"
OUTPUT_FILE="${{RESULT_DIR}}/${{CNF_FILE}}.solving.log"

if [ -z "$CNF_FILE" ]; then
    echo "ERROR: No task found for array task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Skip if already done
if [ -f "$OUTPUT_FILE" ]; then
    echo "Already completed: $CNF_FILE"
    exit 0
fi

echo "Running solver on $CNF_FILE (array task $SLURM_ARRAY_TASK_ID)"

# Run solver with timeout
timeout ${{TIMEOUT}}s "$SOLVER" "$BENCHMARK_PATH/$CNF_FILE" > "$OUTPUT_FILE" 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "TIMEOUT after ${{TIMEOUT}}s" >> "$OUTPUT_FILE"
fi

echo "Solver finished with exit code $EXIT_CODE"
exit $EXIT_CODE
"""
            with open(script_path, "w") as f:
                f.write(script_content)
            os.chmod(script_path, 0o755)

            # Submit job array
            array_range = f"0-{len(chunk_tasks) - 1}"
            slurm_cmd = wrap_command_to_slurm_array(
                script_path=script_path,
                array_range=array_range,
                account=SLURM_ACCOUNT,
                mem=SLURM_MEMORY,
                time=SLURM_WALL_TIME,
                job_name=f"batch_{chunk_num}",
                output_file=f"{batch_dir}/slurm_array_%a.log",
                max_concurrent=SLURM_MAX_CONCURRENT,
            )
            logger.info(f"SLURM command for chunk {chunk_num}: {slurm_cmd}")

            if dry_run:
                logger.info(f"[DRY-RUN] Would submit batch {chunk_num} with {len(chunk_tasks)} tasks")
                print(f"[DRY-RUN] SLURM command:\n{slurm_cmd}")
                continue

            try:
                slurm_output = os.popen(slurm_cmd).read().strip()
                if not slurm_output or "error" in slurm_output.lower():
                    logger.error(f"Failed to submit batch {chunk_num}: {slurm_output}")
                    continue
                slurm_id = int(slurm_output.split()[-1])
                logger.info(f"Submitted batch {chunk_num} job array {slurm_id} with {len(chunk_tasks)} tasks")
                job_ids.append(slurm_id)
            except (ValueError, IndexError) as e:
                logger.error(f"Failed to parse SLURM job ID for batch {chunk_num}: {e}")

        return job_ids

    def run_single_solver(self, code_id: str, build_only: bool = False, dry_run: bool = False, skip_build: bool = False) -> Optional[Tuple[str, str, str]]:
        """
        Build and evaluate a single code result.

        Args:
            code_id: The code result ID to build and evaluate
            build_only: If True, build the solver but don't submit SLURM evaluation
            dry_run: If True, print SLURM commands without submitting
            skip_build: If True, assume solver is already built and skip build step

        Returns:
            Tuple of (solver_path, result_dir, code_id) if build succeeded, None otherwise.
        """
        code_result = get_code_result(code_id)
        if code_result is None:
            logger.error(f"Code result not found for code_id={code_id}")
            return None

        if not build_only and not dry_run and code_result.status == CodeStatus.Evaluating:
            logger.warning(f"Code result {code_id} is already evaluating, skipping")
            return None

        algorithm = get_algorithm_result(code_result.algorithm_id)
        if algorithm is None:
            logger.error(f"Algorithm not found: {code_result.algorithm_id}")
            return None

        # Check if we can skip the build
        if skip_build:
            solver_path = get_solver_dir(code_result.algorithm_id, code_result.id,
                                         generation_tag=self.generation_tag, parent_id=algorithm.parent_id)
            solver_binary = os.path.join(solver_path, "kissat")
            if os.path.exists(solver_binary):
                logger.info(f"Skipping build, solver already exists: {solver_binary}")
            else:
                logger.warning(f"skip_build=True but solver not found at {solver_binary}, skipping this solver")
                return None
        else:
            logger.info(f"Building solver for code_id={code_id}, algorithm_id={code_result.algorithm_id}")
            solver_path = self.build_solver(code_result)
        if solver_path is not None:
            if not skip_build:
                logger.info(f"Solver built successfully: {solver_path}")
            result_dir = get_solver_result_dir(code_result.algorithm_id, code_result.id,
                                               generation_tag=self.generation_tag, parent_id=algorithm.parent_id)

            if build_only:
                logger.info(f"Build-only mode: skipping SLURM evaluation")
                return (solver_path, result_dir, code_id)

            slurm_ids = self.slurm_run_evaluate(solver_path, SAT2025_BENCHMARK_PATH, result_dir, dry_run=dry_run)

            if not dry_run:
                code_result.status = CodeStatus.Evaluating
                update_code_result(code_result)
                if slurm_ids:
                    self.slurm_collect_result(slurm_ids, code_id)

            return (solver_path, result_dir, code_id)
        else:
            if not dry_run:
                code_result.status = CodeStatus.BuildFailed
                update_code_result(code_result)
            logger.error("Solver build failed")
            return None

    def run_all_solvers(self, algorithm_id: str, build_only: bool = False, dry_run: bool = False, skip_build: bool = False) -> None:
        """Build and evaluate all code results for an algorithm."""
        logger.info(f"Running evaluation for algorithm {algorithm_id}")
        algorithm = get_algorithm_result(algorithm_id)
        if algorithm is None:
            logger.error(f"Algorithm not found: {algorithm_id}")
            return

        algorithm_dir = get_algorithm_dir(algorithm_id,
                                          generation_tag=self.generation_tag, parent_id=algorithm.parent_id)
        os.makedirs(algorithm_dir, exist_ok=True)

        code_id_list = algorithm.code_id_list or []
        logger.info(f"Found {len(code_id_list)} code ids to evaluate for algorithm {algorithm_id}")

        for code_id in code_id_list:
            logger.info(f"Starting {'evaluation' if skip_build else 'build'} for code_id={code_id}")
            self.run_single_solver(code_id, build_only=build_only, dry_run=dry_run, skip_build=skip_build)

        if not build_only and not dry_run:
            algorithm.status = AlgorithmStatus.Evaluating
            update_algorithm_result(algorithm)

    def run_all_solvers_batch(
        self,
        algorithms: List,
        build_only: bool = False,
        dry_run: bool = False,
        skip_build: bool = False,
    ) -> None:
        """
        Build all solvers and submit evaluations in efficient batches.

        This method builds all solvers first, then submits all evaluations
        as unified job arrays for efficiency.

        Args:
            algorithms: List of algorithm results to evaluate
            build_only: If True, build solvers but skip SLURM evaluation
            dry_run: If True, print SLURM commands without submitting
            skip_build: If True, assume solvers are already built and skip build step
        """
        # Collect CNF files from benchmark
        cnf_files = []
        for benchmark_file in sorted(os.listdir(SAT2025_BENCHMARK_PATH)):
            if benchmark_file.endswith(".cnf"):
                cnf_files.append(benchmark_file)
        logger.info(f"Found {len(cnf_files)} CNF files in benchmark")

        # Build all solvers and collect successful ones
        solver_tasks = []
        for algorithm in algorithms:
            algorithm_id = algorithm.id
            logger.info(f"Processing algorithm: {algorithm_id}")

            algorithm_dir = get_algorithm_dir(algorithm_id,
                                              generation_tag=self.generation_tag, parent_id=algorithm.parent_id)
            os.makedirs(algorithm_dir, exist_ok=True)

            code_id_list = algorithm.code_id_list or []
            logger.info(f"Found {len(code_id_list)} code ids for algorithm {algorithm_id}")

            for code_id in code_id_list:
                code_result = get_code_result(code_id)
                if code_result is None:
                    logger.error(f"Code result not found for code_id={code_id}")
                    continue

                if skip_build:
                    # Check if solver already exists
                    solver_path = get_solver_dir(algorithm_id, code_id,
                                                 generation_tag=self.generation_tag, parent_id=algorithm.parent_id)
                    solver_binary = os.path.join(solver_path, "kissat")
                    if os.path.exists(solver_binary):
                        logger.info(f"Skipping build, solver already exists: {solver_binary}")
                    else:
                        logger.warning(f"skip_build=True but solver not found at {solver_binary}, skipping")
                        continue
                else:
                    logger.info(f"Building solver for code_id={code_id}")
                    solver_path = self.build_solver(code_result)
                    if solver_path is None:
                        if not dry_run:
                            code_result.status = CodeStatus.BuildFailed
                            update_code_result(code_result)
                        logger.error(f"Solver build failed for code_id={code_id}")
                        continue
                    logger.info(f"Solver built successfully: {solver_path}")

                result_dir = get_solver_result_dir(algorithm_id, code_id,
                                                   generation_tag=self.generation_tag, parent_id=algorithm.parent_id)
                solver_tasks.append((solver_path, result_dir, code_id))

        logger.info(f"Built {len(solver_tasks)} solvers successfully")

        if build_only:
            logger.info("Build-only mode: skipping SLURM evaluation")
            return

        if not solver_tasks:
            logger.warning("No solvers built successfully, nothing to evaluate")
            return

        # Submit all evaluations in batch
        job_ids = self.slurm_run_evaluate_batch(solver_tasks, SAT2025_BENCHMARK_PATH, cnf_files, dry_run=dry_run)

        if not dry_run:
            # Update status for all code results
            for solver_path, result_dir, code_id in solver_tasks:
                code_result = get_code_result(code_id)
                if code_result:
                    code_result.status = CodeStatus.Evaluating
                    update_code_result(code_result)

            # Update algorithm status
            for algorithm in algorithms:
                algorithm.status = AlgorithmStatus.Evaluating
                update_algorithm_result(algorithm)

            # Submit collect result jobs (one per code_id, dependent on all evaluation jobs)
            if job_ids:
                for solver_path, result_dir, code_id in solver_tasks:
                    self.slurm_collect_result(job_ids, code_id)

        logger.info(f"Batch submission complete. Job IDs: {job_ids}")


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
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Print SLURM commands without submitting")
    parser.add_argument("--timeout", type=int, default=5000,
                        help="Timeout per CNF in seconds (default: 5000)")
    parser.add_argument("--max-concurrent", type=int, default=100,
                        help="Max concurrent SLURM tasks per array (default: 100)")
    parser.add_argument("--batch-mode", action="store_true",
                        help="Use batch submission mode (all solvers × CNFs in unified arrays)")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip build step, assume solvers are already built")
    args = parser.parse_args()

    evaluation_pipeline = EvaluationPipeline(generation_tag=args.generation_tag)

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

    if args.build_only:
        mode_str = "build"
    elif args.skip_build:
        mode_str = "evaluation (skip-build)"
    elif args.dry_run:
        mode_str = "dry-run"
    else:
        mode_str = "evaluation"
    logger.info(f"Running {mode_str} for {len(algorithms)} algorithms")

    if args.batch_mode and not args.build_only:
        # Use batch submission for efficiency
        logger.info("Using batch submission mode")
        evaluation_pipeline.run_all_solvers_batch(algorithms, build_only=args.build_only, dry_run=args.dry_run, skip_build=args.skip_build)
    else:
        # Use per-algorithm submission (original behavior)
        for algorithm in algorithms:
            logger.info(f"Processing algorithm: {algorithm.id}")
            evaluation_pipeline.run_all_solvers(algorithm.id, build_only=args.build_only, dry_run=args.dry_run, skip_build=args.skip_build)


if __name__ == "__main__":
    main()
