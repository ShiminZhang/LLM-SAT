from __future__ import annotations

import os
import shutil
import json
import subprocess
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import argparse

from llmsat.llmsat import (
    CHATGPT_DATA_GENERATION_TABLE,
    CodeResult,
    CodeStatus,
    AlgorithmResult,
    AlgorithmStatus,
    Role,
    BASE_SOLVER_PATH,
    BASELINE_PAR2,
    SAT2025_BENCHMARK_PATH,
    get_logger,
    setup_logging,
    normalize_par2,
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
    get_generation_output_dir,
)
from llmsat.utils.utils import wrap_command_to_slurm, wrap_command_to_slurm_array
from llmsat.code_injection import FunctionRegistry, FunctionInjector
from llmsat.debugging import CompilerDebugger
from llmsat.config import DEFAULT_MODEL

logger = get_logger(__name__)

DEFAULT_REGISTRY_PATH = "solvers/base/function_registry.yaml"

# SLURM Configuration for Compute Canada
SLURM_ACCOUNT = "def-vganesh"
SLURM_TIMEOUT_SECONDS = 5000           # 83 min 20 sec per CNF
SLURM_WALL_TIME = "01:30:00"           # 90 min (timeout + buffer)
SLURM_MEMORY = "8G"
SLURM_MAX_CONCURRENT = 1000 
SLURM_MAX_ARRAY_SIZE = 1000
PAR2_PENALTY = 10000                   # 2× timeout for unsolved

# Quick-eval mode: 100 representative CNFs, reduced timeout
QUICK_EVAL_TIMEOUT_SECONDS = 600 
QUICK_EVAL_WALL_TIME = "00:12:00"       
QUICK_EVAL_PAR2_PENALTY = 1200 # 2× quick timeout
QUICK_EVAL_BENCHMARK_LIST = "data/benchmarks/satcomp2025_quick50.txt"
EVALUATION_SOLVER_FLAGS = "-s --check=1"

INSTANCE_CATEGORIES_PATH = "data/benchmarks/instance_categories.json"
# Baseline time threshold (seconds) for easy/hard split. Adjust as needed.
DIFFICULTY_CUTOFF = 100
# Reject PAR2 scores below this threshold (solver likely giving wrong answers)
MIN_PAR2_THRESHOLD = 400


def _compute_average(values: List[float]) -> Optional[float]:
    """Compute average of non-None values."""
    non_none = [v for v in values if v is not None]
    if not non_none:
        return None
    return sum(non_none) / len(non_none)


def _compute_par2_breakdown(
    solving_times: Dict[str, float],
    instance_categories: Dict[str, dict],
    difficulty_cutoff: float = DIFFICULTY_CUTOFF,
) -> Dict[str, Optional[float]]:
    """Compute PAR2 scores broken down by category.

    Returns dict with keys: all, easy, hard, sat, unsat.
    Categories are determined by instance_categories (satisfiability + baseline_time).
    Instances not in instance_categories are included in 'all' only.
    """
    buckets: Dict[str, List[float]] = {"all": [], "easy": [], "hard": [], "sat": [], "unsat": []}

    for instance_name, time_val in solving_times.items():
        buckets["all"].append(time_val)

        cat = instance_categories.get(instance_name)
        if cat is None:
            continue

        baseline_time = cat.get("baseline_time")
        if baseline_time is not None:
            if baseline_time < difficulty_cutoff:
                buckets["easy"].append(time_val)
            else:
                buckets["hard"].append(time_val)

        sat_status = cat.get("satisfiability")
        if sat_status == "SAT":
            buckets["sat"].append(time_val)
        elif sat_status == "UNSAT":
            buckets["unsat"].append(time_val)

    return {k: _compute_average(v) for k, v in buckets.items()}


def _get_activation_cmd() -> str:
    """Return shell command to activate Python environment."""
    from llmsat.config import PYTHON_ACTIVATE_PATH
    return f"source {PYTHON_ACTIVATE_PATH}"


@dataclass
class EvaluationPipeline:
    """Evaluation pipeline for building and evaluating SAT solver variants."""

    def __init__(self, registry_path: str = DEFAULT_REGISTRY_PATH, generation_tag: str = None):
        self.registry = FunctionRegistry(registry_path)
        self.injector = FunctionInjector(self.registry, BASE_SOLVER_PATH)
        self.generation_tag = generation_tag
        # Evaluation parameters (overridden by --quick-eval or --timeout)
        self.timeout = SLURM_TIMEOUT_SECONDS
        self.wall_time = SLURM_WALL_TIME
        self.par2_penalty = PAR2_PENALTY
        self.cnf_files = None  # None = scan benchmark directory; list = use specific files
        # Load instance categories for PAR2 breakdown (SAT/UNSAT, easy/hard)
        if os.path.exists(INSTANCE_CATEGORIES_PATH):
            with open(INSTANCE_CATEGORIES_PATH) as f:
                self.instance_categories = json.load(f)
        else:
            logger.warning(f"Instance categories not found at {INSTANCE_CATEGORIES_PATH}; PAR2 breakdown disabled")
            self.instance_categories = {}
        logger.info(f"Initialized FunctionRegistry with {len(self.registry)} functions: {self.registry.list_functions()}")

    def parse_solving_time(self, file_path: str) -> Optional[float]:
        """Parse solving time from a log file. Returns self.par2_penalty on timeout/error."""
        penalty = self.par2_penalty
        try:
            lines = open(file_path, "r").readlines()
        except Exception as e:
            logger.warning(f"Failed to read log file {file_path}: {e}")
            return penalty

        if not lines:
            logger.warning(f"Empty log file (likely timeout or crash): {file_path}")
            return penalty

        for line in reversed(lines):
            if "process-time" in line:
                match = re.search(r'(\d*\.?\d+)\s+seconds', line)
                if match:
                    return float(match.group(1))
            if "error" in line.lower():
                logger.warning(f"Error found in solving log: {file_path}")
                return penalty
            # Detect timeout from our wrapper script or SLURM
            if "TIMEOUT" in line or "timeout" in line.lower():
                logger.warning(f"Timeout detected: {file_path}")
                return penalty
            if "CANCELLED" in line or "TIME LIMIT" in line:
                logger.warning(f"SLURM timeout/cancellation detected: {file_path}")
                return penalty

        logger.warning(f"No process-time found in log (incomplete run): {file_path}")
        return penalty

    TRACKED_STATS = [
        "conflicts", "decisions", "propagations", "restarts",
        "switched", "chronological", "reductions", "eliminated",
        "rephased", "vivified", "walks", "reordered", "learning_rate", "propagation_rate"
    ]

    def parse_solver_stats(self, file_path: str) -> Optional[Dict[str, Union[int, float]]]:
        """Parse solver internal statistics from a kissat .solving.log file."""
        try:
            with open(file_path, "r") as f:
                content = f.read()
        except Exception:
            return None

        stats_start = content.find("---- [ statistics ]")
        if stats_start == -1:
            return None

        stats_section = content[stats_start:]
        stats = {}
        for match in re.finditer(r'^c\s+([\w_]+):\s+(\d+(?:\.\d+)?)', stats_section, re.MULTILINE):
            name = match.group(1)
            if name in self.TRACKED_STATS:
                value = match.group(2)
                stats[name] = float(value) if "." in value else int(value)

        return stats if stats else None

    def collect_results(self, algorithm_id: str, code_id: str, force_recollect: bool = False) -> Optional[float]:
        """Collect evaluation results from solver logs and compute PAR2 score."""
        algorithm = get_algorithm_result(algorithm_id)
        # Patch mutation_step from the generation-time map file (not stored in DB).
        if algorithm is not None and algorithm.mutation_step is None:
            map_path = os.path.join(
                get_generation_output_dir(self.generation_tag), "mutation_step_map.jsonl"
            )
            if os.path.exists(map_path):
                try:
                    with open(map_path) as _f:
                        for _line in _f:
                            _entry = json.loads(_line.strip())
                            if algorithm_id in _entry:
                                algorithm.mutation_step = _entry[algorithm_id]
                                break
                except Exception:
                    pass
        existing_code_result = get_code_result(code_id)
        algo_role = algorithm.role if algorithm else None
        solver_dir = get_solver_result_dir(algorithm_id, code_id,
                                           generation_tag=self.generation_tag, role=algo_role,
                                           mkdir=False)
        result_path = get_solver_solving_times_path(algorithm_id, code_id,
                                                    generation_tag=self.generation_tag, role=algo_role)

        if os.path.exists(result_path) and not force_recollect:
            logger.warning(f"Results already collected for algorithm {algorithm_id}, code {code_id}")
            return None

        logger.info(f"Collecting results from {solver_dir}")
        solving_times: Dict[str, float] = {}
        solver_stats: Dict[str, Dict[str, Union[int, float]]] = {}
        timeouts_or_errors: List[str] = []
        min_mtime: Optional[float] = None
        max_mtime: Optional[float] = None

        if os.path.isdir(solver_dir):
            for file in os.listdir(solver_dir):
                if file.endswith(".solving.log"):
                    instance_name = file.split(".")[0]
                    log_path = f"{solver_dir}/{file}"
                    instance_time = self.parse_solving_time(log_path)
                    if instance_time is not None:
                        solving_times[instance_name] = instance_time
                        if instance_time >= self.par2_penalty:
                            timeouts_or_errors.append(instance_name)
                    instance_stats = self.parse_solver_stats(log_path)
                    if instance_stats:
                        solver_stats[instance_name] = instance_stats
                    mtime = os.path.getmtime(log_path)
                    if min_mtime is None or mtime < min_mtime:
                        min_mtime = mtime
                    if max_mtime is None or mtime > max_mtime:
                        max_mtime = mtime
        else:
            logger.warning(f"Solver directory missing: {solver_dir}")

        expected_benchmark_count = len(self.cnf_files) if self.cnf_files else 400
        if len(solving_times) < expected_benchmark_count:
            missing_count = expected_benchmark_count - len(solving_times)
            logger.warning(f"Missing results for {missing_count} instances out of {expected_benchmark_count}")

        if not solving_times:
            # do not overwrite an existing PAR2 with None when this iteration skipped evaluation (e.g., --skip-evaluated leaders) or logs are missing.
            if existing_code_result is not None and existing_code_result.par2 is not None:
                logger.warning(
                    f"No solving logs found for algorithm {algorithm_id}, code {code_id}; "
                    f"preserving existing PAR2={existing_code_result.par2:.2f}"
                )
                return existing_code_result.par2
            logger.warning(
                f"No solving logs found for algorithm {algorithm_id}, code {code_id}; "
                "skipping PAR2 update"
            )
            return None

        par2 = _compute_average(list(solving_times.values()))
        logger.info(f"Computed PAR2 for algorithm {algorithm_id}, code {code_id}: {par2}")

        if par2 is not None and par2 < MIN_PAR2_THRESHOLD:
            logger.warning(
                f"PAR2 {par2:.2f} below threshold {MIN_PAR2_THRESHOLD} for "
                f"algorithm {algorithm_id}, code {code_id} — replacing with penalty {self.par2_penalty}"
            )
            par2 = self.par2_penalty

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

        # Append wall-clock timing entry to solver_timing_log.json
        if min_mtime is not None and max_mtime is not None and self.generation_tag:
            timing_log_path = os.path.join(get_generation_output_dir(self.generation_tag), "solver_timing_log.json")
            timing_entries = []
            if os.path.exists(timing_log_path):
                try:
                    with open(timing_log_path) as f:
                        timing_entries = json.load(f)
                except Exception:
                    pass
            timing_entries.append({
                "alg_id": algorithm_id,
                "solver_time": max_mtime - min_mtime,
                "leader_flag": algorithm.parent_id is None if algorithm else True,
            })
            with open(timing_log_path, "w") as f:
                json.dump(timing_entries, f, indent=2)
            logger.info(f"Appended timing entry for {algorithm_id} to {timing_log_path}")

        if solver_stats:
            stats_path = result_path.replace("solving_times_", "solver_stats_")
            with open(stats_path, "w") as f:
                json.dump(solver_stats, f)
            logger.info(f"Wrote solver stats for {len(solver_stats)} instances to {stats_path}")

        if self.instance_categories:
            par2_breakdown = _compute_par2_breakdown(solving_times, self.instance_categories)
            breakdown_path = result_path.replace("solving_times_", "par2_breakdown_")
            with open(breakdown_path, "w") as f:
                json.dump(par2_breakdown, f)
            logger.info(f"Wrote PAR2 breakdown to {breakdown_path}: {par2_breakdown}")

            # Write raw_par2_score [easy, hard, sat, unsat, all] to AlgorithmResult
            if algorithm is not None:
                algorithm.raw_par2_score = [
                    par2_breakdown.get("easy"),
                    par2_breakdown.get("hard"),
                    par2_breakdown.get("sat"),
                    par2_breakdown.get("unsat"),
                    par2_breakdown.get("all"),
                ]

                # Compute normalized PAR2 if baseline is configured
                if BASELINE_PAR2 is not None:
                    algorithm.normalized_par2_score = normalize_par2(
                        algorithm.raw_par2_score, BASELINE_PAR2
                    )
                else:
                    logger.warning(
                        "baseline_par2 not set in path_config.yaml - skipping normalization"
                    )

                update_algorithm_result(algorithm)

                # Save AlgorithmResult to disk as JSON memory bank
                algo_dir = get_algorithm_dir(
                    algorithm_id, generation_tag=self.generation_tag,
                    role=algorithm.role,
                )
                algorithm.save_to_json(algo_dir)
                logger.info(f"Saved AlgorithmResult JSON to {algo_dir}")

        return par2

    def slurm_collect_result(self, slurm_ids: List[int], code_id: str) -> None:
        """Submit SLURM job to collect results after evaluation jobs complete."""
        activate_cmd = _get_activation_cmd()
        code_result = get_code_result(code_id)
        if code_result is None:
            logger.error(f"Code result not found for code_id={code_id}, skipping collect job")
            return
        algorithm_id = code_result.algorithm_id
        algorithm = get_algorithm_result(algorithm_id)
        algo_role = algorithm.role if algorithm else None
        result_dir = get_solver_result_dir(algorithm_id, code_id,
                                           generation_tag=self.generation_tag, role=algo_role)

        gen_tag_flag = f" --generation_tag {self.generation_tag}" if self.generation_tag else ""
        quick_eval_flag = " --quick-eval" if self.cnf_files is not None else ""
        cmd = f"{activate_cmd} && python src/llmsat/pipelines/evaluation.py --algorithm_id {algorithm_id} --code_id {code_id} --collect_result{gen_tag_flag}{quick_eval_flag}"
        output_file = f"{result_dir}/00000000_collect_result.log"

        slurm_cmd = wrap_command_to_slurm(
            cmd,
            output_file=output_file,
            job_name=f"collect_result_{code_id[:8]}",
            dependencies=[str(sid) for sid in slurm_ids],
            dependency_type="afterany",
            mem="1G",
            time="00:05:00",
        )

        try:
            slurm_output = os.popen(slurm_cmd).read().strip()
            if not slurm_output or "error" in slurm_output.lower():
                logger.error(f"Failed to submit collect result job for {code_id[:16]}...: {slurm_output}")
                return
            slurm_id = int(slurm_output.split()[-1])
            logger.info(f"Submitted collect result job {slurm_id}")
        except (ValueError, IndexError) as e:
            logger.error(f"Failed to parse collect result job ID for {code_id[:16]}...: {e}")
            return

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
                    ["./configure", "-c"],  # add "--reboot" for kissat_rebooting_direct
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
            nproc = os.cpu_count() or 1
            all_logs.append(f"\n--- make -j{nproc} ---")
            make_proc = subprocess.run(
                ["make", f"-j{nproc}"],
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

        target_function = algorithm.function_name
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
                                         generation_tag=self.generation_tag, role=algorithm.role)
        algorithm_dir = get_algorithm_dir(code_result.algorithm_id,
                                          generation_tag=self.generation_tag, role=algorithm.role)
        os.makedirs(algorithm_dir, exist_ok=True)

        logger.info(f"Building solver at {new_solver_path}")
        if os.path.exists(new_solver_path):
            shutil.rmtree(new_solver_path)
        shutil.copytree(BASE_SOLVER_PATH, new_solver_path, symlinks=True)

        # Remove stale build system files so ./configure can run cleanly.
        # The base solver may have a pre-existing build/makefile (with
        # hardcoded paths to the old location) and a src/makefile symlink
        # (possibly dangling) that cause configure to fail.
        # We keep .o files in build/ so make only recompiles changed files
        # (avoids hitting compile errors in unrelated source files).
        stale_build_makefile = os.path.join(new_solver_path, "build", "makefile")
        if os.path.exists(stale_build_makefile):
            os.remove(stale_build_makefile)
        stale_makefile = os.path.join(new_solver_path, "src", "makefile")
        if os.path.islink(stale_makefile) or os.path.exists(stale_makefile):
            os.remove(stale_makefile)

        # Ensure configure script is executable after copy (copytree may not preserve exec bit)
        configure_script = os.path.join(new_solver_path, "configure")
        if os.path.exists(configure_script):
            os.chmod(configure_script, os.stat(configure_script).st_mode | 0o111)

        # Compile with debugging support
        func_info = self.registry[target_function]
        modified_file = f"{new_solver_path}/{func_info.file}"

        try:
            build_success, final_code, all_logs = self._compile_with_debugging(
                solver_path=new_solver_path,
                current_code=new_code,
                target_function=target_function,
                max_debug_rounds=3,
                debug_model=DEFAULT_MODEL,
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
        timeout: int = None,
        wall_time: str = None,
        cnf_files: List[str] = None,
        solver_flags: str = EVALUATION_SOLVER_FLAGS,
    ) -> List[int]:
        """
        Submit solver evaluation using a SLURM job array.

        Args:
            solver_path: Path to the built solver directory
            benchmark_path: Path to the benchmark CNF files
            result_dir: Directory to store evaluation results
            max_jobs: Maximum number of CNF files to evaluate
            dry_run: If True, print SLURM commands without submitting
            timeout: Timeout per CNF in seconds (default: self.timeout)
            wall_time: SLURM wall time (default: self.wall_time)
            cnf_files: Explicit list of CNF filenames to evaluate (default: scan benchmark_path)

        Returns:
            List containing the SLURM job array ID, or empty list on failure.
        """
        if timeout is None:
            timeout = self.timeout
        if wall_time is None:
            wall_time = self.wall_time

        logger.info(f"Submitting SLURM job array for solver {solver_path}")
        os.makedirs(result_dir, exist_ok=True)

        # Collect CNF files to evaluate (skip already completed ones)
        source_cnf_files = cnf_files if cnf_files is not None else self.cnf_files
        all_cnf_files = []
        jobs_skipped = 0
        if source_cnf_files is not None:
            # Use explicit file list
            for benchmark_file in source_cnf_files:
                if os.path.exists(f"{result_dir}/{benchmark_file}.solving.log"):
                    jobs_skipped += 1
                    continue
                all_cnf_files.append(benchmark_file)
        else:
            # Scan benchmark directory
            for benchmark_file in sorted(os.listdir(benchmark_path)):
                if benchmark_file.endswith(".cnf"):
                    if os.path.exists(f"{result_dir}/{benchmark_file}.solving.log"):
                        jobs_skipped += 1
                        continue
                    all_cnf_files.append(benchmark_file)
        cnf_files = all_cnf_files

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
SOLVER="{solver_path}/kissat"
BENCHMARK_PATH="{benchmark_path}"
RESULT_DIR="{result_dir}"
TIMEOUT={timeout}
SOLVER_FLAGS="{solver_flags}"

# Locate GNU time (path varies across systems; e.g. CVMFS on Compute Canada)
GNU_TIME=$(which time 2>/dev/null)
if [ -z "$GNU_TIME" ]; then
    echo "ERROR: GNU time not found in PATH" >&2
    exit 1
fi

CNF_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CNF_LIST")
OUTPUT_FILE="${{RESULT_DIR}}/${{CNF_FILE}}.solving.log"
PROOF_DIR="${{RESULT_DIR}}/proofs"
PROOF_FILE="${{PROOF_DIR}}/${{CNF_FILE}}.proof"

if [ -z "$CNF_FILE" ]; then
    echo "ERROR: No CNF file found for array task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Skip if already done
if [ -f "$OUTPUT_FILE" ]; then
    echo "Already completed: $CNF_FILE"
    exit 0
fi

mkdir -p "$PROOF_DIR"

echo "Running solver on $CNF_FILE (array task $SLURM_ARRAY_TASK_ID)"

# Run solver with timeout, measuring CPU time (user + system) via GNU time
"$GNU_TIME" -f "%U %S" -o "${{OUTPUT_FILE}}.time" \
    timeout ${{TIMEOUT}}s "$SOLVER" $SOLVER_FLAGS "$BENCHMARK_PATH/$CNF_FILE" "$PROOF_FILE" > "$OUTPUT_FILE" 2>&1
EXIT_CODE=$?

if [ -f "${{OUTPUT_FILE}}.time" ]; then
    CPU_TIME=$(awk '{{printf "%.6f", $1 + $2}}' "${{OUTPUT_FILE}}.time")
    rm -f "${{OUTPUT_FILE}}.time"
else
    CPU_TIME=""
fi

if [ $EXIT_CODE -eq 124 ]; then
    echo "TIMEOUT after ${{TIMEOUT}}s" >> "$OUTPUT_FILE"
    rm -f "$PROOF_FILE"
elif [ -z "$CPU_TIME" ]; then
    echo "ERROR: process killed (OOM or SLURM limit)" >> "$OUTPUT_FILE"
    rm -f "$PROOF_FILE"
elif [ $EXIT_CODE -eq 10 ] || [ $EXIT_CODE -eq 20 ]; then
    echo "c process-time: $CPU_TIME seconds" >> "$OUTPUT_FILE"
    # Keep proof only for UNSAT (kissat exit code 20).
    if [ $EXIT_CODE -ne 20 ]; then
        rm -f "$PROOF_FILE"
    fi
else
    echo "ERROR: solver crashed (exit code $EXIT_CODE)" >> "$OUTPUT_FILE"
    rm -f "$PROOF_FILE"
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
            time=wall_time,
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
        timeout: int = None,
        wall_time: str = None,
    ) -> List[int]:
        """
        Submit ALL (solver × CNF) pairs as unified job arrays.
        More efficient than one array per solver.

        Args:
            solver_tasks: List of (solver_path, result_dir, code_id) tuples
            benchmark_path: Path to the benchmark CNF files
            cnf_files: List of CNF file names to evaluate
            dry_run: If True, print SLURM commands without submitting
            timeout: Timeout per CNF in seconds (default: self.timeout)
            wall_time: SLURM wall time (default: self.wall_time)

        Returns:
            List of SLURM job array IDs
        """
        if timeout is None:
            timeout = self.timeout
        if wall_time is None:
            wall_time = self.wall_time
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

            # Create a directory for this batch on shared filesystem
            # /tmp is node-local and not visible to compute nodes
            batch_dir = f"solvers/{self.generation_tag}/slurm_batches/batch_{chunk_num}"
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
SOLVER_FLAGS="{EVALUATION_SOLVER_FLAGS}"

# Locate GNU time (path varies across systems; e.g. CVMFS on Compute Canada)
GNU_TIME=$(which time 2>/dev/null)
if [ -z "$GNU_TIME" ]; then
    echo "ERROR: GNU time not found in PATH" >&2
    exit 1
fi

# Read task info for this array index
TASK_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_LIST")
SOLVER_PATH=$(echo "$TASK_LINE" | cut -f1)
RESULT_DIR=$(echo "$TASK_LINE" | cut -f2)
CNF_FILE=$(echo "$TASK_LINE" | cut -f3)

SOLVER="${{SOLVER_PATH}}/kissat"
OUTPUT_FILE="${{RESULT_DIR}}/${{CNF_FILE}}.solving.log"
PROOF_DIR="${{RESULT_DIR}}/proofs"
PROOF_FILE="${{PROOF_DIR}}/${{CNF_FILE}}.proof"

if [ -z "$CNF_FILE" ]; then
    echo "ERROR: No task found for array task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Skip if already done
if [ -f "$OUTPUT_FILE" ]; then
    echo "Already completed: $CNF_FILE"
    exit 0
fi

mkdir -p "$PROOF_DIR"

echo "Running solver on $CNF_FILE (array task $SLURM_ARRAY_TASK_ID)"

# Run solver with timeout, measuring CPU time (user + system) via GNU time
"$GNU_TIME" -f "%U %S" -o "${{OUTPUT_FILE}}.time" \
    timeout ${{TIMEOUT}}s "$SOLVER" $SOLVER_FLAGS "$BENCHMARK_PATH/$CNF_FILE" "$PROOF_FILE" > "$OUTPUT_FILE" 2>&1
EXIT_CODE=$?

if [ -f "${{OUTPUT_FILE}}.time" ]; then
    CPU_TIME=$(awk '{{printf "%.6f", $1 + $2}}' "${{OUTPUT_FILE}}.time")
    rm -f "${{OUTPUT_FILE}}.time"
else
    CPU_TIME=""
fi

if [ $EXIT_CODE -eq 124 ]; then
    echo "TIMEOUT after ${{TIMEOUT}}s" >> "$OUTPUT_FILE"
    rm -f "$PROOF_FILE"
elif [ -z "$CPU_TIME" ]; then
    echo "ERROR: process killed (OOM or SLURM limit)" >> "$OUTPUT_FILE"
    rm -f "$PROOF_FILE"
elif [ $EXIT_CODE -eq 10 ] || [ $EXIT_CODE -eq 20 ]; then
    echo "c process-time: $CPU_TIME seconds" >> "$OUTPUT_FILE"
    # Keep proof only for UNSAT (kissat exit code 20).
    if [ $EXIT_CODE -ne 20 ]; then
        rm -f "$PROOF_FILE"
    fi
else
    echo "ERROR: solver crashed (exit code $EXIT_CODE)" >> "$OUTPUT_FILE"
    rm -f "$PROOF_FILE"
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
                time=wall_time,
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

    def run_single_solver(self, code_id: str, build_only: bool = False, dry_run: bool = False, skip_build: bool = False,
                          sample_size: Optional[int] = None, sample_seed: int = 42) -> Optional[Tuple[str, str, str]]:
        """
        Build and evaluate a single code result.

        Args:
            code_id: The code result ID to build and evaluate
            build_only: If True, build the solver but don't submit SLURM evaluation
            dry_run: If True, print SLURM commands without submitting
            skip_build: If True, assume solver is already built and skip build step
            sample_size: If set, randomly sample this many CNF files for evaluation
            sample_seed: Random seed for reproducible CNF sampling (default: 42)

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
                                         generation_tag=self.generation_tag, role=algorithm.role)
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
                                               generation_tag=self.generation_tag, role=algorithm.role)

            if build_only:
                logger.info(f"Build-only mode: skipping SLURM evaluation")
                return (solver_path, result_dir, code_id)

            # Apply CNF sampling if requested
            eval_cnf_files = self.cnf_files
            if sample_size is not None:
                import random as _random
                # Build full CNF list to sample from
                if eval_cnf_files is None:
                    eval_cnf_files = sorted(f for f in os.listdir(SAT2025_BENCHMARK_PATH) if f.endswith(".cnf"))
                if sample_size < len(eval_cnf_files):
                    rng = _random.Random(sample_seed)
                    eval_cnf_files = sorted(rng.sample(eval_cnf_files, sample_size))
                    logger.info(f"Sampled {sample_size} CNF files (seed={sample_seed})")

            slurm_ids = self.slurm_run_evaluate(solver_path, SAT2025_BENCHMARK_PATH, result_dir, dry_run=dry_run,
                                                   timeout=self.timeout, wall_time=self.wall_time, cnf_files=eval_cnf_files)

            if not dry_run:
                if slurm_ids:
                    code_result.status = CodeStatus.Evaluating
                    update_code_result(code_result)
                    self.slurm_collect_result(slurm_ids, code_id)
                else:
                    logger.error(f"SLURM submission failed for {code_id[:16]}..., status not updated")

            return (solver_path, result_dir, code_id)
        else:
            if not dry_run:
                code_result.status = CodeStatus.BuildFailed
                update_code_result(code_result)
            logger.error("Solver build failed")
            return None

    def run_all_solvers(self, algorithm_id: str, build_only: bool = False, dry_run: bool = False, skip_build: bool = False, skip_evaluated: bool = False) -> None:
        """Build and evaluate all code results for an algorithm."""
        logger.info(f"Running evaluation for algorithm {algorithm_id}")
        algorithm = get_algorithm_result(algorithm_id)
        if algorithm is None:
            logger.error(f"Algorithm not found: {algorithm_id}")
            return

        algorithm_dir = get_algorithm_dir(algorithm_id,
                                          generation_tag=self.generation_tag, role=algorithm.role)
        os.makedirs(algorithm_dir, exist_ok=True)

        code_id_list = algorithm.code_id_list or []
        logger.info(f"Found {len(code_id_list)} code ids to evaluate for algorithm {algorithm_id}")

        for code_id in code_id_list:
            if skip_evaluated:
                code_result = get_code_result(code_id)
                if code_result and code_result.par2 is not None:
                    logger.info(f"Skipping already-evaluated code {code_id[:16]}... (PAR2={code_result.par2:.2f})")
                    continue
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
        skip_evaluated: bool = False,
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
            skip_evaluated: If True, skip code results that already have PAR2 scores
        """
        # Collect CNF files from benchmark
        if self.cnf_files is not None:
            cnf_files = self.cnf_files
            logger.info(f"Using {len(cnf_files)} CNF files from benchmark list")
        else:
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
                                              generation_tag=self.generation_tag, role=algorithm.role)
            os.makedirs(algorithm_dir, exist_ok=True)

            code_id_list = algorithm.code_id_list or []
            logger.info(f"Found {len(code_id_list)} code ids for algorithm {algorithm_id}")

            for code_id in code_id_list:
                code_result = get_code_result(code_id)
                if code_result is None:
                    logger.error(f"Code result not found for code_id={code_id}")
                    continue

                if skip_evaluated and code_result.par2 is not None:
                    logger.info(f"Skipping already-evaluated code {code_id[:16]}... (PAR2={code_result.par2:.2f})")
                    continue

                if skip_build:
                    # Check if solver already exists
                    solver_path = get_solver_dir(algorithm_id, code_id,
                                                 generation_tag=self.generation_tag, role=algorithm.role)
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
                                                   generation_tag=self.generation_tag, role=algorithm.role)
                solver_tasks.append((solver_path, result_dir, code_id))

        logger.info(f"Built {len(solver_tasks)} solvers successfully")

        if build_only:
            logger.info("Build-only mode: skipping SLURM evaluation")
            return

        if not solver_tasks:
            logger.warning("No solvers built successfully, nothing to evaluate")
            return

        # Submit all evaluations in batch
        job_ids = self.slurm_run_evaluate_batch(solver_tasks, SAT2025_BENCHMARK_PATH, cnf_files, dry_run=dry_run,
                                                timeout=self.timeout, wall_time=self.wall_time)

        if not dry_run and job_ids:
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
            for solver_path, result_dir, code_id in solver_tasks:
                self.slurm_collect_result(job_ids, code_id)
        elif not dry_run:
            logger.error("All SLURM batch submissions failed, statuses not updated")

        # Persist job IDs for SLURM polling scripts
        if job_ids and not dry_run:
            job_ids_path = os.path.join(
                get_generation_output_dir(self.generation_tag),
                "submitted_job_ids.json"
            )
            os.makedirs(os.path.dirname(job_ids_path), exist_ok=True)
            with open(job_ids_path, "w") as f:
                json.dump({"job_ids": job_ids, "timestamp": datetime.now().isoformat()}, f)
            logger.info(f"Saved job IDs to {job_ids_path}")

        logger.info(f"Batch submission complete. Job IDs: {job_ids}")

    def _load_generation_algorithms(self) -> Dict[str, AlgorithmResult]:
        """Load all algorithms registered under this generation tag."""
        if not self.generation_tag:
            logger.error("generation_tag is required")
            return {}

        algorithm_ids = get_ids_from_router_table(
            CHATGPT_DATA_GENERATION_TABLE, self.generation_tag
        )
        algorithms: Dict[str, AlgorithmResult] = {}
        for aid in algorithm_ids:
            algo = get_algorithm_result(aid)
            if algo is not None:
                algorithms[aid] = algo
        return algorithms

    @staticmethod
    def _primary_code_id(algo: AlgorithmResult) -> Optional[str]:
        return algo.code_id_list[0] if algo.code_id_list else None

    def _get_algorithm_par2(self, algo: AlgorithmResult) -> Optional[float]:
        code_id = self._primary_code_id(algo)
        if code_id is None:
            return None
        code_result = get_code_result(code_id)
        if code_result is None:
            return None
        return code_result.par2

    def _build_team_rows(self) -> List[Dict[str, object]]:
        """Assemble leader/member teams with cached PAR2 lookups."""
        algorithms = self._load_generation_algorithms()
        logger.info(
            f"Found {len(algorithms)} algorithms for generation '{self.generation_tag}'"
        )

        leaders: Dict[str, AlgorithmResult] = {}
        members_by_leader: Dict[str, List[AlgorithmResult]] = {}
        for algo in algorithms.values():
            if algo.role == Role.LEADER:
                leaders[algo.id] = algo
            else:
                leader_ref = (
                    algo.parent_id[0]
                    if isinstance(algo.parent_id, list) and algo.parent_id
                    else str(algo.parent_id)
                )
                members_by_leader.setdefault(leader_ref, []).append(algo)

        logger.info(
            f"Found {len(leaders)} leaders and "
            f"{sum(len(m) for m in members_by_leader.values())} members"
        )

        teams: List[Dict[str, object]] = []
        for leader_id, leader in leaders.items():
            leader_par2 = self._get_algorithm_par2(leader)
            team_members = members_by_leader.get(leader_id, [])
            member_info: List[Tuple[AlgorithmResult, Optional[float]]] = []

            best_member: Optional[AlgorithmResult] = None
            best_member_par2: Optional[float] = None

            for member in team_members:
                member_par2 = self._get_algorithm_par2(member)
                member_info.append((member, member_par2))
                if leader_par2 is None or member_par2 is None:
                    continue
                if member_par2 < leader_par2 and (
                    best_member_par2 is None or member_par2 < best_member_par2
                ):
                    best_member = member
                    best_member_par2 = member_par2

            teams.append(
                {
                    "leader": leader,
                    "leader_par2": leader_par2,
                    "members": team_members,
                    "member_info": member_info,
                    "best_member": best_member,
                    "best_member_par2": best_member_par2,
                }
            )

        def _team_best_par2(team: Dict[str, object]) -> float:
            scores: List[float] = []
            leader_par2 = team["leader_par2"]
            if leader_par2 is not None:
                scores.append(leader_par2)
            for _, member_par2 in team["member_info"]:
                if member_par2 is not None:
                    scores.append(member_par2)
            return min(scores) if scores else float("inf")

        teams.sort(key=_team_best_par2)
        return teams

    def export_proof_candidates(self, output_path: Optional[str] = None) -> str:
        """
        Export promotable team-best members for proof verification.

        Only members that strictly beat their incumbent leader are exported.
        If a team's leader is still best, no pair is written for that team.
        """
        if not self.generation_tag:
            raise ValueError("export_proof_candidates requires a generation_tag")

        teams = self._build_team_rows()
        output_dir = get_generation_output_dir(self.generation_tag)
        if output_path is None:
            output_path = os.path.join(output_dir, "best_solver_pairs.json")

        pairs: List[Dict[str, object]] = []
        for team in teams:
            leader: AlgorithmResult = team["leader"]  # type: ignore[assignment]
            best_member: Optional[AlgorithmResult] = team["best_member"]  # type: ignore[assignment]
            best_member_par2: Optional[float] = team["best_member_par2"]  # type: ignore[assignment]
            leader_par2: Optional[float] = team["leader_par2"]  # type: ignore[assignment]
            if best_member is None or best_member_par2 is None:
                continue

            member_code_id = self._primary_code_id(best_member)
            if member_code_id is None:
                logger.warning(
                    f"Best member {best_member.id[:12]}... has no code_id, skipping proof candidate export"
                )
                continue

            pairs.append(
                {
                    "algorithm_id": best_member.id,
                    "code_id": member_code_id,
                    "role": "members",
                    "leader_algorithm_id": leader.id,
                    "leader_code_id": self._primary_code_id(leader),
                    "leader_par2": leader_par2,
                    "candidate_par2": best_member_par2,
                }
            )

        with open(output_path, "w") as f:
            json.dump(pairs, f, indent=2)

        logger.info(
            f"Wrote {len(pairs)} proof candidate pair(s) to {output_path}"
        )
        return output_path

    def _load_proof_validation_statuses(self) -> Optional[Dict[Tuple[str, str], bool]]:
        """
        Load per-solver proof validity from outputs/<tag>/proof_verification.json.

        This gate is intentionally about correctness, not completeness:
        ordinary solver timeouts in the evaluation logs are allowed, while
        invalid solver logs, missing proof results, and drat-trim failures are not.
        """
        if not self.generation_tag:
            return None

        report_path = os.path.join(
            get_generation_output_dir(self.generation_tag), "proof_verification.json"
        )
        if not os.path.exists(report_path):
            logger.warning(f"Proof verification report not found: {report_path}")
            return None

        try:
            with open(report_path, "r") as f:
                report = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read proof verification report {report_path}: {e}")
            return None

        status_by_solver: Dict[Tuple[str, str], bool] = {}
        for solver_name, cnf_statuses in report.items():
            if not isinstance(cnf_statuses, dict):
                continue
            if not solver_name.startswith("algorithm_") or "/code_" not in solver_name:
                continue
            algorithm_part, code_id = solver_name.split("/code_", 1)
            algorithm_id = algorithm_part.removeprefix("algorithm_")

            def _allows_promotion(status: object) -> bool:
                if status == "valid":
                    return True
                if not isinstance(status, str):
                    return False
                # Solver run timeouts are acceptable for the validity gate; they
                # reflect performance, not incorrect answers or bad proofs.
                return status.startswith("timeout: Timeout in solving log")

            status_by_solver[(algorithm_id, code_id)] = bool(cnf_statuses) and all(
                _allows_promotion(status) for status in cnf_statuses.values()
            )

        return status_by_solver

    def promote_leaders(
        self,
        dry_run: bool = False,
        require_valid_proof: bool = False,
    ) -> None:
        """
        Promote the best-performing algorithm in each team to be the new leader.

        For each team (leader + members sharing that leader's parent_id),
        compare PAR2 scores and swap if a member strictly beats the leader.

        Args:
            dry_run: If True, preview promotions without making changes.
        """
        if not self.generation_tag:
            logger.error("promote_leaders requires a generation_tag")
            return

        teams = self._build_team_rows()
        proof_status_by_solver = (
            self._load_proof_validation_statuses() if require_valid_proof else None
        )
        if require_valid_proof and proof_status_by_solver is None:
            logger.warning(
                "Proof-gated promotion requested but no proof verification report is available; "
                "promotion will fail closed"
            )

        # 4. Write PAR2 summary report
        output_dir = get_generation_output_dir(self.generation_tag)
        report_path = os.path.join(output_dir, "par2_scores.txt")

        def _trunc(s: str, n: int = 12) -> str:
            return s[:n] if len(s) > n else s

        promotion_count = 0
        blocked_promotions = 0
        promotions = []

        report_lines = [
            f"PAR2 Scores - {self.generation_tag}",
            "=" * 40,
            "",
        ]

        for team in teams:
            leader: AlgorithmResult = team["leader"]  # type: ignore[assignment]
            leader_par2: Optional[float] = team["leader_par2"]  # type: ignore[assignment]
            member_info: List[Tuple[AlgorithmResult, Optional[float]]] = team["member_info"]  # type: ignore[assignment]
            best_member: Optional[AlgorithmResult] = team["best_member"]  # type: ignore[assignment]
            best_member_par2: Optional[float] = team["best_member_par2"]  # type: ignore[assignment]

            # Find best in team
            all_scores = []
            if leader_par2 is not None:
                all_scores.append(("leader", leader_par2))
            for member, mp in member_info:
                if mp is not None:
                    all_scores.append((member.id, mp))

            best_id = None
            best_par2 = float('inf')
            for tag, score in all_scores:
                if score < best_par2:
                    best_par2 = score
                    best_id = tag

            leader_code_id = leader.code_id_list[0] if leader.code_id_list else "N/A"
            best_marker = "  <-- BEST" if best_id == "leader" else ""
            par2_str = f"{leader_par2:.2f}" if leader_par2 is not None else "N/A"
            report_lines.append(f"Team: {_trunc(leader.id)} (Leader)")
            report_lines.append(f"  [L] {_trunc(leader.id)} {_trunc(leader_code_id)} PAR2: {par2_str}{best_marker}")

            for member, mp in member_info:
                member_code_id = member.code_id_list[0] if member.code_id_list else "N/A"
                mp_str = f"{mp:.2f}" if mp is not None else "N/A"
                marker = "  <-- BEST" if best_id == member.id else ""
                report_lines.append(f"  [M] {_trunc(member.id)} {_trunc(member_code_id)} PAR2: {mp_str}{marker}")

            report_lines.append("")

            # Determine if promotion is needed
            if leader_par2 is None:
                logger.warning(f"Leader {leader.id} has no PAR2 score, skipping team")
                continue

            if best_member is not None and best_member_par2 is not None:
                if require_valid_proof:
                    member_code_id = self._primary_code_id(best_member)
                    is_valid = (
                        proof_status_by_solver.get((best_member.id, member_code_id))
                        if proof_status_by_solver is not None and member_code_id is not None
                        else None
                    )
                    if is_valid is not True:
                        blocked_promotions += 1
                        report_lines.append(
                            "  [proof] Best member failed or was missing proof validation; "
                            "keeping incumbent leader"
                        )
                        report_lines.append("")
                        logger.warning(
                            f"Skipping promotion of {best_member.id[:12]}... over "
                            f"{leader.id[:12]}... because proof validation did not pass"
                        )
                        continue

                promotion_count += 1
                promotions.append((leader, best_member, best_member_par2, leader_par2))

        summary_line = (
            f"Summary: {len(teams)} teams, {promotion_count} promotion"
            f"{'s' if promotion_count != 1 else ''}"
        )
        if require_valid_proof:
            summary_line += f", {blocked_promotions} blocked by proof gate"
        report_lines.append(summary_line)

        with open(report_path, "w") as f:
            f.write("\n".join(report_lines) + "\n")
        logger.info(f"Wrote PAR2 summary to {report_path}")

        # 5. Execute promotions
        for leader, best_member, best_member_par2, leader_par2 in promotions:
            logger.info(
                f"{'[DRY-RUN] ' if dry_run else ''}"
                f"Promoting {best_member.id[:12]}... (PAR2={best_member_par2:.2f}) "
                f"over leader {leader.id[:12]}... (PAR2={leader_par2:.2f})"
            )

            if dry_run:
                continue

            # DB: update promoted member to leader (preserve parent_id lineage)
            best_member.role = Role.LEADER
            update_algorithm_result(best_member)

            # DB: update demoted leader to member (preserve parent_id lineage)
            leader.role = Role.MEMBER
            update_algorithm_result(leader)

            # NOTE: We intentionally do NOT move/re-save algorithm JSON files
            # on disk. The leaders/ and members/ directories reflect the
            # *original* placement at generation time and stay stable across
            # promotions. The database is the source of truth for roles; the
            # filesystem is only a human-readable memory bank.

        if dry_run:
            logger.info(f"[DRY-RUN] {promotion_count} promotion(s) would be made")
        else:
            logger.info(
                f"Completed {promotion_count} promotion(s)"
                + (
                    f"; blocked {blocked_promotions} by proof gate"
                    if require_valid_proof
                    else ""
                )
            )


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
    parser.add_argument("--batch-mode", action="store_true",
                        help="Use batch submission mode (all solvers × CNFs in unified arrays)")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip build step, assume solvers are already built")
    parser.add_argument("--promote-leaders", action="store_true",
                        help="Promote best-performing member to leader in each team")
    parser.add_argument("--require-valid-proof", action="store_true",
                        help="Require proof verification to pass before promoting a member")
    parser.add_argument("--export-proof-candidates", action="store_true",
                        help="Write promotable team-best members to outputs/<tag>/best_solver_pairs.json")
    parser.add_argument("--quick-eval", action="store_true",
                        help="Fast evaluation: 100 representative CNFs, 1000s timeout")
    parser.add_argument("--skip-evaluated", action="store_true",
                        help="Skip algorithms whose code already has PAR2 scores (avoids re-evaluating leaders across iterations)")
    args = parser.parse_args()

    evaluation_pipeline = EvaluationPipeline(generation_tag=args.generation_tag)

    # Wire --timeout (fix: was defined but never used)
    if args.timeout != SLURM_TIMEOUT_SECONDS:
        evaluation_pipeline.timeout = args.timeout
        evaluation_pipeline.par2_penalty = args.timeout * 2
        logger.info(f"Custom timeout: {args.timeout}s, PAR2 penalty: {args.timeout * 2}")

    # Wire --quick-eval (overrides --timeout if both set)
    if args.quick_eval:
        evaluation_pipeline.timeout = QUICK_EVAL_TIMEOUT_SECONDS
        evaluation_pipeline.wall_time = QUICK_EVAL_WALL_TIME
        evaluation_pipeline.par2_penalty = QUICK_EVAL_PAR2_PENALTY
        if not os.path.exists(QUICK_EVAL_BENCHMARK_LIST):
            logger.error(f"Quick-eval benchmark list not found: {QUICK_EVAL_BENCHMARK_LIST}")
            logger.error("Generate it first: python scripts/generate_benchmark_subset.py")
            return
        with open(QUICK_EVAL_BENCHMARK_LIST) as f:
            evaluation_pipeline.cnf_files = [line.strip() for line in f if line.strip()]
        logger.info(f"Quick-eval mode: {len(evaluation_pipeline.cnf_files)} CNFs, "
                     f"{QUICK_EVAL_TIMEOUT_SECONDS}s timeout, "
                     f"{QUICK_EVAL_WALL_TIME} wall time")

    if args.export_proof_candidates:
        if not args.generation_tag:
            logger.error("--export-proof-candidates requires --generation_tag")
            return
        evaluation_pipeline.export_proof_candidates()
        return

    if args.promote_leaders:
        if not args.generation_tag:
            logger.error("--promote-leaders requires --generation_tag")
            return
        evaluation_pipeline.promote_leaders(
            dry_run=args.dry_run,
            require_valid_proof=args.require_valid_proof,
        )
        return

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
        evaluation_pipeline.run_all_solvers_batch(algorithms, build_only=args.build_only, dry_run=args.dry_run, skip_build=args.skip_build, skip_evaluated=args.skip_evaluated)
    else:
        # Use per-algorithm submission (original behavior)
        for algorithm in algorithms:
            logger.info(f"Processing algorithm: {algorithm.id}")
            evaluation_pipeline.run_all_solvers(algorithm.id, build_only=args.build_only, dry_run=args.dry_run, skip_build=args.skip_build, skip_evaluated=args.skip_evaluated)


if __name__ == "__main__":
    main()
