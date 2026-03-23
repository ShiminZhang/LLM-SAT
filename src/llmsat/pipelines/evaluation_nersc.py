"""NERSC-specific evaluation pipeline.

Packs SLURM_JOBS_PER_NODE=128 solver runs per node (one per physical core on
Perlmutter CPU nodes): 1 array task = 1 full node running 128 jobs in parallel.

Also patches evaluation.py module globals so that _submit_collect_result
(which calls the module-level wrap_command_to_slurm) uses NERSC SLURM settings.

Re-exports EvaluationPipeline as EvaluationPipelineNERSC so callers can do:
    from llmsat.pipelines.evaluation_nersc import EvaluationPipeline
and get the NERSC-aware subclass transparently.
"""
from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import llmsat.utils.utils_nersc as _nersc_utils
import llmsat.pipelines.evaluation as _eval_mod

# Patch module-level wrap_command_to_slurm used by _submit_collect_result.
_eval_mod.wrap_command_to_slurm = _nersc_utils.wrap_command_to_slurm
_eval_mod.SLURM_ACCOUNT = "m4831"

from llmsat.pipelines.evaluation import (  # noqa: E402 (after patch)
    EvaluationPipeline as _EvaluationPipelineBase,
    SLURM_MEMORY,
    SLURM_WALL_TIME,
    SLURM_TIMEOUT_SECONDS,
    SLURM_MAX_CONCURRENT,
    SLURM_MAX_ARRAY_SIZE,
    QUICK_EVAL_TIMEOUT_SECONDS,
    QUICK_EVAL_WALL_TIME,
    QUICK_EVAL_PAR2_PENALTY,
    QUICK_EVAL_BENCHMARK_LIST,
)

SLURM_ACCOUNT = "m4831"
SLURM_JOBS_PER_NODE = 128  # physical cores per Perlmutter CPU node (2 sockets × 64 cores)


class EvaluationPipelineNERSC(_EvaluationPipelineBase):
    """EvaluationPipeline adapted for NERSC Perlmutter.

    Each SLURM array task = 1 full node running SLURM_JOBS_PER_NODE=128 solver
    jobs in parallel (one per physical core). Uses NERSC account m4831 with
    --qos=regular and --constraint=cpu via utils_nersc.
    """

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
        solver_flags: str = "-s",
    ) -> List[int]:
        if timeout is None:
            timeout = self.timeout
        if wall_time is None:
            wall_time = self.wall_time

        os.makedirs(result_dir, exist_ok=True)

        # Collect CNF files to evaluate (skip already completed ones)
        source_cnf_files = cnf_files if cnf_files is not None else self.cnf_files
        all_cnf_files = []
        jobs_skipped = 0
        if source_cnf_files is not None:
            for benchmark_file in source_cnf_files:
                if os.path.exists(f"{result_dir}/{benchmark_file}.solving.log"):
                    jobs_skipped += 1
                    continue
                all_cnf_files.append(benchmark_file)
        else:
            for benchmark_file in sorted(os.listdir(benchmark_path)):
                if benchmark_file.endswith(".cnf"):
                    if os.path.exists(f"{result_dir}/{benchmark_file}.solving.log"):
                        jobs_skipped += 1
                        continue
                    all_cnf_files.append(benchmark_file)
        cnf_files = all_cnf_files

        if not cnf_files:
            from llmsat.pipelines.evaluation import logger
            logger.info(f"All {jobs_skipped} benchmarks already evaluated")
            return []

        if len(cnf_files) > max_jobs:
            from llmsat.pipelines.evaluation import logger
            logger.warning(f"Limiting evaluation to {max_jobs} benchmarks (out of {len(cnf_files)} remaining)")
            cnf_files = cnf_files[:max_jobs]

        # Write CNF file list
        cnf_list_path = f"{result_dir}/cnf_file_list.txt"
        with open(cnf_list_path, "w") as f:
            for cnf_file in cnf_files:
                f.write(f"{cnf_file}\n")

        # Create wrapper script: each array task = 1 node running SLURM_JOBS_PER_NODE jobs in parallel
        script_path = f"{result_dir}/run_solver_array.sh"
        script_content = f"""#!/bin/bash
CNF_LIST="{cnf_list_path}"
SOLVER="{solver_path}/kissat"
BENCHMARK_PATH="{benchmark_path}"
RESULT_DIR="{result_dir}"
TIMEOUT={timeout}
N_PER_NODE={SLURM_JOBS_PER_NODE}

# Compute the line range for this node's chunk
START_LINE=$(( SLURM_ARRAY_TASK_ID * N_PER_NODE + 1 ))
END_LINE=$(( START_LINE + N_PER_NODE - 1 ))

echo "Node $SLURM_ARRAY_TASK_ID: processing lines $START_LINE to $END_LINE"

# Run all CNF files in this chunk in parallel (one per physical core)
while IFS= read -r CNF_FILE; do
    (
        if [ -z "$CNF_FILE" ]; then
            exit 0
        fi

        OUTPUT_FILE="${{RESULT_DIR}}/${{CNF_FILE}}.solving.log"

        # Skip if already done
        if [ -f "$OUTPUT_FILE" ]; then
            echo "Already completed: $CNF_FILE"
            exit 0
        fi

        echo "Running solver on $CNF_FILE"

        # Run solver with timeout, capturing wall-clock time
        START_TIME=$(date +%s.%N)
        timeout ${{TIMEOUT}}s "$SOLVER" "$BENCHMARK_PATH/$CNF_FILE" > "$OUTPUT_FILE" 2>&1
        EXIT_CODE=$?
        END_TIME=$(date +%s.%N)
        ELAPSED=$(awk "BEGIN {{printf \\"%.6f\\", $END_TIME - $START_TIME}}")

        if [ $EXIT_CODE -eq 124 ]; then
            echo "TIMEOUT after ${{TIMEOUT}}s" >> "$OUTPUT_FILE"
        else
            echo "c process-time: $ELAPSED seconds" >> "$OUTPUT_FILE"
        fi
    ) &
done < <(sed -n "${{START_LINE}},${{END_LINE}}p" "$CNF_LIST")

wait  # wait for all background jobs on this node to finish
echo "Node $SLURM_ARRAY_TASK_ID done"
"""
        with open(script_path, "w") as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)

        # 1 array task = 1 node = SLURM_JOBS_PER_NODE parallel jobs
        n_nodes = math.ceil(len(cnf_files) / SLURM_JOBS_PER_NODE)
        array_range = f"0-{n_nodes - 1}"
        slurm_cmd = _nersc_utils.wrap_command_to_slurm_array(
            script_path=script_path,
            array_range=array_range,
            account=SLURM_ACCOUNT,
            mem=SLURM_MEMORY,
            time=wall_time,
            job_name="solve_array",
            output_file=f"{result_dir}/slurm_array_%a.log",
            max_concurrent=SLURM_MAX_CONCURRENT,
            ntasks=SLURM_JOBS_PER_NODE,
            cpus_per_task=1,
        )
        from llmsat.pipelines.evaluation import logger
        logger.info(f"Wrote {len(cnf_files)} CNF files to {cnf_list_path}")
        logger.info(f"SLURM command: {slurm_cmd}")

        if dry_run:
            logger.info(f"[DRY-RUN] Would submit job array with {n_nodes} nodes ({len(cnf_files)} CNFs, {jobs_skipped} skipped)")
            print(f"[DRY-RUN] SLURM command:\n{slurm_cmd}")
            return []

        try:
            slurm_output = os.popen(slurm_cmd).read().strip()
            if not slurm_output or "error" in slurm_output.lower():
                logger.error(f"Failed to submit job array: {slurm_output}")
                return []
            slurm_id = int(slurm_output.split()[-1])
            logger.info(f"Submitted job array {slurm_id} with {n_nodes} nodes ({len(cnf_files)} CNFs, {jobs_skipped} skipped)")
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
        if timeout is None:
            timeout = self.timeout
        if wall_time is None:
            wall_time = self.wall_time

        from llmsat.pipelines.evaluation import logger
        if not solver_tasks or not cnf_files:
            logger.warning("No solver tasks or CNF files to evaluate")
            return []

        # Build flat list of (solver_path, result_dir, code_id, cnf_file) tasks
        all_tasks = []
        for solver_path, result_dir, code_id in solver_tasks:
            os.makedirs(result_dir, exist_ok=True)
            for cnf_file in cnf_files:
                if os.path.exists(f"{result_dir}/{cnf_file}.solving.log"):
                    continue
                all_tasks.append((solver_path, result_dir, code_id, cnf_file))

        if not all_tasks:
            logger.info("All tasks already completed")
            return []

        logger.info(f"Total tasks to submit: {len(all_tasks)} (from {len(solver_tasks)} solvers × {len(cnf_files)} CNFs)")

        job_ids = []
        for chunk_start in range(0, len(all_tasks), SLURM_MAX_ARRAY_SIZE):
            chunk_end = min(chunk_start + SLURM_MAX_ARRAY_SIZE, len(all_tasks))
            chunk_tasks = all_tasks[chunk_start:chunk_end]
            chunk_num = chunk_start // SLURM_MAX_ARRAY_SIZE

            batch_dir = f"solvers/{self.generation_tag}/slurm_batches/batch_{chunk_num}"
            os.makedirs(batch_dir, exist_ok=True)

            task_list_path = f"{batch_dir}/task_list.txt"
            with open(task_list_path, "w") as f:
                for solver_path, result_dir, code_id, cnf_file in chunk_tasks:
                    f.write(f"{solver_path}\t{result_dir}\t{cnf_file}\n")

            # Each array task = 1 node running SLURM_JOBS_PER_NODE jobs in parallel
            script_path = f"{batch_dir}/run_batch_array.sh"
            script_content = f"""#!/bin/bash
TASK_LIST="{task_list_path}"
BENCHMARK_PATH="{benchmark_path}"
TIMEOUT={timeout}
N_PER_NODE={SLURM_JOBS_PER_NODE}

# Compute the line range for this node's chunk
START_LINE=$(( SLURM_ARRAY_TASK_ID * N_PER_NODE + 1 ))
END_LINE=$(( START_LINE + N_PER_NODE - 1 ))

echo "Node $SLURM_ARRAY_TASK_ID: processing lines $START_LINE to $END_LINE"

# Run all tasks in this chunk in parallel (one background process per core)
while IFS=$'\\t' read -r SOLVER_PATH RESULT_DIR CNF_FILE; do
    (
        if [ -z "$CNF_FILE" ]; then
            exit 0
        fi

        SOLVER="${{SOLVER_PATH}}/kissat"
        OUTPUT_FILE="${{RESULT_DIR}}/${{CNF_FILE}}.solving.log"

        # Skip if already done
        if [ -f "$OUTPUT_FILE" ]; then
            echo "Already completed: $CNF_FILE"
            exit 0
        fi

        echo "Running solver on $CNF_FILE"

        # Run solver with timeout, capturing wall-clock time
        START_TIME=$(date +%s.%N)
        timeout ${{TIMEOUT}}s "$SOLVER" "$BENCHMARK_PATH/$CNF_FILE" > "$OUTPUT_FILE" 2>&1
        EXIT_CODE=$?
        END_TIME=$(date +%s.%N)
        ELAPSED=$(awk "BEGIN {{printf \\"%.6f\\", $END_TIME - $START_TIME}}")

        if [ $EXIT_CODE -eq 124 ]; then
            echo "TIMEOUT after ${{TIMEOUT}}s" >> "$OUTPUT_FILE"
        else
            echo "c process-time: $ELAPSED seconds" >> "$OUTPUT_FILE"
        fi
    ) &
done < <(sed -n "${{START_LINE}},${{END_LINE}}p" "$TASK_LIST")

wait  # wait for all background jobs on this node to finish
echo "Node $SLURM_ARRAY_TASK_ID done"
"""
            with open(script_path, "w") as f:
                f.write(script_content)
            os.chmod(script_path, 0o755)

            # 1 array task = 1 node = SLURM_JOBS_PER_NODE parallel jobs
            n_nodes = math.ceil(len(chunk_tasks) / SLURM_JOBS_PER_NODE)
            array_range = f"0-{n_nodes - 1}"
            slurm_cmd = _nersc_utils.wrap_command_to_slurm_array(
                script_path=script_path,
                array_range=array_range,
                account=SLURM_ACCOUNT,
                mem=SLURM_MEMORY,
                time=wall_time,
                job_name=f"batch_{chunk_num}",
                output_file=f"{batch_dir}/slurm_array_%a.log",
                max_concurrent=SLURM_MAX_CONCURRENT,
                ntasks=SLURM_JOBS_PER_NODE,
                cpus_per_task=1,
            )
            logger.info(f"SLURM command for chunk {chunk_num}: {slurm_cmd}")

            if dry_run:
                logger.info(f"[DRY-RUN] Would submit batch {chunk_num} with {n_nodes} nodes ({len(chunk_tasks)} tasks)")
                print(f"[DRY-RUN] SLURM command:\n{slurm_cmd}")
                continue

            try:
                slurm_output = os.popen(slurm_cmd).read().strip()
                if not slurm_output or "error" in slurm_output.lower():
                    logger.error(f"Failed to submit batch {chunk_num}: {slurm_output}")
                    continue
                slurm_id = int(slurm_output.split()[-1])
                logger.info(f"Submitted batch {chunk_num} job array {slurm_id} with {n_nodes} nodes ({len(chunk_tasks)} tasks)")
                job_ids.append(slurm_id)
            except (ValueError, IndexError) as e:
                logger.error(f"Failed to parse SLURM job ID for batch {chunk_num}: {e}")

        return job_ids


# Export EvaluationPipelineNERSC as EvaluationPipeline for transparent use:
#   from llmsat.pipelines.evaluation_nersc import EvaluationPipeline
EvaluationPipeline = EvaluationPipelineNERSC

# Ensure CLI in evaluation.py uses the NERSC-aware pipeline when this module
# is executed directly.
_eval_mod.EvaluationPipeline = EvaluationPipelineNERSC


if __name__ == "__main__":
    _eval_mod.main()
