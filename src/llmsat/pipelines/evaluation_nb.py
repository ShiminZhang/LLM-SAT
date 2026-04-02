from __future__ import annotations

import argparse
import os
import subprocess
import time
from typing import List, Tuple

from llmsat.llmsat import CHATGPT_DATA_GENERATION_TABLE
from llmsat.utils.aws import get_algorithm_result, get_ids_from_router_table, get_code_result
from llmsat.utils.paths import get_solver_result_dir
from llmsat.utils.utils import wrap_command_to_slurm, wrap_command_to_slurm_array

from llmsat.pipelines.evaluation import (
    EvaluationPipeline as _EvaluationPipelineBase,
    QUICK_EVAL_BENCHMARK_LIST,
    QUICK_EVAL_PAR2_PENALTY,
    QUICK_EVAL_TIMEOUT_SECONDS,
    QUICK_EVAL_WALL_TIME,
    SLURM_ACCOUNT,
    SLURM_MAX_ARRAY_SIZE,
    SLURM_MAX_CONCURRENT,
    SLURM_MEMORY,
    SLURM_TIMEOUT_SECONDS,
    logger,
    setup_logging,
)

SLURM_SUBMIT_LIMIT = 1000
SLURM_SUBMIT_BUFFER = 50
SLURM_SUBMIT_POLL_INTERVAL = 30
SLURM_CHUNK_TASK_LIMIT = 100


class EvaluationPipelineNB(_EvaluationPipelineBase):
    def _current_slurm_task_count(self) -> int | None:
        user = os.environ.get("USER")
        if not user:
            logger.warning("USER is not set; cannot query current SLURM task count")
            return None

        proc = subprocess.run(
            ["squeue", "-r", "-u", user, "-h"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            logger.warning(f"Failed to query squeue for {user}: {proc.stderr.strip()}")
            return None

        return len([line for line in proc.stdout.splitlines() if line.strip()])

    def _wait_for_submit_capacity(self, needed_slots: int) -> None:
        effective_limit = max(1, SLURM_SUBMIT_LIMIT - SLURM_SUBMIT_BUFFER)
        if needed_slots > effective_limit:
            raise ValueError(
                f"Requested submission of {needed_slots} tasks exceeds effective limit {effective_limit}"
            )

        while True:
            current = self._current_slurm_task_count()
            if current is None:
                logger.info(
                    f"Unable to determine current SLURM task count; retrying in {SLURM_SUBMIT_POLL_INTERVAL}s"
                )
                time.sleep(SLURM_SUBMIT_POLL_INTERVAL)
                continue

            projected = current + needed_slots
            if projected <= effective_limit:
                logger.info(
                    f"SLURM submit capacity available: current={current}, needed={needed_slots}, "
                    f"effective_limit={effective_limit}"
                )
                return

            logger.info(
                f"Waiting for SLURM submit capacity: current={current}, needed={needed_slots}, "
                f"effective_limit={effective_limit}; sleeping {SLURM_SUBMIT_POLL_INTERVAL}s"
            )
            time.sleep(SLURM_SUBMIT_POLL_INTERVAL)

    def slurm_collect_result(self, slurm_ids: List[int], code_id: str) -> None:
        activate_cmd = self._get_activation_cmd_local()
        code_result = get_code_result(code_id)
        if code_result is None:
            logger.error(f"Code result not found for code_id={code_id}, skipping collect job")
            return
        algorithm_id = code_result.algorithm_id
        algorithm = get_algorithm_result(algorithm_id)
        parent_id = algorithm.parent_id if algorithm else None
        result_dir = get_solver_result_dir(
            algorithm_id,
            code_id,
            generation_tag=self.generation_tag,
            parent_id=parent_id,
        )

        gen_tag_flag = f" --generation_tag {self.generation_tag}" if self.generation_tag else ""
        quick_eval_flag = " --quick-eval" if self.cnf_files is not None else ""
        cmd = (
            f"{activate_cmd} && python src/llmsat/pipelines/evaluation_nb.py "
            f"--algorithm_id {algorithm_id} --code_id {code_id} --collect_result"
            f"{gen_tag_flag}{quick_eval_flag}"
        )
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

        self._wait_for_submit_capacity(1)
        slurm_output = os.popen(slurm_cmd).read().strip()
        if not slurm_output or "error" in slurm_output.lower():
            logger.error(f"Failed to submit collect result job for {code_id}: {slurm_output}")
            return
        slurm_id = int(slurm_output.split()[-1])
        logger.info(f"Submitted collect result job {slurm_id}")

    def slurm_run_evaluate_batch(
        self,
        solver_tasks: List[Tuple[str, str, str]],
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
        if not solver_tasks or not cnf_files:
            logger.warning("No solver tasks or CNF files to evaluate")
            return []

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

        logger.info(
            f"Total tasks to submit: {len(all_tasks)} "
            f"(from {len(solver_tasks)} solvers x {len(cnf_files)} CNFs)"
        )

        chunk_size = min(
            SLURM_MAX_ARRAY_SIZE,
            SLURM_SUBMIT_LIMIT - SLURM_SUBMIT_BUFFER,
            SLURM_CHUNK_TASK_LIMIT,
        )
        job_ids = []
        for chunk_start in range(0, len(all_tasks), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(all_tasks))
            chunk_tasks = all_tasks[chunk_start:chunk_end]
            chunk_num = chunk_start // chunk_size

            if len(chunk_tasks) > SLURM_CHUNK_TASK_LIMIT:
                raise ValueError(
                    f"chunk_tasks length {len(chunk_tasks)} exceeds limit {SLURM_CHUNK_TASK_LIMIT}"
                )

            batch_dir = f"solvers/{self.generation_tag}/slurm_batches/batch_{chunk_num}"
            os.makedirs(batch_dir, exist_ok=True)

            task_list_path = f"{batch_dir}/task_list.txt"
            with open(task_list_path, "w") as f:
                for solver_path, result_dir, code_id, cnf_file in chunk_tasks:
                    f.write(f"{solver_path}\t{result_dir}\t{cnf_file}\n")

            script_path = f"{batch_dir}/run_batch_array.sh"
            script_content = f"""#!/bin/bash
TASK_LIST="{task_list_path}"
BENCHMARK_PATH="{benchmark_path}"
TIMEOUT={timeout}
SOLVER_FLAGS="-s"

GNU_TIME=$(which time 2>/dev/null)
if [ -z "$GNU_TIME" ]; then
    echo "ERROR: GNU time not found in PATH" >&2
    exit 1
fi

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

if [ -f "$OUTPUT_FILE" ]; then
    echo "Already completed: $CNF_FILE"
    exit 0
fi

mkdir -p "$PROOF_DIR"

echo "Running solver on $CNF_FILE (array task $SLURM_ARRAY_TASK_ID)"

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
else
    echo "c process-time: $CPU_TIME seconds" >> "$OUTPUT_FILE"
    if [ $EXIT_CODE -ne 20 ]; then
        rm -f "$PROOF_FILE"
    fi
fi

echo "Solver finished with exit code $EXIT_CODE"
exit $EXIT_CODE
"""
            with open(script_path, "w") as f:
                f.write(script_content)
            os.chmod(script_path, 0o755)

            array_range = f"0-{len(chunk_tasks) - 1}"
            slurm_cmd = wrap_command_to_slurm_array(
                script_path=script_path,
                array_range=array_range,
                account=SLURM_ACCOUNT,
                mem=SLURM_MEMORY,
                time=wall_time,
                job_name=f"batch_{chunk_num}",
                output_file=f"{batch_dir}/slurm_array_%a.log",
                max_concurrent=min(SLURM_MAX_CONCURRENT, chunk_size),
            )
            logger.info(f"SLURM command for chunk {chunk_num}: {slurm_cmd}")

            if dry_run:
                logger.info(f"[DRY-RUN] Would submit batch {chunk_num} with {len(chunk_tasks)} tasks")
                print(f"[DRY-RUN] SLURM command:\n{slurm_cmd}")
                continue

            self._wait_for_submit_capacity(len(chunk_tasks))
            slurm_output = os.popen(slurm_cmd).read().strip()
            if not slurm_output or "error" in slurm_output.lower():
                logger.error(f"Failed to submit batch {chunk_num}: {slurm_output}")
                continue
            slurm_id = int(slurm_output.split()[-1])
            logger.info(f"Submitted batch {chunk_num} job array {slurm_id} with {len(chunk_tasks)} tasks")
            job_ids.append(slurm_id)

        return job_ids

    @staticmethod
    def _get_activation_cmd_local() -> str:
        from llmsat.config import PYTHON_ACTIVATE_PATH
        return f"source {PYTHON_ACTIVATE_PATH}"


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Evaluate SAT solver variants (nb throttled mode)")
    parser.add_argument("--algorithm_id", type=str, help="Single algorithm ID to evaluate")
    parser.add_argument("--code_id", type=str, help="Single code ID (used with --collect_result)")
    parser.add_argument("--first_n", type=int, help="Only evaluate first N algorithms")
    parser.add_argument("--run_all", action="store_true", help="Evaluate all algorithms in generation tag")
    parser.add_argument("--collect_result", action="store_true", help="Collect results for a single algorithm/code")
    parser.add_argument("--collect_all_results", action="store_true", help="Collect results for all algorithms in generation tag")
    parser.add_argument("--generation_tag", type=str, help="Generation tag to evaluate")
    parser.add_argument("--build-only", action="store_true", help="Build solvers but skip SLURM evaluation")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Print SLURM commands without submitting")
    parser.add_argument("--timeout", type=int, default=5000, help="Timeout per CNF in seconds (default: 5000)")
    parser.add_argument("--batch-mode", action="store_true", help="Use batch submission mode (all solvers x CNFs in unified arrays)")
    parser.add_argument("--skip-build", action="store_true", help="Skip build step, assume solvers are already built")
    parser.add_argument("--promote-leaders", action="store_true", help="Promote best-performing member to leader in each team")
    parser.add_argument("--quick-eval", action="store_true", help="Fast evaluation: 100 representative CNFs, 1000s timeout")
    parser.add_argument("--skip-evaluated", action="store_true", help="Skip algorithms whose code already has PAR2 scores")
    args = parser.parse_args()

    evaluation_pipeline = EvaluationPipelineNB(generation_tag=args.generation_tag)

    if args.timeout != SLURM_TIMEOUT_SECONDS:
        evaluation_pipeline.timeout = args.timeout
        evaluation_pipeline.par2_penalty = args.timeout * 2
        logger.info(f"Custom timeout: {args.timeout}s, PAR2 penalty: {args.timeout * 2}")

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
        logger.info(
            f"Quick-eval mode: {len(evaluation_pipeline.cnf_files)} CNFs, "
            f"{QUICK_EVAL_TIMEOUT_SECONDS}s timeout, {QUICK_EVAL_WALL_TIME} wall time"
        )

    if args.promote_leaders:
        if not args.generation_tag:
            logger.error("--promote-leaders requires --generation_tag")
            return
        evaluation_pipeline.promote_leaders(dry_run=args.dry_run)
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
        logger.info("Using batch submission mode")
        evaluation_pipeline.run_all_solvers_batch(
            algorithms,
            build_only=args.build_only,
            dry_run=args.dry_run,
            skip_build=args.skip_build,
            skip_evaluated=args.skip_evaluated,
        )
    else:
        for algorithm in algorithms:
            logger.info(f"Processing algorithm: {algorithm.id}")
            evaluation_pipeline.run_all_solvers(
                algorithm.id,
                build_only=args.build_only,
                dry_run=args.dry_run,
                skip_build=args.skip_build,
                skip_evaluated=args.skip_evaluated,
            )


if __name__ == "__main__":
    main()
