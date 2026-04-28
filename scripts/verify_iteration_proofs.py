#!/usr/bin/env python3
"""Verify UNSAT proofs for successful solvers in one generation tag using drat-trim."""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

# Ensure local package imports work when invoked as `python scripts/...`.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llmsat.llmsat import CHATGPT_DATA_GENERATION_TABLE, SAT2025_BENCHMARK_PATH, Role
from llmsat.utils.aws import (
    get_algorithm_result,
    get_code_result,
    get_ids_from_router_table,
)
from llmsat.utils.paths import (
    get_generation_output_dir,
    get_solver_proof_path,
    get_solver_result_dir,
)
from llmsat.utils.utils import wrap_command_to_slurm, wrap_command_to_slurm_array
from llmsat.utils.utils_nersc import (
    wrap_command_to_slurm as wrap_command_to_slurm_nersc,
    wrap_command_to_slurm_array as wrap_command_to_slurm_array_nersc,
)

INSTANCE_CATEGORIES_PATH = REPO_ROOT / "data" / "benchmarks" / "instance_categories.json"
SAT_CHECKFAIL_MARKERS = (
    "kissat: fatal error:",
    "unsatisfied clause:",
)
SIGNAL_MARKERS = (
    "raising signal ",
    "raise signal ",
    "raised signal ",
    "caught signal ",
)


@dataclass
class ProofCheckRecord:
    algorithm_id: str
    code_id: str
    cnf_file: str
    cnf_path: str
    proof_path: str
    status: str
    returncode: Optional[int] = None
    message: str = ""


def _load_successful_pairs(
    generation_tag: str,
) -> List[tuple[str, str, Optional[List[str]], Optional[Role]]]:
    pairs: List[tuple[str, str, Optional[List[str]], Optional[Role]]] = []
    best_pairs_path = REPO_ROOT / "outputs" / generation_tag / "best_solver_pairs.json"

    if best_pairs_path.exists():
        try:
            with open(best_pairs_path, "r") as f:
                best_pairs = json.load(f)
        except Exception:
            best_pairs = []

        for pair in best_pairs:
            role_name = str(pair.get("role", "")).lower()
            role = Role.LEADER if role_name == "leaders" else Role.MEMBER if role_name == "members" else None
            pairs.append(
                (
                    pair["algorithm_id"],
                    pair["code_id"],
                    None,
                    role,
                )
            )

        # Respect the explicit pair selection file even when it is empty. This
        # allows callers to say "verify no solvers for this tag" without
        # falling back to every successful build in the generation.
        return pairs

    algorithm_ids = get_ids_from_router_table(CHATGPT_DATA_GENERATION_TABLE, generation_tag)

    for algorithm_id in algorithm_ids:
        algo = get_algorithm_result(algorithm_id)
        if algo is None:
            continue
        for code_id in algo.code_id_list or []:
            code_result = get_code_result(code_id)
            if code_result is None:
                continue
            if code_result.build_success is True:
                pairs.append((algorithm_id, code_id, algo.parent_id, algo.role))
    return pairs


def _is_unsat_log(log_path: str) -> bool:
    try:
        with open(log_path, "r", errors="ignore") as f:
            content = f.read()
        return "s UNSATISFIABLE" in content
    except Exception:
        return False


def _read_log_content(log_path: str) -> str:
    try:
        with open(log_path, "r", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def _is_sat_log(log_path: str) -> bool:
    try:
        with open(log_path, "r", errors="ignore") as f:
            content = f.read()
        return "s SATISFIABLE" in content
    except Exception:
        return False


def _is_timeout_log(log_path: str) -> bool:
    try:
        with open(log_path, "r", errors="ignore") as f:
            content = f.read()
        upper_content = content.upper()
        return (
            "TIMEOUT" in upper_content
            or "CANCELLED" in upper_content
            or "TIME LIMIT" in upper_content
        )
    except Exception:
        return False


def _load_instance_categories() -> dict[str, str]:
    if not INSTANCE_CATEGORIES_PATH.exists():
        return {}
    try:
        with open(INSTANCE_CATEGORIES_PATH, "r") as f:
            data = json.load(f)
    except Exception:
        return {}
    return {
        instance_name: str(meta.get("satisfiability", "UNKNOWN")).upper()
        for instance_name, meta in data.items()
    }


def _instance_name_from_cnf_file(cnf_file: str) -> str:
    if cnf_file.endswith(".normalised.cnf"):
        return cnf_file[: -len(".normalised.cnf")]
    if cnf_file.endswith(".cnf"):
        return cnf_file[: -len(".cnf")]
    return cnf_file


def _is_expected_sat_instance(cnf_file: str, instance_categories: dict[str, str]) -> bool:
    return instance_categories.get(_instance_name_from_cnf_file(cnf_file)) == "SAT"


def _has_sat_checkfail(log_content: str) -> bool:
    lowered = log_content.lower()
    return any(marker in lowered for marker in SAT_CHECKFAIL_MARKERS)


def _has_signal_raise(log_content: str) -> bool:
    lowered = log_content.lower()
    return any(marker in lowered for marker in SIGNAL_MARKERS)


@dataclass
class ProofTask:
    algorithm_id: str
    code_id: str
    cnf_file: str
    cnf_path: str
    proof_path: str


def gather_proof_tasks(
    generation_tag: str, benchmark_path: str
) -> tuple[
    List[ProofTask],
    List[ProofCheckRecord],
    List[ProofCheckRecord],
    List[ProofCheckRecord],
]:
    tasks: List[ProofTask] = []
    valid_records: List[ProofCheckRecord] = []
    invalid_records: List[ProofCheckRecord] = []
    timeout_records: List[ProofCheckRecord] = []
    instance_categories = _load_instance_categories()
    pairs = _load_successful_pairs(generation_tag)

    for algorithm_id, code_id, parent_id, role in pairs:
        result_dir = get_solver_result_dir(
            algorithm_id,
            code_id,
            generation_tag=generation_tag,
            parent_id=parent_id,
            role=role,
        )

        for log_path in sorted(glob.glob(os.path.join(result_dir, "*.solving.log"))):
            cnf_file = os.path.basename(log_path)[: -len(".solving.log")]
            cnf_path = os.path.join(benchmark_path, cnf_file)
            proof_path = get_solver_proof_path(
                algorithm_id,
                code_id,
                cnf_file,
                generation_tag=generation_tag,
                parent_id=parent_id,
                role=role,
                create_dir=False,
            )
            log_content = _read_log_content(log_path)
            expected_sat = _is_expected_sat_instance(cnf_file, instance_categories)
            sat_check_failed = _has_sat_checkfail(log_content)
            signal_raised = _has_signal_raise(log_content)
            is_timeout = _is_timeout_log(log_path)

            # Check timeouts BEFORE signal check: Kissat prints
            # "raising signal 15 (SIGTERM)" on timeout, which would
            # otherwise be misclassified as a crash.
            if is_timeout:
                timeout_records.append(
                    ProofCheckRecord(
                        algorithm_id=algorithm_id,
                        code_id=code_id,
                        cnf_file=cnf_file,
                        cnf_path=cnf_path,
                        proof_path=proof_path,
                        status="timeout",
                        message="Timeout in solving log",
                    )
                )
                continue

            if signal_raised:
                invalid_records.append(
                    ProofCheckRecord(
                        algorithm_id=algorithm_id,
                        code_id=code_id,
                        cnf_file=cnf_file,
                        cnf_path=cnf_path,
                        proof_path=proof_path,
                        status="invalid",
                        message="Signal raised in solving log",
                    )
                )
                continue

            if expected_sat or _is_sat_log(log_path):
                if sat_check_failed:
                    invalid_records.append(
                        ProofCheckRecord(
                            algorithm_id=algorithm_id,
                            code_id=code_id,
                            cnf_file=cnf_file,
                            cnf_path=cnf_path,
                            proof_path=proof_path,
                            status="invalid",
                            message="SAT check failure in solving log",
                        )
                    )
                    continue

                valid_records.append(
                    ProofCheckRecord(
                        algorithm_id=algorithm_id,
                        code_id=code_id,
                        cnf_file=cnf_file,
                        cnf_path=cnf_path,
                        proof_path=proof_path,
                        status="valid",
                        message="SAT log without check failure",
                    )
                )
                continue

            if _is_unsat_log(log_path):
                tasks.append(
                    ProofTask(
                        algorithm_id=algorithm_id,
                        code_id=code_id,
                        cnf_file=cnf_file,
                        cnf_path=cnf_path,
                        proof_path=proof_path,
                    )
                )
                continue

            if _is_timeout_log(log_path):
                timeout_records.append(
                    ProofCheckRecord(
                        algorithm_id=algorithm_id,
                        code_id=code_id,
                        cnf_file=cnf_file,
                        cnf_path=cnf_path,
                        proof_path=proof_path,
                        status="timeout",
                        message="Timeout in solving log",
                    )
                )
                continue

            if sat_check_failed:
                invalid_records.append(
                    ProofCheckRecord(
                        algorithm_id=algorithm_id,
                        code_id=code_id,
                        cnf_file=cnf_file,
                        cnf_path=cnf_path,
                        proof_path=proof_path,
                        status="invalid",
                        message="SAT check failure in solving log",
                    )
                )
                continue

            invalid_records.append(
                ProofCheckRecord(
                    algorithm_id=algorithm_id,
                    code_id=code_id,
                    cnf_file=cnf_file,
                    cnf_path=cnf_path,
                    proof_path=proof_path,
                    status="invalid",
                    message="Missing SAT/UNSAT status in solving log",
                )
            )
    return tasks, valid_records, invalid_records, timeout_records


def run_proof_task(
    task: ProofTask,
    drat_trim_cmd: str,
    check_timeout_sec: int,
) -> ProofCheckRecord:
    if not os.path.exists(task.cnf_path):
        return ProofCheckRecord(
            algorithm_id=task.algorithm_id,
            code_id=task.code_id,
            cnf_file=task.cnf_file,
            cnf_path=task.cnf_path,
            proof_path=task.proof_path,
            status="failed",
            message="CNF file not found",
        )

    if not os.path.exists(task.proof_path):
        return ProofCheckRecord(
            algorithm_id=task.algorithm_id,
            code_id=task.code_id,
            cnf_file=task.cnf_file,
            cnf_path=task.cnf_path,
            proof_path=task.proof_path,
            status="failed",
            message="Proof file not found",
        )

    try:
        proc = subprocess.run(
            [drat_trim_cmd, task.cnf_path, task.proof_path],
            capture_output=True,
            text=True,
            timeout=check_timeout_sec,
        )
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        ok = proc.returncode == 0 and "VERIFIED" in out.upper()
        if ok:
            return ProofCheckRecord(
                algorithm_id=task.algorithm_id,
                code_id=task.code_id,
                cnf_file=task.cnf_file,
                cnf_path=task.cnf_path,
                proof_path=task.proof_path,
                status="valid",
                returncode=proc.returncode,
                message="drat-trim verified",
            )

        tail = out.strip()[-800:]
        return ProofCheckRecord(
            algorithm_id=task.algorithm_id,
            code_id=task.code_id,
            cnf_file=task.cnf_file,
            cnf_path=task.cnf_path,
            proof_path=task.proof_path,
            status="invalid",
            returncode=proc.returncode,
            message=tail or "drat-trim failed without output",
        )
    except subprocess.TimeoutExpired:
        return ProofCheckRecord(
            algorithm_id=task.algorithm_id,
            code_id=task.code_id,
            cnf_file=task.cnf_file,
            cnf_path=task.cnf_path,
            proof_path=task.proof_path,
            status="timeout",
            message=f"drat-trim timeout ({check_timeout_sec}s)",
        )


def verify_generation_proofs(
    generation_tag: str,
    benchmark_path: str,
    drat_trim_cmd: str,
    check_timeout_sec: int,
    stop_on_first_fail: bool,
) -> tuple[int, int, int, List[ProofCheckRecord]]:
    all_unsat_tasks, valid_records, invalid_records, timeout_records = gather_proof_tasks(
        generation_tag, benchmark_path
    )
    records: List[ProofCheckRecord] = (
        list(valid_records) + list(invalid_records) + list(timeout_records)
    )

    checked = len(valid_records) + len(invalid_records) + len(timeout_records)
    failed = len(invalid_records) + len(timeout_records)
    skipped = 0

    for task in all_unsat_tasks:
        record = run_proof_task(task, drat_trim_cmd, check_timeout_sec)
        checked += 1
        if record.status != "valid":
            failed += 1
        records.append(record)
        if stop_on_first_fail and record.status != "valid":
            return checked, failed, skipped, records

    return checked, failed, skipped, records


def _solver_name(algorithm_id: str, code_id: str) -> str:
    return f"algorithm_{algorithm_id}/code_{code_id}"


def _validation_result(record: ProofCheckRecord) -> str:
    if record.status == "valid":
        return "valid"
    if record.message:
        return f"{record.status}: {record.message}"
    return record.status


def build_validation_report(records: List[ProofCheckRecord]) -> dict[str, dict[str, str]]:
    report: dict[str, dict[str, str]] = {}
    for record in records:
        solver_name = _solver_name(record.algorithm_id, record.code_id)
        if solver_name not in report:
            report[solver_name] = {}
        report[solver_name][record.cnf_file] = _validation_result(record)
    return report


def _record_to_dict(record: ProofCheckRecord) -> dict:
    return {
        "algorithm_id": record.algorithm_id,
        "code_id": record.code_id,
        "cnf_file": record.cnf_file,
        "cnf_path": record.cnf_path,
        "proof_path": record.proof_path,
        "status": record.status,
        "returncode": record.returncode,
        "message": record.message,
    }


def _record_from_dict(data: dict) -> ProofCheckRecord:
    return ProofCheckRecord(
        algorithm_id=data["algorithm_id"],
        code_id=data["code_id"],
        cnf_file=data["cnf_file"],
        cnf_path=data["cnf_path"],
        proof_path=data["proof_path"],
        status=data["status"],
        returncode=data.get("returncode"),
        message=data.get("message", ""),
    )


def _task_to_dict(task: ProofTask) -> dict:
    return {
        "algorithm_id": task.algorithm_id,
        "code_id": task.code_id,
        "cnf_file": task.cnf_file,
        "cnf_path": task.cnf_path,
        "proof_path": task.proof_path,
    }


def _task_from_dict(data: dict) -> ProofTask:
    return ProofTask(
        algorithm_id=data["algorithm_id"],
        code_id=data["code_id"],
        cnf_file=data["cnf_file"],
        cnf_path=data["cnf_path"],
        proof_path=data["proof_path"],
    )


def _task_result_path(task_output_dir: str, task: ProofTask) -> str:
    return os.path.join(
        task_output_dir,
        f"{task.algorithm_id}__{task.code_id}__{task.cnf_file}.json",
    )


def _validation_log_root(generation_tag: str) -> Path:
    return REPO_ROOT / "logs" / generation_tag


def submit_proof_verification_slurm(
    generation_tag: str,
    benchmark_path: str,
    drat_trim_cmd: str,
    check_timeout_sec: int,
    slurm_account: str,
    slurm_mem: str,
    slurm_time: str,
    max_concurrent: int,
    nersc: bool = False,
    slurm_constraint: Optional[str] = None,
    slurm_qos: Optional[str] = None,
) -> int:
    out_dir = get_generation_output_dir(generation_tag)
    tasks, valid_records, invalid_records, timeout_records = gather_proof_tasks(
        generation_tag, benchmark_path
    )
    report_path = os.path.join(out_dir, "proof_verification.json")
    metadata_path = os.path.join(out_dir, "proof_verification_job.json")
    valid_records_path = os.path.join(out_dir, "proof_verification_valid.json")
    invalid_records_path = os.path.join(out_dir, "proof_verification_invalid.json")
    timeout_records_path = os.path.join(out_dir, "proof_verification_timeout.json")

    with open(valid_records_path, "w") as f:
        json.dump([_record_to_dict(record) for record in valid_records], f, indent=2)
    with open(invalid_records_path, "w") as f:
        json.dump([_record_to_dict(record) for record in invalid_records], f, indent=2)
    with open(timeout_records_path, "w") as f:
        json.dump([_record_to_dict(record) for record in timeout_records], f, indent=2)

    if not tasks:
        with open(report_path, "w") as f:
            json.dump(
                build_validation_report(valid_records + invalid_records + timeout_records),
                f,
                indent=2,
            )
        with open(metadata_path, "w") as f:
            json.dump(
                {
                    "generation_tag": generation_tag,
                    "status": "no_tasks",
                    "task_count": 0,
                    "valid_count": len(valid_records),
                    "invalid_count": len(invalid_records),
                    "timeout_count": len(timeout_records),
                    "valid_records_path": valid_records_path,
                    "invalid_records_path": invalid_records_path,
                    "timeout_records_path": timeout_records_path,
                    "report_path": report_path,
                },
                f,
                indent=2,
            )
        print(f"No UNSAT proof tasks found for {generation_tag}")
        print(f"Recorded {len(valid_records)} valid SAT log(s)")
        print(f"Recorded {len(invalid_records)} invalid log(s)")
        print(f"Recorded {len(timeout_records)} timeout log(s)")
        print(f"Report written to: {report_path}")
        return 0

    tasks_path = os.path.join(out_dir, "proof_verification_tasks.json")
    parts_dir = os.path.join(out_dir, "proof_verification_parts")
    os.makedirs(parts_dir, exist_ok=True)
    pending_tasks: List[ProofTask] = []
    reused_task_records: List[ProofCheckRecord] = []
    for task in tasks:
        result_path = _task_result_path(parts_dir, task)
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                reused_task_records.append(_record_from_dict(json.load(f)))
        else:
            pending_tasks.append(task)

    records = valid_records + invalid_records + timeout_records + reused_task_records
    with open(valid_records_path, "w") as f:
        json.dump(
            [_record_to_dict(record) for record in records if record.status == "valid"],
            f,
            indent=2,
        )
    with open(invalid_records_path, "w") as f:
        json.dump(
            [_record_to_dict(record) for record in records if record.status == "invalid"],
            f,
            indent=2,
        )
    with open(timeout_records_path, "w") as f:
        json.dump(
            [_record_to_dict(record) for record in records if record.status == "timeout"],
            f,
            indent=2,
        )

    if not pending_tasks:
        with open(report_path, "w") as f:
            json.dump(build_validation_report(records), f, indent=2)
        with open(metadata_path, "w") as f:
            json.dump(
                {
                    "generation_tag": generation_tag,
                    "status": "all_tasks_already_collected",
                    "task_count": 0,
                    "reused_task_count": len(reused_task_records),
                    "valid_count": sum(record.status == "valid" for record in records),
                    "invalid_count": sum(record.status == "invalid" for record in records),
                    "timeout_count": sum(record.status == "timeout" for record in records),
                    "valid_records_path": valid_records_path,
                    "invalid_records_path": invalid_records_path,
                    "timeout_records_path": timeout_records_path,
                    "report_path": report_path,
                },
                f,
                indent=2,
            )
        print(f"No pending UNSAT proof tasks to submit for {generation_tag}")
        print(f"Reused {len(reused_task_records)} existing UNSAT verification result(s)")
        print(f"Report written to: {report_path}")
        return 0

    with open(tasks_path, "w") as f:
        json.dump([_task_to_dict(task) for task in pending_tasks], f, indent=2)

    logs_dir = _validation_log_root(generation_tag)
    logs_dir.mkdir(parents=True, exist_ok=True)
    python_bin = sys.executable
    array_script_path = os.path.join(out_dir, "run_proof_verification_array.sh")
    collect_log = str(logs_dir / f"proof_collect_{generation_tag}.log")
    array_log = "/dev/null"
    task_log_root = str(logs_dir)

    array_script = f"""#!/bin/bash
set -euo pipefail
TASKS_PATH="{tasks_path}"
PARTS_DIR="{parts_dir}"
TASK_JSON=$("{python_bin}" - <<'PY'
import json
import os
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
with open(r"{tasks_path}", "r") as f:
    tasks = json.load(f)
print(json.dumps(tasks[idx]))
PY
)
export TASK_JSON
ALGORITHM_ID=$("{python_bin}" - <<'PY'
import json
import os
print(json.loads(os.environ["TASK_JSON"])["algorithm_id"])
PY
)
CODE_ID=$("{python_bin}" - <<'PY'
import json
import os
print(json.loads(os.environ["TASK_JSON"])["code_id"])
PY
)
CNF_FILE=$("{python_bin}" - <<'PY'
import json
import os
from pathlib import Path
task = json.loads(os.environ["TASK_JSON"])
print(Path(task["cnf_file"]).name)
PY
)
TASK_LOG_DIR="{task_log_root}/$ALGORITHM_ID"
mkdir -p "$TASK_LOG_DIR"
TASK_LOG_PATH="$TASK_LOG_DIR/proof_verify_${{CODE_ID}}_${{CNF_FILE}}_${{SLURM_ARRAY_JOB_ID:-nojob}}_${{SLURM_ARRAY_TASK_ID:-notask}}.log"
exec >"$TASK_LOG_PATH" 2>&1
mkdir -p "$PARTS_DIR"
cd "{REPO_ROOT}"
PYTHONPATH=src "{python_bin}" scripts/verify_iteration_proofs.py \\
  --run-task-json-env TASK_JSON \\
  --task-output-dir "$PARTS_DIR" \\
  --drat_trim "{drat_trim_cmd}" \\
  --check_timeout "{check_timeout_sec}"
"""
    with open(array_script_path, "w") as f:
        f.write(array_script)
    os.chmod(array_script_path, 0o755)

    if nersc:
        slurm_array_wrapper = wrap_command_to_slurm_array_nersc
        slurm_single_wrapper = wrap_command_to_slurm_nersc
    else:
        slurm_array_wrapper = wrap_command_to_slurm_array
        slurm_single_wrapper = wrap_command_to_slurm

    slurm_kwargs: dict[str, str] = {}
    if slurm_constraint:
        slurm_kwargs["constraint"] = slurm_constraint
    if nersc and slurm_qos:
        slurm_kwargs["qos"] = slurm_qos

    array_cmd = slurm_array_wrapper(
        script_path=array_script_path,
        array_range=f"0-{len(pending_tasks) - 1}",
        account=slurm_account,
        mem=slurm_mem,
        time=slurm_time,
        job_name=f"proof_verify_{generation_tag}",
        output_file=array_log,
        max_concurrent=max_concurrent,
        **slurm_kwargs,
    )
    array_output = os.popen(array_cmd).read().strip()
    if not array_output or "error" in array_output.lower():
        print(f"ERROR: failed to submit proof verification array: {array_output}", file=sys.stderr)
        return 2
    array_job_id = array_output.split()[-1]

    collect_cmd = (
        f"cd {REPO_ROOT} && "
        f"PYTHONPATH=src {python_bin} scripts/verify_iteration_proofs.py "
        f"--collect-slurm-results "
        f"--generation_tag {generation_tag} "
        f"--tasks-path {tasks_path} "
        f"--task-output-dir {parts_dir} "
        f"--valid-records-path {valid_records_path} "
        f"--invalid-records-path {invalid_records_path} "
        f"--timeout-records-path {timeout_records_path}"
    )
    collect_sbatch = slurm_single_wrapper(
        collect_cmd,
        account=slurm_account,
        mem="1G",
        time="02:00:00",
        job_name=f"proof_collect_{generation_tag}",
        output_file=collect_log,
        dependencies=[array_job_id],
        dependency_type="afterany",
        **slurm_kwargs,
    )
    collect_output = os.popen(collect_sbatch).read().strip()
    if not collect_output or "error" in collect_output.lower():
        print(f"ERROR: failed to submit proof verification collector: {collect_output}", file=sys.stderr)
        return 2
    collect_job_id = collect_output.split()[-1]

    with open(metadata_path, "w") as f:
        json.dump(
                {
                    "generation_tag": generation_tag,
                    "status": "submitted",
                    "task_count": len(pending_tasks),
                    "total_unsat_task_count": len(tasks),
                    "reused_task_count": len(reused_task_records),
                    "valid_count": sum(record.status == "valid" for record in records),
                    "invalid_count": sum(record.status == "invalid" for record in records),
                    "timeout_count": sum(record.status == "timeout" for record in records),
                    "array_job_id": array_job_id,
                    "collector_job_id": collect_job_id,
                    "tasks_path": tasks_path,
                    "parts_dir": parts_dir,
                "valid_records_path": valid_records_path,
                "invalid_records_path": invalid_records_path,
                "timeout_records_path": timeout_records_path,
                "report_path": report_path,
                "array_log_root": task_log_root,
                "collector_log": collect_log,
                "drat_trim": drat_trim_cmd,
                "nersc": nersc,
                "slurm_constraint": slurm_constraint,
                "slurm_qos": slurm_qos,
            },
            f,
            indent=2,
        )

    print(
        f"Submitted proof verification array job {array_job_id} with "
        f"{len(pending_tasks)} task(s); reused {len(reused_task_records)} existing result(s)"
    )
    print(f"Submitted proof verification collector job {collect_job_id}")
    print(f"Submission metadata written to: {metadata_path}")
    return 0


def run_single_task_from_env(
    env_var: str,
    task_output_dir: str,
    drat_trim_cmd: str,
    check_timeout_sec: int,
) -> int:
    task_json = os.environ.get(env_var)
    if not task_json:
        print(f"ERROR: environment variable {env_var} is empty", file=sys.stderr)
        return 2
    task = _task_from_dict(json.loads(task_json))
    record = run_proof_task(task, drat_trim_cmd, check_timeout_sec)
    os.makedirs(task_output_dir, exist_ok=True)
    output_path = _task_result_path(task_output_dir, task)
    with open(output_path, "w") as f:
        json.dump(_record_to_dict(record), f, indent=2)
    print(f"Wrote proof verification result to: {output_path}")
    return 0


def collect_slurm_results(
    generation_tag: str,
    tasks_path: str,
    task_output_dir: str,
    valid_records_path: Optional[str] = None,
    invalid_records_path: Optional[str] = None,
    timeout_records_path: Optional[str] = None,
) -> int:
    out_dir = get_generation_output_dir(generation_tag)
    report_path = os.path.join(out_dir, "proof_verification.json")

    with open(tasks_path, "r") as f:
        tasks = [_task_from_dict(item) for item in json.load(f)]

    records: List[ProofCheckRecord] = []
    if valid_records_path and os.path.exists(valid_records_path):
        with open(valid_records_path, "r") as f:
            records.extend(_record_from_dict(item) for item in json.load(f))
    if invalid_records_path and os.path.exists(invalid_records_path):
        with open(invalid_records_path, "r") as f:
            records.extend(_record_from_dict(item) for item in json.load(f))
    if timeout_records_path and os.path.exists(timeout_records_path):
        with open(timeout_records_path, "r") as f:
            records.extend(_record_from_dict(item) for item in json.load(f))

    for task in tasks:
        result_path = os.path.join(
            task_output_dir,
            f"{task.algorithm_id}__{task.code_id}__{task.cnf_file}.json",
        )
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                records.append(_record_from_dict(json.load(f)))
        else:
            records.append(
                ProofCheckRecord(
                    algorithm_id=task.algorithm_id,
                    code_id=task.code_id,
                    cnf_file=task.cnf_file,
                    cnf_path=task.cnf_path,
                    proof_path=task.proof_path,
                    status="failed",
                    message="Missing per-task validation result",
                )
            )

    report = build_validation_report(records)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Collected {len(records)} proof verification results")
    print(f"Report written to: {report_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify UNSAT proofs for a generation tag")
    parser.add_argument("--generation_tag", help="Generation tag to verify")
    parser.add_argument(
        "--benchmark_path",
        default=SAT2025_BENCHMARK_PATH,
        help=f"Benchmark CNF directory (default: {SAT2025_BENCHMARK_PATH})",
    )
    parser.add_argument(
        "--drat_trim",
        default="drat-trim",
        help="drat-trim executable (default: drat-trim)",
    )
    parser.add_argument(
        "--check_timeout",
        type=int,
        default=7200,
        help="Timeout per drat-trim call in seconds (default: 7200)",
    )
    parser.add_argument(
        "--stop-on-first-fail",
        action="store_true",
        help="Stop immediately when a proof check fails",
    )
    parser.add_argument("--submit-slurm", action="store_true", help="Submit one SLURM array task per solver-instance pair plus a collector job")
    parser.add_argument("--collect-slurm-results", action="store_true", help="Collect per-task SLURM verification outputs into final JSON")
    parser.add_argument("--run-task-json-env", help="Environment variable containing one serialized proof task JSON")
    parser.add_argument("--task-output-dir", help="Directory to store one JSON output per proof task")
    parser.add_argument("--tasks-path", help="Path to serialized proof verification task list")
    parser.add_argument("--valid-records-path", help="Path to serialized valid SAT log records")
    parser.add_argument("--invalid-records-path", help="Path to serialized invalid log records")
    parser.add_argument("--timeout-records-path", help="Path to serialized timeout log records")
    parser.add_argument("--slurm-account", default=None, help="SLURM account for submitted validation jobs (default: def-vganesh, or m4831 with --nersc)")
    parser.add_argument("--slurm-mem", default="8G", help="Memory per validation task")
    parser.add_argument("--slurm-time", default="01:00:00", help="Wall time per validation task")
    parser.add_argument("--slurm-max-concurrent", type=int, default=200, help="Max concurrent proof validation array tasks")
    parser.add_argument("--nersc", action="store_true", help="Use NERSC SLURM wrapper (supports qos/constraint)")
    parser.add_argument("--slurm-constraint", default=None, help="SLURM constraint (default: cpu with --nersc)")
    parser.add_argument("--slurm-qos", default=None, help="SLURM QoS (default: regular with --nersc)")
    args = parser.parse_args()

    if args.run_task_json_env:
        drat_path = resolve_executable(args.drat_trim)
        if drat_path is None:
            print(
                f"ERROR: drat-trim executable not found: {args.drat_trim}",
                file=sys.stderr,
            )
            return 2
        if not args.task_output_dir:
            print("ERROR: --task-output-dir is required with --run-task-json-env", file=sys.stderr)
            return 2
        return run_single_task_from_env(
            args.run_task_json_env,
            args.task_output_dir,
            drat_path,
            args.check_timeout,
        )

    if args.collect_slurm_results:
        if not args.generation_tag:
            print("ERROR: --generation_tag is required with --collect-slurm-results", file=sys.stderr)
            return 2
        if not args.tasks_path or not args.task_output_dir:
            print("ERROR: --tasks-path and --task-output-dir are required with --collect-slurm-results", file=sys.stderr)
            return 2
        return collect_slurm_results(
            args.generation_tag,
            args.tasks_path,
            args.task_output_dir,
            args.valid_records_path,
            args.invalid_records_path,
            args.timeout_records_path,
        )

    if not args.generation_tag:
        print("ERROR: --generation_tag is required", file=sys.stderr)
        return 2

    if not os.path.isdir(args.benchmark_path):
        print(f"ERROR: benchmark path not found: {args.benchmark_path}", file=sys.stderr)
        return 2

    drat_path = resolve_executable(args.drat_trim)
    if drat_path is None:
        print(
            f"ERROR: drat-trim executable not found: {args.drat_trim}",
            file=sys.stderr,
        )
        return 2

    if args.submit_slurm:
        slurm_account = args.slurm_account or ("m4831" if args.nersc else "def-vganesh")
        slurm_constraint = args.slurm_constraint
        slurm_qos = args.slurm_qos
        if args.nersc:
            if not slurm_constraint:
                slurm_constraint = "cpu"
            if not slurm_qos:
                slurm_qos = "regular"

        return submit_proof_verification_slurm(
            generation_tag=args.generation_tag,
            benchmark_path=args.benchmark_path,
            drat_trim_cmd=drat_path,
            check_timeout_sec=args.check_timeout,
            slurm_account=slurm_account,
            slurm_mem=args.slurm_mem,
            slurm_time=args.slurm_time,
            max_concurrent=args.slurm_max_concurrent,
            nersc=args.nersc,
            slurm_constraint=slurm_constraint,
            slurm_qos=slurm_qos,
        )

    checked, failed, skipped, records = verify_generation_proofs(
        generation_tag=args.generation_tag,
        benchmark_path=args.benchmark_path,
        drat_trim_cmd=drat_path,
        check_timeout_sec=args.check_timeout,
        stop_on_first_fail=args.stop_on_first_fail,
    )

    out_dir = get_generation_output_dir(args.generation_tag)
    report_path = os.path.join(out_dir, "proof_verification.json")
    report = build_validation_report(records)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(
        f"Proof verification summary for {args.generation_tag}: "
        f"checked={checked}, failed={failed}, skipped={skipped}"
    )
    print(f"Report written to: {report_path}")

    return 1 if failed > 0 else 0


def shutil_which(cmd: str) -> Optional[str]:
    # Tiny helper to avoid importing shutil just for one call in hot loops.
    for p in os.environ.get("PATH", "").split(os.pathsep):
        candidate = os.path.join(p, cmd)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def resolve_executable(cmd: str) -> Optional[str]:
    # Accept explicit relative/absolute paths in addition to PATH lookups.
    if os.path.sep in cmd or (os.path.altsep and os.path.altsep in cmd):
        candidate = os.path.abspath(cmd)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
        return None
    return shutil_which(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
