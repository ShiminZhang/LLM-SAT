#!/usr/bin/env python3
"""OpenEvolve evaluator for configurable single-function LLM-SAT comparisons.

Each evaluation copies the fixed LLM-SAT base solver, injects the candidate
implementation, builds Kissat, and submits one eight-core SLURM job. The job
runs independent single-threaded Kissat processes over the configured benchmark
family. Fitness is a monotone transform of per-instance PAR2 CPU runtime only.

The protocol and candidate hashes form a durable cache key. Repeated programs
reuse a measurement only when the complete evaluation protocol also matches.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openevolve.evaluation_result import EvaluationResult


TARGET_SPECS = {
    "kissat_decide_phase": (
        re.compile(
            r"\bint\s+kissat_decide_phase\s*\(\s*kissat\s*\*\s*solver\s*,"
            r"\s*unsigned\s+idx\s*\)\s*\{",
            re.MULTILINE,
        ),
        Path("src/decide.c"),
    ),
    "kissat_restarting": (
        re.compile(
            r"\bbool\s+kissat_restarting\s*\(\s*kissat\s*\*\s*solver\s*\)\s*\{",
            re.MULTILINE,
        ),
        Path("src/restart.c"),
    ),
}


def _target_spec(name: str) -> Tuple[re.Pattern[str], Path]:
    try:
        pattern, default_source = TARGET_SPECS[name]
    except KeyError as exc:
        supported = ", ".join(sorted(TARGET_SPECS))
        raise ValueError(f"Unsupported OE_TARGET_FUNCTION={name!r}; choose: {supported}") from exc
    source_setting = os.environ.get("OE_TARGET_SOURCE")
    source = Path(source_setting) if source_setting else default_source
    if source.is_absolute() or ".." in source.parts or not source.parts:
        raise ValueError(f"OE_TARGET_SOURCE must be a safe relative path: {source}")
    return pattern, source


TARGET_FUNCTION = os.environ.get("OE_TARGET_FUNCTION", "kissat_decide_phase")
TARGET_PATTERN, TARGET_SOURCE = _target_spec(TARGET_FUNCTION)

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("OE_REPO_ROOT", THIS_DIR.parents[1])).resolve()
BASE_SOLVER = Path(
    os.environ.get("OE_BASE_SOLVER", REPO_ROOT / "solvers" / "base")
).resolve()
BENCHMARK_DIR = Path(
    os.environ.get(
        "OE_BENCHMARK_DIR",
        REPO_ROOT / "data" / "benchmarks" / "formula-families" / "cryptography-ascon",
    )
).resolve()
WORK_ROOT = Path(os.environ.get("OE_WORK_DIR", THIS_DIR / "work")).resolve()

SOLVER_TIMEOUT = int(os.environ.get("OE_TIMEOUT", "1200"))
PAR2_PENALTY = float(os.environ.get("OE_PAR2_PENALTY", str(2 * SOLVER_TIMEOUT)))
SLURM_WALL_TIME = os.environ.get("OE_WALL_TIME", "10:00:00")
SLURM_ACCOUNT = os.environ.get("OE_SLURM_ACCOUNT", "def-vganesh")
SLURM_MEMORY = os.environ.get("OE_SLURM_MEMORY", "32G")
SLURM_CPUS = int(os.environ.get("OE_SLURM_CPUS", "8"))
SLURM_CONSTRAINT = os.environ.get("OE_SLURM_CONSTRAINT", "").strip()
# Retained for reading/resuming legacy array-backed candidate work directories.
SLURM_MAX_CONCURRENT = int(os.environ.get("OE_SLURM_MAX_CONCURRENT", "1000"))
POLL_INTERVAL = int(os.environ.get("OE_POLL_INTERVAL", "30"))
MAX_SUBMISSION_ATTEMPTS = int(os.environ.get("OE_MAX_SUBMISSION_ATTEMPTS", "2"))
SUBMIT_RETRY_ATTEMPTS = int(os.environ.get("OE_SUBMIT_RETRY_ATTEMPTS", "120"))
SUBMIT_RETRY_INTERVAL = int(os.environ.get("OE_SUBMIT_RETRY_INTERVAL", "30"))
KEEP_PROOFS = os.environ.get("OE_KEEP_PROOFS", "0").lower() in {"1", "true", "yes"}
CACHE_FORMAT_VERSION = 2


def _tail(text: str, limit: int = 12000) -> str:
    return text if len(text) <= limit else text[-limit:]


def _run(
    command: List[str],
    *,
    cwd: Optional[Path] = None,
    timeout: Optional[int] = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )


def _matching_brace(source: str, opening: int) -> int:
    """Return the matching brace while ignoring C comments and strings."""
    depth = 0
    state = "code"
    i = opening
    while i < len(source):
        char = source[i]
        nxt = source[i + 1] if i + 1 < len(source) else ""

        if state == "code":
            if char == "/" and nxt == "/":
                state = "line_comment"
                i += 2
                continue
            if char == "/" and nxt == "*":
                state = "block_comment"
                i += 2
                continue
            if char == '"':
                state = "string"
            elif char == "'":
                state = "char"
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return i
        elif state == "line_comment":
            if char == "\n":
                state = "code"
        elif state == "block_comment":
            if char == "*" and nxt == "/":
                state = "code"
                i += 2
                continue
        elif state in {"string", "char"}:
            if char == "\\":
                i += 2
                continue
            if (state == "string" and char == '"') or (state == "char" and char == "'"):
                state = "code"
        i += 1
    raise ValueError(f"Unbalanced braces in {TARGET_FUNCTION}")


def _function_span(source: str) -> Tuple[int, int]:
    matches = list(TARGET_PATTERN.finditer(source))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one exact {TARGET_FUNCTION} definition; found {len(matches)}"
        )
    match = matches[0]
    opening = source.find("{", match.start(), match.end())
    closing = _matching_brace(source, opening)
    return match.start(), closing + 1


def _extract_candidate(program_path: str) -> str:
    source = Path(program_path).read_text(encoding="utf-8")
    start, end = _function_span(source)
    function = source[start:end].strip()
    if "#include" in function:
        raise ValueError("The evolved function may not add #include directives")
    return function + "\n"


def _inject_function(target_path: Path, candidate: str) -> None:
    source = target_path.read_text(encoding="utf-8")
    start, end = _function_span(source)
    target_path.write_text(
        source[:start] + candidate.rstrip() + source[end:], encoding="utf-8"
    )


def _build_solver(candidate: str, candidate_dir: Path) -> Tuple[Optional[Path], str]:
    solver_dir = candidate_dir / "solver"
    build_log = candidate_dir / "build.log"
    binary = solver_dir / "kissat"
    if binary.is_file() and os.access(binary, os.X_OK):
        return solver_dir, build_log.read_text(errors="replace") if build_log.exists() else ""

    if solver_dir.exists():
        shutil.rmtree(solver_dir)
    shutil.copytree(BASE_SOLVER, solver_dir, symlinks=True)
    target_path = solver_dir / TARGET_SOURCE
    if not target_path.is_file():
        return None, f"Target source file not found in copied solver: {target_path}"
    _inject_function(target_path, candidate)

    # The copied base solver contains location-specific generated makefiles.
    for stale in (solver_dir / "build" / "makefile", solver_dir / "src" / "makefile"):
        if stale.is_symlink() or stale.exists():
            stale.unlink()

    configure = solver_dir / "configure"
    configure.chmod(configure.stat().st_mode | 0o111)
    logs: List[str] = []

    try:
        configured = _run(["./configure", "-c"], cwd=solver_dir, timeout=120)
        logs.append("$ ./configure -c\n" + configured.stdout)
        if configured.returncode != 0:
            build_log.write_text("\n".join(logs), encoding="utf-8")
            return None, "\n".join(logs)

        built = _run(["make", "-j4"], cwd=solver_dir, timeout=600)
        logs.append("$ make -j4\n" + built.stdout)
        if built.returncode != 0 or not (solver_dir / "build" / "kissat").exists():
            build_log.write_text("\n".join(logs), encoding="utf-8")
            return None, "\n".join(logs)
    except subprocess.TimeoutExpired as exc:
        logs.append(f"Build timed out: {exc}")
        build_log.write_text("\n".join(logs), encoding="utf-8")
        return None, "\n".join(logs)

    shutil.copy2(solver_dir / "build" / "kissat", binary)
    binary.chmod(binary.stat().st_mode | 0o111)
    build_log.write_text("\n".join(logs), encoding="utf-8")
    return solver_dir, "\n".join(logs)


def _prune_solver_tree(candidate_dir: Path) -> None:
    """Drop the copied/build solver after durable metrics have been written.

    Candidate source, per-instance logs/proofs, metrics, and artifacts stay in
    the cache.  A cache hit only needs metrics.json and artifacts.json, while a
    cache miss can rebuild from the clean base solver.
    """
    solver_dir = candidate_dir / "solver"
    if solver_dir.exists():
        shutil.rmtree(solver_dir, ignore_errors=True)


def _benchmark_files() -> List[str]:
    list_setting = os.environ.get("OE_BENCHMARK_LIST")
    if list_setting:
        list_path = Path(list_setting)
        if not list_path.is_absolute():
            list_path = REPO_ROOT / list_path
        names = [
            line.strip()
            for line in list_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        names = sorted(path.name for path in BENCHMARK_DIR.glob("*.cnf"))

    missing = [name for name in names if not (BENCHMARK_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} benchmark files are missing; first missing file: {missing[0]}"
        )
    if not names:
        raise ValueError(f"No CNF files selected from {BENCHMARK_DIR}")
    return names


def _protocol_fingerprint(names: List[str]) -> str:
    """Identify the complete runtime protocol used by a cached measurement."""
    digest = hashlib.sha256()
    settings = {
        "benchmark_dir": str(BENCHMARK_DIR),
        "solver_timeout": SOLVER_TIMEOUT,
        "par2_penalty": PAR2_PENALTY,
        "candidate_job_cpus": SLURM_CPUS,
        "candidate_job_walltime": SLURM_WALL_TIME,
        "slurm_constraint": SLURM_CONSTRAINT,
        "solver_flags": "-s --check=1",
        "benchmarks": names,
    }
    # Preserve the existing decide-phase cache namespace. Alternative targets
    # add explicit identity fields so their measurements can never collide.
    if TARGET_FUNCTION != "kissat_decide_phase" or TARGET_SOURCE != Path("src/decide.c"):
        settings["target_function"] = TARGET_FUNCTION
        settings["target_source"] = str(TARGET_SOURCE)
    digest.update(json.dumps(settings, sort_keys=True).encode("utf-8"))

    # Cache entries must also be invalidated if the fixed base solver changes.
    source_root = BASE_SOLVER / "src"
    source_files = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix in {".c", ".h"}
    )
    for extra in (BASE_SOLVER / "configure", BASE_SOLVER / "makefile.in"):
        if extra.is_file():
            source_files.append(extra)
    for path in source_files:
        digest.update(str(path.relative_to(BASE_SOLVER)).encode("utf-8"))
        digest.update(path.read_bytes())

    # Hashing every multi-GB CNF on each worker would be disproportionate. Names,
    # sizes, and mtimes still detect normal benchmark replacement or regeneration.
    for name in names:
        stat = (BENCHMARK_DIR / name).stat()
        digest.update(f"{name}\0{stat.st_size}\0{stat.st_mtime_ns}".encode("utf-8"))
    return digest.hexdigest()[:20]


def _write_array_script(candidate_dir: Path, solver_dir: Path) -> Path:
    script = candidate_dir / "run_solver_array.sh"
    keep_proofs = "1" if KEEP_PROOFS else "0"
    content = f"""#!/bin/bash
CNF_LIST={json.dumps(str(candidate_dir / 'cnf_files.txt'))}
SOLVER={json.dumps(str(solver_dir / 'kissat'))}
BENCHMARK_PATH={json.dumps(str(BENCHMARK_DIR))}
RESULT_DIR={json.dumps(str(candidate_dir / 'results'))}
TIMEOUT={SOLVER_TIMEOUT}
KEEP_PROOFS={keep_proofs}
SOLVER_FLAGS="-s --check=1"

if [ -x /usr/bin/time ]; then
    GNU_TIME=/usr/bin/time
else
    GNU_TIME=$(which time 2>/dev/null)
fi

CNF_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CNF_LIST")
OUTPUT_FILE="$RESULT_DIR/${{CNF_FILE}}.solving.log"
PROOF_DIR="$RESULT_DIR/proofs"
PROOF_FILE="$PROOF_DIR/${{CNF_FILE}}.proof"

if [ -z "$CNF_FILE" ]; then
    exit 0
fi
if [ -f "$OUTPUT_FILE" ] && grep -q '^OE_STATUS=' "$OUTPUT_FILE"; then
    exit 0
fi

mkdir -p "$PROOF_DIR"
"$GNU_TIME" -f "OE_GNU_TIME=%U %S" -o "${{OUTPUT_FILE}}.time" \\
    timeout "${{TIMEOUT}}s" "$SOLVER" $SOLVER_FLAGS \\
    "$BENCHMARK_PATH/$CNF_FILE" "$PROOF_FILE" > "$OUTPUT_FILE" 2>&1
EXIT_CODE=$?

if [ -f "${{OUTPUT_FILE}}.time" ]; then
    # Kissat returns 10/20 for SAT/UNSAT, so GNU time also writes a diagnostic
    # line. Select only our tagged timing record instead of parsing both lines.
    CPU_TIME=$(awk -F '[= ]+' '/^OE_GNU_TIME=/ {{printf "%.6f", $2 + $3}}' \\
        "${{OUTPUT_FILE}}.time")
    rm -f "${{OUTPUT_FILE}}.time"
else
    CPU_TIME=""
fi

if [ "$EXIT_CODE" -eq 10 ] || [ "$EXIT_CODE" -eq 20 ]; then
    echo "OE_STATUS=SOLVED" >> "$OUTPUT_FILE"
    echo "OE_CPU_TIME=$CPU_TIME" >> "$OUTPUT_FILE"
elif [ "$EXIT_CODE" -eq 124 ]; then
    echo "OE_STATUS=TIMEOUT" >> "$OUTPUT_FILE"
else
    echo "OE_STATUS=ERROR" >> "$OUTPUT_FILE"
    echo "OE_EXIT_CODE=$EXIT_CODE" >> "$OUTPUT_FILE"
fi

if [ "$KEEP_PROOFS" -ne 1 ] || [ "$EXIT_CODE" -ne 20 ]; then
    rm -f "$PROOF_FILE"
fi
exit 0
"""
    script.write_text(content, encoding="utf-8")
    script.chmod(0o755)
    return script


def _write_candidate_script(candidate_dir: Path, solver_dir: Path) -> Path:
    """Write one eight-core job that dynamically evaluates all pending CNFs."""
    script = candidate_dir / "run_candidate_job.sh"
    keep_proofs = "1" if KEEP_PROOFS else "0"
    content = f"""#!/bin/bash
CNF_LIST={json.dumps(str(candidate_dir / 'cnf_files.txt'))}
SOLVER={json.dumps(str(solver_dir / 'kissat'))}
BENCHMARK_PATH={json.dumps(str(BENCHMARK_DIR))}
RESULT_DIR={json.dumps(str(candidate_dir / 'results'))}
TIMEOUT={SOLVER_TIMEOUT}
EVAL_CORES={SLURM_CPUS}
KEEP_PROOFS={keep_proofs}
SOLVER_FLAGS="-s --check=1"

if [ -x /usr/bin/time ]; then
    GNU_TIME=/usr/bin/time
else
    GNU_TIME=$(which time 2>/dev/null)
fi

run_one() {{
    local CNF_FILE="$1"
    local OUTPUT_FILE="$RESULT_DIR/${{CNF_FILE}}.solving.log"
    local PROOF_DIR="$RESULT_DIR/proofs"
    local PROOF_FILE="$PROOF_DIR/${{CNF_FILE}}.proof"
    local EXIT_CODE CPU_TIME

    if [ -f "$OUTPUT_FILE" ] && grep -q '^OE_STATUS=' "$OUTPUT_FILE"; then
        return 0
    fi

    mkdir -p "$PROOF_DIR"
    "$GNU_TIME" -f "OE_GNU_TIME=%U %S" -o "${{OUTPUT_FILE}}.time" \
        timeout "${{TIMEOUT}}s" "$SOLVER" $SOLVER_FLAGS \
        "$BENCHMARK_PATH/$CNF_FILE" "$PROOF_FILE" > "$OUTPUT_FILE" 2>&1
    EXIT_CODE=$?

    if [ -f "${{OUTPUT_FILE}}.time" ]; then
        CPU_TIME=$(awk -F '[= ]+' '/^OE_GNU_TIME=/ {{printf "%.6f", $2 + $3}}' \
            "${{OUTPUT_FILE}}.time")
        rm -f "${{OUTPUT_FILE}}.time"
    else
        CPU_TIME=""
    fi

    if [ "$EXIT_CODE" -eq 10 ] || [ "$EXIT_CODE" -eq 20 ]; then
        echo "OE_STATUS=SOLVED" >> "$OUTPUT_FILE"
        echo "OE_CPU_TIME=$CPU_TIME" >> "$OUTPUT_FILE"
    elif [ "$EXIT_CODE" -eq 124 ]; then
        echo "OE_STATUS=TIMEOUT" >> "$OUTPUT_FILE"
    else
        echo "OE_STATUS=ERROR" >> "$OUTPUT_FILE"
        echo "OE_EXIT_CODE=$EXIT_CODE" >> "$OUTPUT_FILE"
    fi

    if [ "$KEEP_PROOFS" -ne 1 ] || [ "$EXIT_CODE" -ne 20 ]; then
        rm -f "$PROOF_FILE"
    fi
    return 0
}}

# Dynamic worker pool: a free core immediately takes the next CNF.
RUNNING=0
while IFS= read -r CNF_FILE; do
    [ -n "$CNF_FILE" ] || continue
    run_one "$CNF_FILE" &
    RUNNING=$((RUNNING + 1))
    if [ "$RUNNING" -ge "$EVAL_CORES" ]; then
        wait -n
        RUNNING=$((RUNNING - 1))
    fi
done < "$CNF_LIST"
wait
"""
    script.write_text(content, encoding="utf-8")
    script.chmod(0o755)
    return script


def _is_job_active(job_id: int) -> bool:
    result = _run(["squeue", "-h", "-j", str(job_id), "-o", "%T"], timeout=60)
    if result.returncode != 0:
        # Slurm reports old/purged job IDs as an error rather than an empty
        # queue result. Such a job cannot still be active.
        if "invalid job id" in result.stdout.lower():
            return False
        # Treat transient scheduler-query failures as active to avoid duplicate arrays.
        return True
    return bool(result.stdout.strip())


def _wait_for_job(job_id: int) -> None:
    while _is_job_active(job_id):
        time.sleep(max(1, POLL_INTERVAL))


def _pending_benchmarks(candidate_dir: Path, names: Iterable[str]) -> List[str]:
    result_dir = candidate_dir / "results"
    pending = []
    for name in names:
        log = result_dir / f"{name}.solving.log"
        if not log.exists() or "OE_STATUS=" not in log.read_text(errors="replace"):
            pending.append(name)
    return pending


def _retryable_submission_failure(output: str) -> bool:
    """Return whether a failed sbatch is likely to clear without code changes."""
    lowered = output.lower()
    return any(
        marker in lowered
        for marker in (
            "assocmaxsubmitjoblimit",
            "job violates accounting/qos policy",
            "qosmaxsubmitjobperuserlimit",
            "qosmaxjobperuserlimit",
            "socket timed out",
            "temporarily unable",
            "slurm controller not responding",
        )
    )


def _submit_array(candidate_dir: Path, script: Path, names: List[str], attempt: int) -> int:
    (candidate_dir / "cnf_files.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    array_limit = min(len(names), SLURM_MAX_CONCURRENT)
    command = [
        "sbatch",
        "--parsable",
        f"--account={SLURM_ACCOUNT}",
        f"--mem={SLURM_MEMORY}",
        f"--time={SLURM_WALL_TIME}",
        f"--array=0-{len(names) - 1}%{array_limit}",
        f"--job-name=oe_{candidate_dir.name[:10]}",
        f"--output={candidate_dir}/slurm_{attempt}_%A_%a.log",
        str(script),
    ]
    if SLURM_CONSTRAINT:
        command.insert(-1, f"--constraint={SLURM_CONSTRAINT}")
    for retry in range(1, max(1, SUBMIT_RETRY_ATTEMPTS) + 1):
        submitted = _run(command, cwd=REPO_ROOT, timeout=120)
        if submitted.returncode == 0:
            raw_id = submitted.stdout.strip().splitlines()[-1].split(";", 1)[0]
            if not raw_id.isdigit():
                raise RuntimeError(
                    f"Could not parse sbatch job id from: {submitted.stdout!r}"
                )
            return int(raw_id)

        output = submitted.stdout.strip()
        if retry >= max(1, SUBMIT_RETRY_ATTEMPTS) or not _retryable_submission_failure(
            output
        ):
            raise RuntimeError(f"sbatch failed: {output}")

        print(
            "Transient Slurm submission limit; "
            f"retrying sbatch in {SUBMIT_RETRY_INTERVAL}s "
            f"({retry}/{SUBMIT_RETRY_ATTEMPTS}): {_tail(output, 1000)}",
            flush=True,
        )
        time.sleep(max(1, SUBMIT_RETRY_INTERVAL))

    raise AssertionError("unreachable")


def _submit_candidate_job(
    candidate_dir: Path, script: Path, names: List[str], attempt: int
) -> int:
    """Submit one multi-core job for all currently pending CNFs."""
    (candidate_dir / "cnf_files.txt").write_text(
        "\n".join(names) + "\n", encoding="utf-8"
    )
    command = [
        "sbatch",
        "--parsable",
        f"--account={SLURM_ACCOUNT}",
        f"--mem={SLURM_MEMORY}",
        f"--time={SLURM_WALL_TIME}",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={SLURM_CPUS}",
        f"--job-name=oe_{candidate_dir.name[:10]}",
        f"--output={candidate_dir}/slurm_{attempt}_%j.log",
        str(script),
    ]
    if SLURM_CONSTRAINT:
        command.insert(-1, f"--constraint={SLURM_CONSTRAINT}")
    for retry in range(1, max(1, SUBMIT_RETRY_ATTEMPTS) + 1):
        submitted = _run(command, cwd=REPO_ROOT, timeout=120)
        if submitted.returncode == 0:
            raw_id = submitted.stdout.strip().splitlines()[-1].split(";", 1)[0]
            if not raw_id.isdigit():
                raise RuntimeError(
                    f"Could not parse sbatch job id from: {submitted.stdout!r}"
                )
            return int(raw_id)

        output = submitted.stdout.strip()
        if retry >= max(1, SUBMIT_RETRY_ATTEMPTS) or not _retryable_submission_failure(
            output
        ):
            raise RuntimeError(f"sbatch failed: {output}")

        print(
            "Transient Slurm submission limit; "
            f"retrying sbatch in {SUBMIT_RETRY_INTERVAL}s "
            f"({retry}/{SUBMIT_RETRY_ATTEMPTS}): {_tail(output, 1000)}",
            flush=True,
        )
        time.sleep(max(1, SUBMIT_RETRY_INTERVAL))

    raise AssertionError("unreachable")


def _evaluate_on_slurm(candidate_dir: Path, solver_dir: Path, names: List[str]) -> List[int]:
    (candidate_dir / "results").mkdir(parents=True, exist_ok=True)
    script = _write_candidate_script(candidate_dir, solver_dir)
    jobs_path = candidate_dir / "job_ids.json"
    job_ids: List[int] = []
    if jobs_path.exists():
        job_ids = [int(value) for value in json.loads(jobs_path.read_text())]
        if job_ids and _is_job_active(job_ids[-1]):
            _wait_for_job(job_ids[-1])

    for attempt in range(len(job_ids) + 1, MAX_SUBMISSION_ATTEMPTS + 1):
        pending = _pending_benchmarks(candidate_dir, names)
        if not pending:
            break
        job_id = _submit_candidate_job(candidate_dir, script, pending, attempt)
        job_ids.append(job_id)
        jobs_path.write_text(json.dumps(job_ids, indent=2) + "\n", encoding="utf-8")
        _wait_for_job(job_id)
    return job_ids


def _parse_cpu_time(text: str) -> Tuple[Optional[float], bool]:
    """Parse CPU time and recover logs written by cache format version 1.

    Version 1 ran awk over both GNU time's diagnostic line and its numeric line,
    concatenating a zero with the real time (for example
    ``0.0000002728.530000``). The suffix is unambiguous and lets us reuse the
    completed solver measurements instead of resubmitting them.
    """
    match = re.search(r"^OE_CPU_TIME=(\S+)\s*$", text, re.MULTILINE)
    if not match:
        return None, False

    raw = match.group(1)
    try:
        return float(raw), False
    except ValueError:
        legacy_prefix = "0.000000"
        if raw.startswith(legacy_prefix):
            try:
                return float(raw[len(legacy_prefix) :]), True
            except ValueError:
                pass
    return None, False


def _parse_runtime(candidate_dir: Path, names: List[str]) -> Tuple[float, Dict[str, Any]]:
    times: List[float] = []
    solved = 0
    timed_out = 0
    errors = 0
    missing = 0
    legacy_times_recovered = 0

    for name in names:
        log_path = candidate_dir / "results" / f"{name}.solving.log"
        if not log_path.exists():
            missing += 1
            errors += 1
            times.append(PAR2_PENALTY)
            continue
        text = log_path.read_text(errors="replace")
        status_match = re.search(r"^OE_STATUS=(\w+)\s*$", text, re.MULTILINE)
        cpu_time, recovered_legacy_time = _parse_cpu_time(text)
        status = status_match.group(1) if status_match else "ERROR"
        if status == "SOLVED" and cpu_time is not None:
            if math.isfinite(cpu_time) and 0.0 <= cpu_time < SOLVER_TIMEOUT:
                solved += 1
                times.append(cpu_time)
                legacy_times_recovered += int(recovered_legacy_time)
                continue
        if status == "TIMEOUT":
            timed_out += 1
        else:
            errors += 1
        times.append(PAR2_PENALTY)

    par2 = sum(times) / len(times)
    details = {
        "instances": len(names),
        "solved": solved,
        "timeouts": timed_out,
        "errors": errors,
        "missing": missing,
        "legacy_times_recovered": legacy_times_recovered,
        "solver_timeout": SOLVER_TIMEOUT,
        "par2_penalty": PAR2_PENALTY,
        "candidate_job_cpus": SLURM_CPUS,
        "candidate_job_walltime": SLURM_WALL_TIME,
        "slurm_constraint": SLURM_CONSTRAINT,
    }
    return par2, details


def _result(metrics: Dict[str, float], artifacts: Dict[str, Any]) -> EvaluationResult:
    return EvaluationResult(metrics=metrics, artifacts=artifacts)


def evaluate(program_path: str) -> EvaluationResult:
    """Build and score one OpenEvolve candidate using runtime-only PAR2 fitness."""
    try:
        candidate = _extract_candidate(program_path)
    except Exception as exc:
        return _result(
            {"combined_score": 0.0, "par2": PAR2_PENALTY},
            {"stage": "candidate_parse", "error": str(exc)},
        )

    try:
        names = _benchmark_files()
        protocol = _protocol_fingerprint(names)
    except Exception as exc:
        return _result(
            {"combined_score": 0.0, "par2": PAR2_PENALTY},
            {"stage": "setup", "error": str(exc)},
        )

    digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
    candidate_dir = WORK_ROOT / protocol / digest
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "candidate.c").write_text(candidate, encoding="utf-8")

    lock_path = candidate_dir / "evaluate.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        metrics_path = candidate_dir / "metrics.json"
        artifacts_path = candidate_dir / "artifacts.json"
        if metrics_path.exists() and artifacts_path.exists():
            try:
                cached_metrics = json.loads(metrics_path.read_text())
                cached_artifacts = json.loads(artifacts_path.read_text())
                if cached_artifacts.get("cache_format_version") == CACHE_FORMAT_VERSION:
                    _prune_solver_tree(candidate_dir)
                    return _result(cached_metrics, cached_artifacts)
            except (OSError, ValueError, TypeError):
                # Rebuild or reparse incomplete/corrupt cache metadata below.
                pass

        if not BASE_SOLVER.is_dir():
            return _result(
                {"combined_score": 0.0, "par2": PAR2_PENALTY},
                {"stage": "setup", "error": f"Base solver not found: {BASE_SOLVER}"},
            )

        try:
            solver_dir, build_output = _build_solver(candidate, candidate_dir)
        except Exception as exc:
            solver_dir, build_output = None, f"Build setup failed: {exc}"
        if solver_dir is None:
            metrics = {"combined_score": 0.0, "par2": PAR2_PENALTY}
            artifacts = {
                "stage": "compile",
                "target_function": TARGET_FUNCTION,
                "target_source": str(TARGET_SOURCE),
                "cache_format_version": CACHE_FORMAT_VERSION,
                "candidate_hash": digest,
                "build_output": _tail(build_output),
            }
            metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
            artifacts_path.write_text(json.dumps(artifacts, indent=2) + "\n")
            _prune_solver_tree(candidate_dir)
            return _result(metrics, artifacts)

        try:
            job_ids = _evaluate_on_slurm(candidate_dir, solver_dir, names)
            par2, details = _parse_runtime(candidate_dir, names)
            # This is strictly monotone in PAR2 and contains no non-runtime term.
            combined_score = PAR2_PENALTY / max(par2, 1e-9)
            metrics = {"combined_score": combined_score, "par2": par2}
            artifacts = {
                "stage": "evaluation",
                "target_function": TARGET_FUNCTION,
                "target_source": str(TARGET_SOURCE),
                "cache_format_version": CACHE_FORMAT_VERSION,
                "candidate_hash": digest,
                "protocol_hash": protocol,
                "job_ids": job_ids,
                **details,
            }
        except Exception as exc:
            metrics = {"combined_score": 0.0, "par2": PAR2_PENALTY}
            artifacts = {
                "stage": "evaluation",
                "target_function": TARGET_FUNCTION,
                "target_source": str(TARGET_SOURCE),
                "cache_format_version": CACHE_FORMAT_VERSION,
                "candidate_hash": digest,
                "protocol_hash": protocol,
                "error": str(exc),
            }

        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        artifacts_path.write_text(json.dumps(artifacts, indent=2) + "\n", encoding="utf-8")
        _prune_solver_tree(candidate_dir)
        return _result(metrics, artifacts)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", help="Candidate .c program")
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Only parse, inject, and compile the candidate; do not submit SLURM jobs",
    )
    args = parser.parse_args()

    if not args.build_only:
        outcome = evaluate(args.program)
        print(json.dumps({"metrics": outcome.metrics, "artifacts": outcome.artifacts}, indent=2))
    else:
        code = _extract_candidate(args.program)
        digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
        names = _benchmark_files()
        directory = WORK_ROOT / _protocol_fingerprint(names) / digest
        directory.mkdir(parents=True, exist_ok=True)
        solver, log = _build_solver(code, directory)
        print(json.dumps({"success": solver is not None, "work_dir": str(directory)}, indent=2))
        if solver is None:
            print(_tail(log))
            raise SystemExit(1)
