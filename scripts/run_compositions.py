#!/usr/bin/env python3
"""
Build and full-eval a set of 2-function compositions.

Each composition pairs one variant of `restart_mab` with one variant of
`kissat_bump_score_increment` on a fresh copy of the AE_kissat2025_MAB base.

Steps (idempotent — rerun is safe):
  1. Build each composition under solvers/compositions/<name>/.
  2. Submit a full-eval SLURM job array for each successful build.
  3. Poll squeue until all submitted job arrays drain.
  4. Collect PAR2 from each result_full/ directory.
  5. Write outputs/compositions/full_eval_results.{json,txt}.

Usage:
  PYTHONPATH=src python scripts/run_compositions.py
  PYTHONPATH=src python scripts/run_compositions.py --build-only
  PYTHONPATH=src python scripts/run_compositions.py --collect-only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llmsat.llmsat import SAT2025_BENCHMARK_PATH, setup_logging, get_logger
from llmsat.pipelines.evaluation import EvaluationPipeline

setup_logging()
logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COMPO_FILE = REPO_ROOT / "outputs/compositions/restart_mab_x_kissat_bump_score_increment.json"
COMPO_SOLVER_ROOT = REPO_ROOT / "solvers/compositions"
COMPO_OUTPUT_DIR = REPO_ROOT / "outputs/compositions"
STATE_PATH = COMPO_OUTPUT_DIR / "state.json"
POLL_SECONDS = 90

# A clean AE_kissat2025_MAB extracted from the tarball — bypasses the
# existing solvers/AE_kissat2025_MAB/ which has only src/test/scripts (no
# top-level configure/makefile.in).
AE_TARBALL = REPO_ROOT / "AE_kissat2025_MAB.tar.xz"
AE_CLEAN_BASE = REPO_ROOT / "solvers/AE_kissat2025_MAB_clean"


def ensure_ae_clean_base() -> Path:
    """Extract AE_kissat2025_MAB.tar.xz into a clean dir if not already present."""
    sentinel = AE_CLEAN_BASE / "configure"
    if sentinel.exists():
        return AE_CLEAN_BASE
    AE_CLEAN_BASE.parent.mkdir(parents=True, exist_ok=True)
    if AE_CLEAN_BASE.exists():
        shutil.rmtree(AE_CLEAN_BASE)
    AE_CLEAN_BASE.mkdir()
    logger.info(f"Extracting {AE_TARBALL.name} -> {AE_CLEAN_BASE}")
    # Tarball wraps in AE_kissat2025_MAB/; strip it
    subprocess.run(
        ["tar", "-xf", str(AE_TARBALL), "--strip-components=1", "-C", str(AE_CLEAN_BASE)],
        check=True,
    )
    return AE_CLEAN_BASE


# ---------- state ----------

def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}

def save_state(state: dict) -> None:
    COMPO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


# ---------- build ----------

def build_composition(compo: dict, doc: dict) -> Path | None:
    """
    Build one composition. Returns solver path on success, None on failure.

    Strategy:
      - The .restart.c / .bump.c files in the source generation tag are
        complete copies of AE_kissat2025_MAB's src/{restart,bump}.c with
        only the target function body swapped out. So we just copy the
        AE base solver and overwrite those two files.
    """
    name = compo["name"]
    target = COMPO_SOLVER_ROOT / name

    # Resolve source files for both halves
    r_src = REPO_ROOT / find_candidate(doc, "restart_mab", compo["restart_mab_rank"])["function_source"]
    b_src = REPO_ROOT / find_candidate(doc, "kissat_bump_score_increment", compo["kissat_bump_score_increment_rank"])["function_source"]
    if not r_src.exists() or not b_src.exists():
        logger.error(f"[{name}] missing source: r_src={r_src.exists()} b_src={b_src.exists()}")
        return None

    # Skip rebuild if binary already exists
    bin_path = target / "build/kissat"
    if bin_path.exists():
        logger.info(f"[{name}] binary already built — skipping rebuild")
        return target

    # Fresh copy of AE base (extracted from tarball on first call)
    base = ensure_ae_clean_base()
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(base, target, symlinks=True)

    # Overwrite restart.c and bump.c
    shutil.copy2(r_src, target / "src/restart.c")
    shutil.copy2(b_src, target / "src/bump.c")

    # Clean stale build files (mirrors evaluation.py:586-591)
    for stale in [target / "build/makefile", target / "src/makefile"]:
        if stale.is_symlink() or stale.exists():
            stale.unlink()

    # Ensure configure is executable
    cfg = target / "configure"
    if cfg.exists():
        cfg.chmod(cfg.stat().st_mode | 0o111)

    # configure (no -c → NDEBUG, asserts off — matches base build)
    r = subprocess.run(["./configure"], cwd=target, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        logger.error(f"[{name}] configure failed:\n{r.stderr}")
        return None

    # Replace src/makefile symlink with regular file copy (same fix as setup_solver.sh / build_solver)
    src_makefile = target / "src/makefile"
    if src_makefile.is_symlink() or src_makefile.exists():
        src_makefile.unlink()
    shutil.copy2(target / "makefile", src_makefile)

    # make
    r = subprocess.run(["make", "-j4"], cwd=target, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        logger.error(f"[{name}] make failed:\n{r.stderr[-2000:]}")
        return None

    if not bin_path.exists():
        logger.error(f"[{name}] build/kissat missing after make")
        return None

    # Symlink/copy binary to top-level for convenience
    try:
        shutil.copy2(bin_path, target / "kissat")
    except Exception:
        pass

    logger.info(f"[{name}] build OK")
    return target


# ---------- submit ----------

def submit_full_eval(compo_name: str, solver_path: Path) -> int | None:
    """Submit a full-eval SLURM array for one composition. Returns job_id."""
    pipeline = EvaluationPipeline()
    result_dir = solver_path / "result_full"
    result_dir.mkdir(parents=True, exist_ok=True)

    # 400 CNFs from the SAT2025 benchmark dir (default)
    job_ids = pipeline.slurm_run_evaluate(
        solver_path=str(solver_path),
        benchmark_path=SAT2025_BENCHMARK_PATH,
        result_dir=str(result_dir),
    )
    if not job_ids:
        logger.error(f"[{compo_name}] SLURM submission failed")
        return None
    return job_ids[0]


# ---------- wait ----------

def wait_for_jobs(state: dict) -> None:
    pending = [
        (name, info["job_id"])
        for name, info in state.items()
        if info.get("job_id") and not info.get("collected")
    ]
    if not pending:
        return
    logger.info(f"Polling squeue for {len(pending)} job(s) every {POLL_SECONDS}s...")
    while True:
        remaining = []
        for name, jid in pending:
            try:
                r = subprocess.run(
                    ["squeue", "-u", os.environ["USER"], "-h", "-j", str(jid)],
                    capture_output=True, text=True, timeout=30,
                )
                if r.stdout.strip():
                    remaining.append((name, jid))
            except Exception as e:
                logger.warning(f"squeue check failed for {name} ({jid}): {e}")
        if not remaining:
            logger.info("All composition eval jobs cleared the queue")
            return
        logger.info(f"  waiting on: {[f'{n}({j})' for n, j in remaining]}")
        time.sleep(POLL_SECONDS)


# ---------- collect ----------

def collect_composition(compo_name: str, solver_path: Path) -> dict | None:
    """Parse result_full logs into a PAR2 summary."""
    pipeline = EvaluationPipeline()  # default = full eval (timeout 5000, penalty 10000)
    result_dir = solver_path / "result_full"
    if not result_dir.is_dir():
        logger.warning(f"[{compo_name}] result_full missing")
        return None

    logs = sorted(p for p in result_dir.iterdir() if p.name.endswith(".solving.log"))
    if not logs:
        logger.warning(f"[{compo_name}] no .solving.log files in {result_dir}")
        return None

    times: dict[str, float] = {}
    for log in logs:
        instance = log.name.rsplit(".cnf.solving.log", 1)[0]
        t = pipeline.parse_solving_time(str(log))
        if t is not None:
            times[instance] = t

    if not times:
        logger.warning(f"[{compo_name}] all log parses failed")
        return None

    penalty = pipeline.par2_penalty
    solved = sum(1 for t in times.values() if t < penalty)
    par2 = sum(times.values()) / len(times)

    out = {
        "instances": len(times),
        "solved": solved,
        "timeouts": len(times) - solved,
        "par2": round(par2, 4),
    }
    (result_dir / "solving_times.json").write_text(json.dumps(times, indent=2))
    logger.info(f"[{compo_name}] PAR2={out['par2']}  solved={solved}/{len(times)}")
    return out


# ---------- report ----------

def write_report(doc: dict, state: dict) -> None:
    rows = []
    for compo in doc["compositions"]:
        name = compo["name"]
        info = state.get(name, {})
        result = info.get("result") or {}
        rows.append({
            "name": name,
            "restart_mab_rank": compo["restart_mab_rank"],
            "kissat_bump_score_increment_rank": compo["kissat_bump_score_increment_rank"],
            "build_ok": info.get("build_ok"),
            "job_id": info.get("job_id"),
            "instances": result.get("instances"),
            "solved": result.get("solved"),
            "timeouts": result.get("timeouts"),
            "par2": result.get("par2"),
        })

    json_path = COMPO_OUTPUT_DIR / "full_eval_results.json"
    json_path.write_text(json.dumps({
        "baseline_full_par2": doc["base_solver"]["baseline_full_par2"],
        "individual_full_pars": {
            "restart_mab_top": [c["full_par2"] for c in doc["functions"]["restart_mab"]["candidates"]],
            "kissat_bump_score_increment_top": [c.get("full_par2") for c in doc["functions"]["kissat_bump_score_increment"]["candidates"]],
        },
        "compositions": rows,
    }, indent=2))

    # Pretty .txt table sorted by PAR2 (None last)
    sorted_rows = sorted(rows, key=lambda r: r["par2"] if r["par2"] is not None else float("inf"))
    lines = []
    baseline = doc["base_solver"]["baseline_full_par2"]
    lines.append(f"Composition full-eval results (baseline AE_kissat2025_MAB = {baseline})")
    lines.append("=" * 80)
    lines.append(f"{'Name':<14} {'R':>2} {'B':>2} {'PAR2':>10} {'Solved':>10} {'TO':>5} {'JobID':>10}")
    lines.append("-" * 80)
    for r in sorted_rows:
        par2_s = f"{r['par2']:.2f}" if r["par2"] is not None else "—"
        solved_s = f"{r['solved']}/{r['instances']}" if r["solved"] is not None else "—"
        to_s = str(r["timeouts"]) if r["timeouts"] is not None else "—"
        jid_s = str(r["job_id"]) if r["job_id"] else "—"
        lines.append(
            f"{r['name']:<14} {r['restart_mab_rank']:>2} {r['kissat_bump_score_increment_rank']:>2} "
            f"{par2_s:>10} {solved_s:>10} {to_s:>5} {jid_s:>10}"
        )
    txt_path = COMPO_OUTPUT_DIR / "full_eval_results.txt"
    txt_path.write_text("\n".join(lines) + "\n")
    logger.info(f"Wrote {json_path} and {txt_path}")
    print("\n".join(lines))


# ---------- helpers ----------

def find_candidate(doc: dict, fn_name: str, rank: int) -> dict:
    for c in doc["functions"][fn_name]["candidates"]:
        if c["rank"] == rank:
            return c
    raise KeyError(f"no rank {rank} candidate for {fn_name}")


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compositions-file", default=str(DEFAULT_COMPO_FILE))
    parser.add_argument("--build-only", action="store_true", help="Stop after building solvers")
    parser.add_argument("--skip-build", action="store_true", help="Skip build step (assume already built)")
    parser.add_argument("--no-wait", action="store_true", help="Submit but don't wait for jobs / collect")
    parser.add_argument("--collect-only", action="store_true", help="Skip build/submit, only collect + report")
    args = parser.parse_args()

    doc = json.loads(Path(args.compositions_file).read_text())
    state = load_state()

    # 1. Build
    if not args.skip_build and not args.collect_only:
        for compo in doc["compositions"]:
            name = compo["name"]
            entry = state.setdefault(name, {})
            sp = build_composition(compo, doc)
            entry["build_ok"] = sp is not None
            entry["solver_path"] = str(sp.relative_to(REPO_ROOT)) if sp else None
            save_state(state)
        if args.build_only:
            return

    # 2. Submit any built compositions that haven't been submitted yet
    if not args.collect_only:
        for compo in doc["compositions"]:
            name = compo["name"]
            entry = state.get(name, {})
            if not entry.get("build_ok") or entry.get("job_id"):
                continue
            sp = COMPO_SOLVER_ROOT / name
            jid = submit_full_eval(name, sp)
            entry["job_id"] = jid
            save_state(state)

    # 3. Wait
    if args.no_wait:
        logger.info("--no-wait: skipping queue poll. Re-run with --collect-only after jobs finish.")
        write_report(doc, state)
        return

    if not args.collect_only:
        wait_for_jobs(state)

    # 4. Collect
    for compo in doc["compositions"]:
        name = compo["name"]
        entry = state.get(name, {})
        if not entry.get("build_ok"):
            continue
        sp = COMPO_SOLVER_ROOT / name
        result = collect_composition(name, sp)
        if result is not None:
            entry["result"] = result
            entry["collected"] = True
            save_state(state)

    # 5. Report
    write_report(doc, state)


if __name__ == "__main__":
    main()
