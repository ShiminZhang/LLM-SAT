#!/usr/bin/env python3
"""Measure compile-rate for a set of strategies.

This script:
1) Reads strategies JSONL (from scripts/generate_diverse_batch.py)
2) For each strategy, calls the coder prompt template to generate a full
   `kissat_restarting` function in <code> tags
3) Injects into a fresh solver copy (FunctionRegistry/FunctionInjector)
4) Builds the solver and records success/failure

Outputs:
- per_strategy_results.csv
- leader_summary.csv (aggregated compile-rate by leader)

Note:
- This can be expensive if you run many strategies.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from llmsat.code_injection.injector import FunctionInjector
from llmsat.code_injection.registry import FunctionRegistry
from llmsat.utils.chatgpt_helper import get_response_from_chatgpt


TARGET_FUNCTION = "kissat_restarting"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _render_coder_prompt(template: str, algorithm_text: str) -> str:
    # Avoid .format(): template includes literal braces in baseline snippet.
    if "{algorithm}" not in template:
        raise ValueError("Coder template missing '{algorithm}' placeholder")
    return template.replace("{algorithm}", algorithm_text)


@dataclass(frozen=True)
class CompileConfig:
    strategies_path: Path
    base_solver: Path
    registry_path: Path
    code_prompt_template_path: Path
    out_dir: Path
    model: Optional[str]
    system_message: Optional[str]
    temperature: float
    max_strategies: Optional[int]
    only_members: bool


def _compile_solver(solver_path: Path, jobs: int = 8) -> tuple[bool, str]:
    """Build solver with ./configure && make. Returns (ok, build_log_text)."""
    env = os.environ.copy()

    def run(cmd: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd,
            cwd=str(solver_path),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    out = []

    p1 = run(["./configure"])
    out.append("$ ./configure\n" + p1.stdout)
    if p1.returncode != 0:
        return False, "\n".join(out)

    p2 = run(["make", f"-j{jobs}"])
    out.append(f"$ make -j{jobs}\n" + p2.stdout)
    if p2.returncode != 0:
        return False, "\n".join(out)

    kissat_bin = solver_path / "build" / "kissat"
    if not kissat_bin.exists():
        return False, "\n".join(out) + "\nMissing build/kissat"  # sanity

    return True, "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", type=Path, required=True, help="strategies.jsonl")
    ap.add_argument("--base-solver", type=Path, default=Path("solvers/base"))
    ap.add_argument("--registry", type=Path, default=Path("solvers/base/function_registry.yaml"))
    ap.add_argument("--code-template", type=Path, default=Path("data/prompts/kissat_mab_code.txt"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/compile_rate"))
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--system-message", type=str, default=None)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-strategies", type=int, default=None)
    ap.add_argument("--only-members", action="store_true", help="Compile only member strategies")

    args = ap.parse_args()

    cfg = CompileConfig(
        strategies_path=args.strategies,
        base_solver=args.base_solver,
        registry_path=args.registry,
        code_prompt_template_path=args.code_template,
        out_dir=args.out_dir,
        model=args.model,
        system_message=args.system_message,
        temperature=float(args.temperature),
        max_strategies=args.max_strategies,
        only_members=bool(args.only_members),
    )

    rows = _read_jsonl(cfg.strategies_path)
    if cfg.only_members:
        rows = [r for r in rows if r.get("type") == "member"]

    if cfg.max_strategies is not None:
        rows = rows[: int(cfg.max_strategies)]

    if not rows:
        raise SystemExit("No strategies to compile")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    template = cfg.code_prompt_template_path.read_text(encoding="utf-8")

    registry = FunctionRegistry(cfg.registry_path)
    injector = FunctionInjector(registry, cfg.base_solver)

    results: List[Dict[str, Any]] = []

    pbar = tqdm(rows, desc="Compile", total=len(rows))
    for r in pbar:
        sid = str(r.get("id"))
        leader_id = str(r.get("leader_id"))
        stype = str(r.get("type"))
        spec = r.get("spec") or {}
        name = spec.get("name")
        algorithm_text = spec.get("algorithm") or ""

        run_dir = cfg.out_dir / sid
        solver_copy = run_dir / "solver"
        run_dir.mkdir(parents=True, exist_ok=True)

        ok = False
        error: Optional[str] = None
        build_log = ""
        llm_raw = ""

        try:
            # Fresh copy each time
            if solver_copy.exists():
                shutil.rmtree(solver_copy)
            shutil.copytree(cfg.base_solver, solver_copy)

            prompt = _render_coder_prompt(template, algorithm_text)
            llm_raw = get_response_from_chatgpt(
                prompt=prompt,
                system_message=cfg.system_message,
                model=cfg.model,
                temperature=cfg.temperature,
            )

            parsed = injector.parse_llm_output(llm_raw, expected_function=TARGET_FUNCTION)
            injector.replace_function(solver_copy, parsed.function_name, parsed.code)

            ok, build_log = _compile_solver(solver_copy)
        except Exception as exc:
            ok = False
            error = str(exc)

        (run_dir / "llm_raw.txt").write_text(llm_raw, encoding="utf-8")
        (run_dir / "build.log").write_text(build_log, encoding="utf-8")

        results.append(
            {
                "id": sid,
                "type": stype,
                "leader_id": leader_id,
                "name": name,
                "ok": bool(ok),
                "error": error,
                "model": cfg.model or os.environ.get("OPENAI_MODEL"),
                "temperature": cfg.temperature,
                "created_at": _utc_now_iso(),
            }
        )

        pbar.set_postfix({"ok": int(sum(1 for x in results if x["ok"]))})

    pbar.close()

    df = pd.DataFrame(results)
    per_path = cfg.out_dir / "per_strategy_results.csv"
    df.to_csv(per_path, index=False)

    leader_summary = (
        df.groupby("leader_id")["ok"]
        .agg(["count", "sum"])
        .rename(columns={"sum": "compiled"})
        .reset_index()
    )
    leader_summary["compile_rate"] = leader_summary["compiled"] / leader_summary["count"].clip(lower=1)

    leader_path = cfg.out_dir / "leader_summary.csv"
    leader_summary.to_csv(leader_path, index=False)

    print(f"Wrote: {per_path}")
    print(f"Wrote: {leader_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
