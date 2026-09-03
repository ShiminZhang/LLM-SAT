#!/usr/bin/env python3
"""Shinka evaluator backed by the shared comparison SAT protocol."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(
    os.environ.get("SAT_REPO_ROOT", Path(__file__).resolve().parents[2])
).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment.common.openevolve_runtime import build_only, evaluate_runtime


PAR2_PENALTY = float(os.environ.get("OE_PAR2_PENALTY", "2400"))


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _shinka_result(
    raw_metrics: dict[str, Any], artifacts: dict[str, Any]
) -> tuple[dict[str, Any], bool, str]:
    stage = str(artifacts.get("stage", "unknown"))
    correct = stage == "evaluation" and not artifacts.get("error")
    error = ""
    if not correct:
        detail = artifacts.get("error") or artifacts.get("build_output") or "unknown error"
        error = f"SAT evaluator failed during {stage}: {detail}"

    metrics = {
        "combined_score": _number(raw_metrics.get("combined_score")),
        "public": {
            "par2": _number(raw_metrics.get("par2"), PAR2_PENALTY),
            "instances": _integer(artifacts.get("instances")),
            "solved": _integer(artifacts.get("solved")),
            "timeouts": _integer(artifacts.get("timeouts")),
            "errors": _integer(artifacts.get("errors")),
            "missing": _integer(artifacts.get("missing")),
        },
        "private": {
            "runtime_only": True,
            "proof_validation": "not_run",
            "target_function": artifacts.get("target_function"),
            "target_source": artifacts.get("target_source"),
            "candidate_hash": artifacts.get("candidate_hash"),
            "protocol_hash": artifacts.get("protocol_hash"),
            "job_ids": artifacts.get("job_ids", []),
            "cache_format_version": artifacts.get("cache_format_version"),
        },
    }
    return metrics, correct, error


def evaluate(program_path: str | Path, results_dir: str | Path) -> bool:
    output_dir = Path(results_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        raw_metrics, artifacts = evaluate_runtime(program_path)
        metrics, correct, error = _shinka_result(raw_metrics, artifacts)
    except Exception as exc:
        metrics = {
            "combined_score": 0.0,
            "public": {"par2": PAR2_PENALTY},
            "private": {"runtime_only": True, "proof_validation": "not_run"},
        }
        correct = False
        error = f"SAT evaluator adapter failed: {exc}"

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "correct.json").write_text(
        json.dumps({"correct": correct, "error": error}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "correct": correct,
                "combined_score": metrics["combined_score"],
                "par2": metrics["public"].get("par2"),
            },
            sort_keys=True,
        )
    )
    return correct


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Compile the candidate without submitting benchmark jobs.",
    )
    args = parser.parse_args()

    if args.build_only:
        outcome = build_only(args.program_path)
        Path(args.results_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.results_dir) / "build_only.json").write_text(
            json.dumps(outcome, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(outcome, indent=2, sort_keys=True))
        return 0 if outcome["success"] else 1

    return 0 if evaluate(args.program_path, args.results_dir) else 1


if __name__ == "__main__":
    raise SystemExit(main())
