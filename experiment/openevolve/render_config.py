#!/usr/bin/env python3
"""Resolve comparison-only placeholders and preserve the exact run config."""

from __future__ import annotations

import argparse
from pathlib import Path


TOKENS = {
    "model": "${COMPARISON_MODEL}",
    "reasoning_effort": "${COMPARISON_REASONING_EFFORT}",
    "parallel_evaluations": "${COMPARISON_MAX_CANDIDATE_JOBS}",
}


def render(
    template: str,
    *,
    model: str,
    reasoning_effort: str,
    parallel_evaluations: int,
) -> str:
    if parallel_evaluations < 1:
        raise ValueError("parallel_evaluations must be positive")
    values = {
        "model": model,
        "reasoning_effort": reasoning_effort,
        "parallel_evaluations": str(parallel_evaluations),
    }
    rendered = template
    for name, token in TOKENS.items():
        if token not in rendered:
            raise ValueError(f"missing required placeholder {token}")
        rendered = rendered.replace(token, values[name])
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning-effort", required=True)
    parser.add_argument("--parallel-evaluations", required=True, type=int)
    args = parser.parse_args()

    template = args.template.read_text(encoding="utf-8")
    resolved = render(
        template,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        parallel_evaluations=args.parallel_evaluations,
    )
    args.output.write_text(resolved, encoding="utf-8")


if __name__ == "__main__":
    main()
