#!/usr/bin/env python3
"""Remove copied Kissat build trees after an LLM-SAT generation is collected."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


CODE_TREE = re.compile(r"code_[0-9a-f]{64}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag-prefix", required=True)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    solvers = root / "solvers"
    outputs = root / "outputs"
    removed = 0

    for generation in sorted(solvers.glob(f"{args.tag_prefix}_iter*")):
        # par2_scores.txt is written only after all result logs have been
        # collected into the durable algorithm JSON and breakdown files.
        if not (outputs / generation.name / "par2_scores.txt").is_file():
            continue
        for role in ("leaders", "members"):
            role_dir = generation / role
            if not role_dir.is_dir():
                continue
            for candidate in role_dir.glob("algorithm_*/code_*"):
                if (
                    candidate.is_dir()
                    and candidate.parent.name != "result"
                    and CODE_TREE.fullmatch(candidate.name)
                ):
                    shutil.rmtree(candidate)
                    removed += 1

    if removed:
        print(f"Pruned {removed} completed LLM-SAT solver build trees")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
