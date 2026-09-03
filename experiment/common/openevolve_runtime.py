"""Reuse the established OpenEvolve SAT runtime evaluator from other runners.

The OpenEvolve evaluator is the single source of truth for candidate parsing,
Kissat injection/building, SLURM submission, PAR2 scoring, and the durable
candidate cache.  Keeping this adapter thin prevents the Shinka comparison
from silently drifting to a different benchmark protocol.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any


def repo_root() -> Path:
    default = Path(__file__).resolve().parents[2]
    return Path(os.environ.get("SAT_REPO_ROOT", default)).resolve()


@lru_cache(maxsize=1)
def runtime_module() -> ModuleType:
    """Load the existing evaluator without requiring experiment packages."""
    root = repo_root()
    evaluator_path = root / "experiment" / "openevolve" / "evaluator.py"
    if not evaluator_path.is_file():
        raise FileNotFoundError(f"OpenEvolve evaluator not found: {evaluator_path}")

    os.environ.setdefault("OE_REPO_ROOT", str(root))
    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)

    module_name = "llm_sat_shared_openevolve_evaluator"
    spec = importlib.util.spec_from_file_location(module_name, evaluator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load evaluator module from {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def evaluate_runtime(program_path: str | Path) -> tuple[dict[str, float], dict[str, Any]]:
    """Evaluate a candidate with the shared comparison runtime protocol."""
    result = runtime_module().evaluate(str(Path(program_path).resolve()))
    return dict(result.metrics), dict(result.artifacts)


def build_only(program_path: str | Path) -> dict[str, Any]:
    """Parse, inject, and compile a candidate without submitting SLURM work."""
    evaluator = runtime_module()
    candidate = evaluator._extract_candidate(str(Path(program_path).resolve()))
    digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
    names = evaluator._benchmark_files()
    protocol = evaluator._protocol_fingerprint(names)
    directory = evaluator.WORK_ROOT / protocol / digest
    directory.mkdir(parents=True, exist_ok=True)
    solver, build_output = evaluator._build_solver(candidate, directory)
    return {
        "success": solver is not None,
        "work_dir": str(directory),
        "candidate_hash": digest,
        "protocol_hash": protocol,
        "build_output": evaluator._tail(build_output),
    }
