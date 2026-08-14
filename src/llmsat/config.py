"""Path configuration loader.

Usage:
    from llmsat.config import BASE_SOLVER_PATH, PYTHON_ACTIVATE_PATH
"""
import os
import yaml
from pathlib import Path


def _find_config():
    """Walk up from cwd to find path_config.yaml."""
    cwd = Path.cwd()
    for d in [cwd] + list(cwd.parents):
        candidate = d / "path_config.yaml"
        if candidate.exists():
            return candidate
        if (d / "src" / "llmsat").is_dir():
            break  # hit project root without finding config
    raise FileNotFoundError(
        "path_config.yaml not found. Copy path_config.template.yaml to path_config.yaml "
        "and fill in your paths."
    )


_cfg = yaml.safe_load(open(_find_config()))

BASE_SOLVER_PATH: str = os.path.expanduser(_cfg["base_solver"])
PYTHON_ACTIVATE_PATH: str = os.path.expanduser(_cfg["python_activate"])
BASELINE_PAR2: float | None = _cfg.get("baseline_par2")  # None if not configured

# Model roles — single source of truth (previously five divergent defaults
# were scattered across helpers/pools). Availability verified 2026-08-14,
# see docs/model_probe_2026-08-14.md.
DEFAULT_MODEL: str = _cfg.get("default_model") or "gemini-3.1-pro-preview"
# Algorithm/mutation generation (leaders + variants).
GENERATION_MODEL: str = _cfg.get("generation_model") or DEFAULT_MODEL
# C code generation. gemini-3.6-flash is the cheaper coding-focused option.
CODER_MODEL: str = _cfg.get("coder_model") or DEFAULT_MODEL
# Memory-bank causal analysis (cheap, high-volume).
ANALYSIS_MODEL: str = _cfg.get("analysis_model") or "gemini-3.5-flash-lite"

EXPERIENCE_POOL_DATA_ROOT: str | None = _cfg.get("experience_pool_data_root", None)
