#!/usr/bin/env python3
"""Update the mutation experience pool after one iteration of Loop A.

Reads experience_pool_data_root from path_config.yaml, locates the
solver output directory for the given generation tag, and calls
ExperiencePoolManager.update("mutation", ...).

Called from run_loop_a.sh after each "collect results" step:
    python scripts/update_experience_pool.py --generation_tag "${ITER_TAG}"

Non-fatal: all errors are logged as warnings. Exit code is always 0.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add src/ to path so llmsat and experience_pool packages are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def find_path_config() -> Path:
    """Walk up from cwd to find path_config.yaml."""
    cwd = Path.cwd()
    for d in [cwd] + list(cwd.parents):
        candidate = d / "path_config.yaml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "path_config.yaml not found. "
        "Copy path_config.template.yaml to path_config.yaml and fill in your paths."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Update the mutation experience pool with iteration results."
    )
    parser.add_argument(
        "--generation_tag",
        required=True,
        help="Generation tag for the completed iteration (e.g. gemini_trial5_iter1)",
    )
    parser.add_argument(
        "--top_k_good",
        type=int,
        default=10,
        help="Number of top-improving mutations to persist (default: 5)",
    )
    parser.add_argument(
        "--top_k_bad",
        type=int,
        default=10,
        help="Number of top-degrading mutations to persist (default: 5)",
    )
    args = parser.parse_args()

    if os.environ.get("MUTATION_POOL", "1").strip() == "0":
        logger.info("[exp_pool] MUTATION_POOL=0 — skipping mutation pool update")
        return 0

    # --- Locate path_config.yaml ---
    try:
        config_path = find_path_config()
        cfg = yaml.safe_load(config_path.read_text()) or {}
    except FileNotFoundError as e:
        logger.warning(f"[exp_pool] {e}")
        return 0

    data_root = cfg.get("experience_pool_data_root")
    if not data_root:
        logger.info(
            "[exp_pool] experience_pool_data_root not set in path_config.yaml — "
            "skipping pool update. Run 'python scripts/configure_target.py <func_name>' "
            "to configure it."
        )
        return 0

    # --- Resolve input_dir ---
    input_dir = os.path.join("solvers", args.generation_tag)
    if not os.path.isdir(input_dir):
        logger.warning(
            f"[exp_pool] Solver directory does not exist: {input_dir} — "
            "skipping pool update"
        )
        return 0

    leaders_dir = os.path.join(input_dir, "leaders")
    members_dir = os.path.join(input_dir, "members")
    if not os.path.isdir(leaders_dir) or not os.path.isdir(members_dir):
        logger.warning(
            f"[exp_pool] Missing leaders/ or members/ in {input_dir} — "
            "skipping pool update"
        )
        return 0

    # --- Import and run update ---
    try:
        from experience_pool import ExperiencePoolManager
    except ImportError as e:
        logger.warning(f"[exp_pool] Cannot import experience_pool package: {e}")
        return 0

    try:
        logger.info(
            f"[exp_pool] Initializing ExperiencePoolManager "
            f"(data_root={data_root!r})"
        )
        manager = ExperiencePoolManager(data_root=data_root)
        logger.info(
            f"[exp_pool] Updating mutation pool: "
            f"input_dir={input_dir!r}, data_root={data_root!r}"
        )
        summary = manager.update(
            "mutation",
            input_dir=input_dir,
            top_k_good=args.top_k_good,
            top_k_bad=args.top_k_bad,
        )

        if summary:
            errors = summary.get("errors", [])
            if errors:
                logger.warning(
                    f"[exp_pool] Update completed with {len(errors)} non-fatal error(s):"
                )
                for err in errors:
                    logger.warning(f"  [exp_pool]   {err}")
            summary_clean = {k: v for k, v in summary.items() if k != "errors"}
            logger.info(f"[exp_pool] Update summary: {json.dumps(summary_clean)}")
        else:
            logger.info("[exp_pool] Update returned no summary.")

    except Exception as e:
        logger.warning(
            f"[exp_pool] Experience pool update failed (non-fatal): "
            f"{type(e).__name__}: {e}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
