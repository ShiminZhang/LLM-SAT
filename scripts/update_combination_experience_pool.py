#!/usr/bin/env python3
"""Update the combination experience pool after one run_bridge.sh execution.

Reads experience_pool_data_root from path_config.yaml, locates the solver
output directories for the given tags, and calls
ExperiencePoolManager.update("combination", ...).

Called from run_bridge.sh after the --collect_results step:
    python scripts/update_combination_experience_pool.py \
        --output_tag "${OUTPUT_TAG}" \
        --input_tag "${INPUT_TAG}"

Path derivation (mirrors run_bridge.sh logic):
    combined_dir      = solvers/{output_tag}_iter1   (GE offspring, stored as leaders)
    parent_source_dir = solvers/{input_tag}          (original leaders/members)

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
        description="Update the combination experience pool with run_bridge results."
    )
    parser.add_argument(
        "--output_tag",
        required=True,
        help=(
            "Output tag from run_bridge.sh (e.g. mike_kissat_dynamicsat_april7_run4_iter1_gen1). "
            "Used to derive combined_dir = solvers/{output_tag}_iter1."
        ),
    )
    parser.add_argument(
        "--input_tag",
        required=True,
        help=(
            "Input tag passed to run_bridge.sh (e.g. mike_kissat_dynamicsat_april7_run4_iter1). "
            "Used to derive parent_source_dir = solvers/{input_tag}."
        ),
    )
    parser.add_argument(
        "--top_k_good",
        type=int,
        default=3,
        help="Number of top-improving combinations to persist (default: 10)",
    )
    parser.add_argument(
        "--top_k_bad",
        type=int,
        default=3,
        help="Number of top-degrading combinations to persist (default: 10)",
    )
    args = parser.parse_args()

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
            "skipping combination pool update. Run 'python scripts/configure_target.py <func_name>' "
            "to configure it."
        )
        return 0

    # --- Resolve directories ---
    # combined_dir: GE pipeline stores offspring under {output_tag}_iter1/leaders/
    combined_dir = os.path.join("solvers", f"{args.output_tag}_iter1")
    # parent_source_dir: original leaders/members from the input tag
    parent_source_dir = os.path.join("solvers", args.input_tag)

    if not os.path.isdir(combined_dir):
        logger.warning(
            f"[exp_pool] Combined solver directory does not exist: {combined_dir} — "
            "skipping combination pool update"
        )
        return 0

    leaders_dir = os.path.join(combined_dir, "leaders")
    if not os.path.isdir(leaders_dir):
        logger.warning(
            f"[exp_pool] Missing leaders/ in {combined_dir} — "
            "skipping combination pool update"
        )
        return 0

    if not os.path.isdir(parent_source_dir):
        logger.warning(
            f"[exp_pool] Parent source directory does not exist: {parent_source_dir} — "
            "skipping combination pool update"
        )
        return 0

    parent_has_data = os.path.isdir(os.path.join(parent_source_dir, "leaders")) or \
                      os.path.isdir(os.path.join(parent_source_dir, "members"))
    if not parent_has_data:
        logger.warning(
            f"[exp_pool] Missing leaders/ and members/ in {parent_source_dir} — "
            "skipping combination pool update"
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
            f"[exp_pool] Updating combination pool: "
            f"combined_dir={combined_dir!r}, parent_source_dir={parent_source_dir!r}"
        )
        summary = manager.update(
            "combination",
            combined_dir=combined_dir,
            parent_source_dir=parent_source_dir,
            top_k_good=args.top_k_good,
            top_k_bad=args.top_k_bad,
        )

        if summary:
            errors = summary.get("errors", [])
            if errors:
                logger.warning(
                    f"[exp_pool] Combination pool update completed with "
                    f"{len(errors)} non-fatal error(s):"
                )
                for err in errors:
                    logger.warning(f"  [exp_pool]   {err}")
            summary_clean = {k: v for k, v in summary.items() if k != "errors"}
            logger.info(
                f"[exp_pool] Combination pool update summary: "
                f"{json.dumps(summary_clean)}"
            )
        else:
            logger.info("[exp_pool] Combination pool update returned no summary.")

    except Exception as e:
        logger.warning(
            f"[exp_pool] Combination experience pool update failed (non-fatal): "
            f"{type(e).__name__}: {e}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
