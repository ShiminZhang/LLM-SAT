#!/usr/bin/env python3
"""CLI wrapper for chatgpt_data_generation.generate_team_data.

This avoids editing Python files to change generation parameters.

Writes batch outputs under:
  outputs/<generation_tag>/batch_<leader_batch_id>/
"""

from __future__ import annotations

import argparse

from llmsat.pipelines.chatgpt_data_generation import generate_team_data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generation-tag", required=True, help="Output tag under outputs/")
    ap.add_argument(
        "--designer-prompt",
        default="data/prompts/ae_prompt_restart.txt",
        help="Leader prompt path",
    )
    ap.add_argument(
        "--variant-prompt",
        default="data/prompts/variant_prompt.txt",
        help="Member prompt path",
    )
    ap.add_argument(
        "--coder-prompt",
        default="data/prompts/coder_prompt.txt",
        help="Coder prompt path",
    )
    ap.add_argument("--leaders", type=int, default=10)
    ap.add_argument("--members-per-leader", type=int, default=5)
    ap.add_argument(
        "--model",
        default=None,
        help="OpenAI model (if not set, OPENAI_MODEL env var is used)",
    )

    args = ap.parse_args()

    # generate_team_data uses create_batch_input_file which now respects the model argument.
    generate_team_data(
        designer_prompt_path=str(args.designer_prompt),
        variant_prompt_path=str(args.variant_prompt),
        code_prompt_template_path=str(args.coder_prompt),
        generation_tag=str(args.generation_tag),
        n_leaders=int(args.leaders),
        m_variants_per_leader=int(args.members_per_leader),
        model=args.model or None,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
