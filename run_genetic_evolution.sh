#!/bin/bash
# Genetic Evolution Pipeline Runner
#
# Prerequisites:
#   1. Run gemini_data_generation.py to create leaders + members
#   2. Run evaluation.py --run_all to build & evaluate all solvers
#   3. Run evaluation.py --collect_all_results after SLURM jobs complete
#   4. Run evaluation.py --promote-leaders to promote best team members
#   5. THEN run this script to evolve the promoted leaders
#
# See docs/pipeline_workflow.txt for the full pipeline sequence.

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
export DB_PASS=""
export OPENAI_API_KEY=""
export GOOGLE_API_KEY=""

GENERATION_TAG="gemini_trial5"  # Must match the tag used in data generation + evaluation

# Iter 1 — fresh start, loads promoted leaders from DB
python src/llmsat/pipelines/genetic_evolution.py \
    --generation_tag "$GENERATION_TAG" \
    --output_tag "${GENERATION_TAG}_gen1_v1" \
    --code_prompt_path data/prompts/coder_prompt.txt \
    --evaluate \
    --top_k 5 \
    --minibatch_size 10 \
    --rubric_min 6.0 \
    --rubric_keep_top_n 10 \
    --par2_keep_top_n 7 \
    --model gemini-3-flash-preview
