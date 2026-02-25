export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
export DB_PASS="Damn123,"
export OPENAI_API_KEY=""
export GOOGLE_API_KEY=""


# # Iter 1 — fresh start
python src/llmsat/pipelines/genetic_evolution.py \
    --folder outputs/gemini_trial5 \
    --output_tag gemini_trial5_gen1_v1 \
    --code_prompt_path data/prompts/coder_prompt.txt \
    --evaluate \
    --top_k 5 \
    --minibatch_size 10 \
    --rubric_min 6.0 \
    --rubric_keep_top_n 10 \
    --par2_keep_top_n 7 \
    --model gemini-3-flash-preview

# Iter 2 — carries forward improvements after evaluation from slurm finished
python src/llmsat/pipelines/genetic_evolution.py \
    --folder outputs/gemini_trial5 \
    --output_tag gemini_trial5_gen1_v2 \
    --prev_output_tag gemini_trial5_gen1_v1 \
    --code_prompt_path data/prompts/coder_prompt.txt \
    --evaluate \
    --top_k 5 \
    --minibatch_size 10 \
    --rubric_min 6.0 \
    --rubric_keep_top_n 10 \
    --par2_keep_top_n 7 \
    --model gemini-3-flash-preview