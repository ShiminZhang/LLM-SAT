export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
export DB_PASS="Damn123,"
export OPENAI_API_KEY="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"
# python src/llmsat/pipelines/chatgpt_data_generation.py

# python eval_by_batch.py

module load conda
conda activate /pscratch/sd/j/jsong/conda_env/llmsat

# python src/llmsat/pipelines/genetic_evolution.py \
#     --generation_tag controlled_mutation \
#     --folder outputs/gemini_trial1/batch_batches/9taxatd89wfj9iy1pinp11r1t55bn3d3pvc7 \
#     --top_k 5 \
#     --max_pairs 10 \
#     --par2_keep_top_n 10

python src/llmsat/pipelines/genetic_evolution.py \
    --folder outputs/gemini_trial5 \
    --code_prompt_path data/prompts/coder_prompt.txt \
    --top_k 5 \
    --max_pairs 20 \
    --max_iterations 1 \
    --rubric_keep_top_n 10 \
    --model gpt-4o-mini \
    --evaluate 
    # --skip_causal

    
# python src/map_causal_to_code.py \
#     --folder outputs/diversity_testing/batch_batch_698bb702a31481908fe4274698c9711d \
#     --causal_reports outputs/controlled_mutation_gen1/causal_reports.json \
#     --output /pscratch/sd/j/jsong/LLM-SAT/outputs/controlled_mutation_gen1/mapping.json  # optional, saves to file
