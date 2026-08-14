export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
module load conda
conda activate /pscratch/sd/j/jsong/conda_env/llmsat

# Full eval, trial 1:
# python scripts/evaluate_baseline.py --nersc --seed 0
# python scripts/evaluate_baseline.py --nersc --seed 1 &
# python scripts/evaluate_baseline.py --nersc --seed 2

# Quick eval, trial 2:
# python scripts/evaluate_baseline.py --nersc --quick-eval --seed 2

# Collect trial 1 full eval:
# python scripts/evaluate_baseline.py --collect --nersc --seed 1 &
# python scripts/evaluate_baseline.py --collect --nersc --seed 0 &
# python scripts/evaluate_baseline.py --collect --nersc --seed 2

# Collect trial 2 quick eval:
# python scripts/evaluate_baseline.py --collect --nersc --quick-eval --seed 2


# python src/llmsat/pipelines/evaluation_nersc.py \
# --algorithm_id cee194034a6f78b6552b5d246e855e66f1ff2cd2fe039f1bfc4008d987a25bce \
# --code_id 462fc8f76750f22b957b62158221d53faaddba8c8c543457a1fc802c482d21c \
# --build-only

# Submit seed 1, 2, 3 (full eval):
python evaluate_multi.py -a cee194034a6f78b6552b5d246e855e66f1ff2cd2fe039f1bfc4008d987a25bce \
                         -c 462fc8f76750f22b957b62158221d53faaddba8c8c543457a1fc802c482d21c \
                         --seed 1 --collect &
python evaluate_multi.py -a cee194034a6f78b6552b5d246e855e66f1ff2cd2fe039f1bfc4008d987a25bce \
                         -c 462fc8f76750f22b957b62158221d53faaddba8c8c543457a1fc802c482d21c \
                         --seed 2 --collect & 
python evaluate_multi.py -a cee194034a6f78b6552b5d246e855e66f1ff2cd2fe039f1bfc4008d987a25bce \
                         -c 462fc8f76750f22b957b62158221d53faaddba8c8c543457a1fc802c482d21c \
                         --seed 0 --collect #--quick-eval

# python evaluate_multi.py -a cee194034a6f78b6552b5d246e855e66f1ff2cd2fe039f1bfc4008d987a25bce \
                        #  -c 462fc8f76750f22b957b62158221d53faaddba8c8c543457a1fc802c482d21c 

# Or quick eval:
# python evaluate_multi.py -a cee194034a6f... -c 462fc8f767... --seed 1 --quick-eval

# After jobs finish, collect each:
# python evaluate_multi.py -a cee194034a6f... -c 462fc8f767... --seed 1 --collect
# python evaluate_multi.py -a cee194034a6f... -c 462fc8f767... --seed 2 --collect

# Print PAR2 mean/std across all collected seeds:
# python evaluate_multi.py -a cee194034a6f... -c 462fc8f767... --summary
