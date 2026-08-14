#!/bin/bash
# run_bridge.sh — Bridge: GE Offspring → New Leader Pool → Next GE Run
#
# Promotes top-N offspring from a GE run as new leaders, kicks off the next
# GE run (LLM stages + SLURM submission), waits for this run's SLURM jobs,
# then collects PAR2 results and selects the top offspring. Scans ALL
# _iter* variants of input_tag and selects top-N by lowest PAR2.
#
# Usage:
#   ./run_bridge.sh <cc|nersc> <input_tag> <output_tag>
#
# Examples:
#   ./run_bridge.sh nersc gemini_trial5_gen1_v1 gemini_trial5_ge1_gen1
#
# Arguments:
#   cc|nersc    - Cluster to run on
#   input_tag   - Tag to read evaluated offspring from (scans _iter1, _iter2, ... automatically)
#   output_tag  - Tag for the next GE run's offspring
#
# The intermediate leader pool is auto-derived as: {output_tag}_ge
#
# Hyperparameter overrides (env vars):
#   TOP_K            - LLM combination proposals per minibatch (default: 10)
#   MINIBATCH_SIZE   - Leaders per LLM proposal call (default: 10)
#   RUBRIC_MIN       - Minimum proposal score to proceed (default: 6.0)
#   RUBRIC_KEEP_TOP_N - Keep top-N proposals after score filter (default: 50)
#   PAR2_KEEP_TOP_N  - Keep top-N offspring by PAR2 in Step 3 collection (default: 50)
#   MODEL            - LLM model (default: default_model from path_config.yaml)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/loop_common.sh
source "${SCRIPT_DIR}/scripts/lib/loop_common.sh"

if [ $# -lt 3 ]; then
    echo "Usage: $0 <cc|nersc> <input_tag> <output_tag>"
    exit 1
fi

CLUSTER="$1"
INPUT_TAG="$2"
OUTPUT_TAG="$3"

# Auto-derive intermediate leader pool tag from input
TARGET_TAG="${OUTPUT_TAG}_ge" #assume never repetitive

# Hyperparameter defaults (matching LLM-SAT_ run_genetic_evolution.sh)
TOP_K="${TOP_K:-10}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-10}"
RUBRIC_MIN="${RUBRIC_MIN:-6.0}"
RUBRIC_KEEP_TOP_N="${RUBRIC_KEEP_TOP_N:-50}"
MODEL="${MODEL:-$(cfg_default_model)}"
SHUFFLE_PASSES="${SHUFFLE_PASSES:-2}"
PAR2_KEEP_TOP_N="${PAR2_KEEP_TOP_N:-50}"
POLL_INTERVAL="${POLL_INTERVAL:-120}"
N_API_THREADS="${N_API_THREADS:-5}"
N_BUILD_THREADS="${N_BUILD_THREADS:-10}"
QUICK_EVAL="${QUICK_EVAL:-1}"

case "$CLUSTER" in
    cc)
        GE_SCRIPT="src/llmsat/pipelines/genetic_evolution.py"
        NERSC_FLAG=""
        ;;
    nersc)
        GE_SCRIPT="src/llmsat/pipelines/genetic_evolution.py"
        NERSC_FLAG="--nersc"
        ;;
    *)
        echo "ERROR: cluster must be 'cc' or 'nersc', got '$CLUSTER'"
        exit 1
        ;;
esac

echo "============================================"
echo "Bridge: GE Offspring -> Leaders -> Next GE"
echo "  Cluster:          $CLUSTER"
echo "  Input tag:        $INPUT_TAG  (+ all _iter* variants)"
echo "  Leader pool:      $TARGET_TAG  (auto-derived)"
echo "  Output tag:       $OUTPUT_TAG"
echo "  top_k:            $TOP_K"
echo "  minibatch_size:   $MINIBATCH_SIZE"
echo "  rubric_min:       $RUBRIC_MIN"
echo "  rubric_keep_top_n: $RUBRIC_KEEP_TOP_N"
echo "  shuffle_passes:   $SHUFFLE_PASSES"
echo "  model:            $MODEL"
echo "  par2_keep_top_n:  $PAR2_KEEP_TOP_N"
echo "  poll_interval:    ${POLL_INTERVAL}s"
echo "  n_api_threads:    $N_API_THREADS"
echo "  n_build_threads:  $N_BUILD_THREADS"
echo "  Quick eval:       $QUICK_EVAL"
echo "============================================"

# Step 1: Promote top-N offspring to leaders, then run GE Phase 1
#   - promote_offspring_to_leaders: select top-N from INPUT_TAG by PAR2 → register under TARGET_TAG
#   - run_evolution: causal analysis → LLM proposals → crossover → codegen → build → SLURM submit
echo ""
echo "[Step 1] Promoting offspring from ${INPUT_TAG} → ${TARGET_TAG}, then running GE pipeline..."
echo "         (causal analysis, combination proposals, crossover, code generation, SLURM submit)"
python "$GE_SCRIPT" \
    --promote-offspring \
    --source_tag "$INPUT_TAG" \
    --target_tag "$TARGET_TAG" \
    --output_tag "$OUTPUT_TAG" \
    --top_n_promote "$RUBRIC_KEEP_TOP_N" \
    --evaluate \
    --top_k "$TOP_K" \
    --minibatch_size "$MINIBATCH_SIZE" \
    --rubric_min "$RUBRIC_MIN" \
    --rubric_keep_top_n "$RUBRIC_KEEP_TOP_N" \
    --shuffle_passes "$SHUFFLE_PASSES" \
    --model "$MODEL" \
    $NERSC_FLAG \
    $( [ "$QUICK_EVAL" = "1" ] && printf '%s' "--quick_eval" ) \
    --n_api_threads "$N_API_THREADS" \
    --n_build_threads "$N_BUILD_THREADS"
echo "[Step 1] Done."

# Step 2: Poll SLURM until this run's submitted jobs finish. The GE pipeline
# records its SLURM array IDs under outputs/${OUTPUT_TAG}/ (for --evaluate runs
# in evolution_summary.json's slurm_job_ids); poll_slurm_job_ids merges them
# with submitted_job_ids.json/.jsonl so unrelated jobs of $USER are ignored.
echo ""
echo "[Step 2] Waiting for this run's SLURM jobs to complete..."
poll_slurm_job_ids "$OUTPUT_TAG" "$POLL_INTERVAL"

# Step 3: Collect GE results and run PAR2 selection
echo ""
echo "[Step 3] Collecting PAR2 results and selecting top-${PAR2_KEEP_TOP_N} offspring..."
python "$GE_SCRIPT" \
    --generation_tag "$TARGET_TAG" \
    --output_tag "$OUTPUT_TAG" \
    --collect_results \
    --par2_keep_top_n "$PAR2_KEEP_TOP_N" \
    --model "$MODEL" \
    $NERSC_FLAG
echo "[Step 3] Done."

# Step 4: Update combination experience pool (non-fatal)
echo ""
echo "[Step 4] Updating combination experience pool for ${OUTPUT_TAG}..."
echo "         combined_dir=solvers/${OUTPUT_TAG}_iter1  parent_source_dir=solvers/${INPUT_TAG}"
python scripts/update_combination_experience_pool.py \
    --output_tag "$OUTPUT_TAG" \
    --input_tag "$INPUT_TAG" \
    || echo "  [Step 4] WARNING: combination pool update failed (non-fatal)"
echo "[Step 4] Done."

echo ""
echo "============================================"
echo "Bridge complete. Results saved under: outputs/$OUTPUT_TAG/"
echo ""
echo "Next: run Loop A on the promoted leaders:"
echo "  ./run_loop_a.sh $CLUSTER $OUTPUT_TAG <n_iterations>"
echo "============================================"
