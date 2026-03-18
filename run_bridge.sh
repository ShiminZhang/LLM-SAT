#!/bin/bash
# run_bridge.sh — Bridge: GE Offspring → New Leader Pool → Next GE Run (Phase 1)
#
# Promotes top-N offspring from a GE run as new leaders, then immediately
# kicks off the next GE run (LLM stages + SLURM submission). Scans ALL
# _iter* variants of ge_output_tag and selects top-N by lowest PAR2.
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
#   output_tag  - Tag for the next GE run's offspring (Phase 1: submit_only)
#
# The intermediate leader pool is auto-derived as: {input_tag}_promoted_iter0
#
# Hyperparameter overrides (env vars):
#   TOP_K            - LLM combination proposals per minibatch (default: 5)
#   MINIBATCH_SIZE   - Leaders per LLM proposal call (default: 10)
#   RUBRIC_MIN       - Minimum proposal score to proceed (default: 6.0)
#   RUBRIC_KEEP_TOP_N - Keep top-N proposals after score filter (default: 10)
#   MODEL            - LLM model (default: gemini-3-flash-preview)
# Note: PAR2_KEEP_TOP_N is used in run_ge_collect.sh (Phase 2), not here.

set -euo pipefail

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
TOP_K="${TOP_K:-5}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-10}"
RUBRIC_MIN="${RUBRIC_MIN:-6.0}"
RUBRIC_KEEP_TOP_N="${RUBRIC_KEEP_TOP_N:-10}"
_CFG_MODEL=$(grep '^default_model:' path_config.yaml 2>/dev/null | sed 's/^default_model:[[:space:]]*//' | tr -d '"' || true)
MODEL="${MODEL:-${_CFG_MODEL:-gemini-3-flash-preview}}"
SHUFFLE_PASSES="${SHUFFLE_PASSES:-1}"
PAR2_KEEP_TOP_N="${PAR2_KEEP_TOP_N:-7}"
POLL_INTERVAL="${POLL_INTERVAL:-120}"

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
echo "============================================"

# Promote top-N offspring to leaders, then run GE Phase 1 (submit_only)
python "$GE_SCRIPT" \
    --promote-offspring \
    --source_tag "$INPUT_TAG" \
    --target_tag "$TARGET_TAG" \
    --output_tag "$OUTPUT_TAG" \
    --evaluate \
    --submit_only \
    --top_k "$TOP_K" \
    --minibatch_size "$MINIBATCH_SIZE" \
    --rubric_min "$RUBRIC_MIN" \
    --rubric_keep_top_n "$RUBRIC_KEEP_TOP_N" \
    --shuffle_passes "$SHUFFLE_PASSES" \
    --model "$MODEL" \
    $NERSC_FLAG

# Poll SLURM until all user jobs finish
echo ""
echo "[Polling] Waiting for all SLURM jobs to complete..."
echo "  (Note: polling all jobs for user $USER)"
while true; do
    RUNNING=$(squeue -u "$USER" -h 2>/dev/null | wc -l)
    if [ "$RUNNING" -eq 0 ]; then
        echo "  All SLURM jobs completed"
        break
    fi
    echo "  $RUNNING jobs still running/pending, waiting ${POLL_INTERVAL}s..."
    sleep "$POLL_INTERVAL"
done

# Collect GE results (absorbed from run_ge_collect.sh)
echo ""
echo "[Collecting] Gathering PAR2 results..."
python "$GE_SCRIPT" \
    --generation_tag "$TARGET_TAG" \
    --output_tag "$OUTPUT_TAG" \
    --collect_results \
    --par2_keep_top_n "$PAR2_KEEP_TOP_N" \
    --model "$MODEL" \
    $NERSC_FLAG

echo ""
echo "============================================"
echo "Bridge complete. Results saved under: outputs/$OUTPUT_TAG/"
echo ""
echo "Next: run Loop A on the promoted leaders:"
echo "  ./run_loop_a.sh $CLUSTER $OUTPUT_TAG <n_iterations> $OUTPUT_TAG"
echo "============================================"
