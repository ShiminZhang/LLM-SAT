#!/bin/bash
# run_bridge.sh — Bridge: GE Offspring → New Leader Pool
#
# Takes the top N offspring from a genetic evolution run and converts them
# into clean leaders (parent_id=None) under a new tag. These leaders can
# then be fed into Loop A (run_loop_a.sh) for further refinement.
#
# Usage:
#   ./run_bridge.sh <ge_output_tag> <target_base_tag> [top_n]
#
# Examples:
#   # Promote top 10 offspring from GE, then refine with 3 Loop A iterations
#   ./run_bridge.sh gemini_trial5_gen1_v1_iter1 gemini_trial5_ge1 10
#   ./run_loop_a.sh gemini_trial5_ge1 3 gemini_trial5_ge1_iter0
#
# Arguments:
#   ge_output_tag    - GE output tag containing evaluated offspring
#   target_base_tag  - Base tag for the new leader pool (leaders go to {target}_iter0)
#   top_n            - (Optional) Number of top offspring to promote (default: 10)

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <ge_output_tag> <target_base_tag> [top_n]"
    exit 1
fi

GE_TAG="$1"
TARGET_BASE="$2"
TOP_N="${3:-10}"
TARGET_TAG="${TARGET_BASE}_iter0"

echo "============================================"
echo "Bridge: GE Offspring -> Leaders"
echo "  GE output tag:  $GE_TAG"
echo "  Target tag:      $TARGET_TAG"
echo "  Top N:           $TOP_N"
echo "============================================"

python src/llmsat/pipelines/genetic_evolution.py \
    --promote-offspring \
    --source_tag "$GE_TAG" \
    --target_tag "$TARGET_TAG" \
    --top_n_promote "$TOP_N"

echo ""
echo "============================================"
echo "Bridge complete!"
echo "New leaders are in: $TARGET_TAG"
echo ""
echo "Next steps:"
echo "  # Run Loop A to refine these leaders"
echo "  ./run_loop_a.sh $TARGET_BASE <n_iterations> $TARGET_TAG"
echo ""
echo "  # Or run genetic evolution on them"
echo "  python src/llmsat/pipelines/genetic_evolution.py \\"
echo "      --generation_tag $TARGET_TAG \\"
echo "      --output_tag ${TARGET_BASE}_gen2 \\"
echo "      --evaluate ..."
echo "============================================"
