#!/bin/bash
# run_ge_collect.sh — Phase 2: collect PAR2 results after SLURM jobs finish
#
# Run this after SLURM jobs submitted by genetic_evolution.py --submit_only
# have completed. Collects PAR2 scores and selects top-k offspring.
#
# Usage:
#   ./run_ge_collect.sh <cc|nersc> <generation_tag> <output_tag> [options]
#
# Examples:
#   ./run_ge_collect.sh nersc gemini_trial5 gemini_trial5_gen1_v1
#
# Arguments:
#   cc|nersc         - Cluster (selects evaluation backend)
#   generation_tag   - Source population tag (same as used in Phase 1)
#   output_tag       - Output tag (same as used in Phase 1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/loop_common.sh
source "${SCRIPT_DIR}/scripts/lib/loop_common.sh"

if [ $# -lt 3 ]; then
    echo "Usage: $0 <cc|nersc> <generation_tag> <output_tag>"
    exit 1
fi

CLUSTER="$1"
GENERATION_TAG="$2"
OUTPUT_TAG="$3"

NERSC_FLAG=""
if [ "$CLUSTER" = "nersc" ]; then
    NERSC_FLAG="--nersc"
elif [ "$CLUSTER" != "cc" ]; then
    echo "ERROR: cluster must be 'cc' or 'nersc', got '$CLUSTER'"
    exit 1
fi

PAR2_KEEP_TOP_N="${PAR2_KEEP_TOP_N:-7}"
MODEL="${MODEL:-$(cfg_default_model)}"

echo "============================================"
echo "GE Phase 2: Collect Results"
echo "  Cluster:        $CLUSTER"
echo "  Generation tag: $GENERATION_TAG"
echo "  Output tag:     $OUTPUT_TAG"
echo "  par2_keep_top_n: $PAR2_KEEP_TOP_N"
echo "============================================"

python src/llmsat/pipelines/genetic_evolution.py \
    --generation_tag "$GENERATION_TAG" \
    --output_tag "$OUTPUT_TAG" \
    --collect_results \
    --par2_keep_top_n "$PAR2_KEEP_TOP_N" \
    --model "$MODEL" \
    $NERSC_FLAG

echo ""
echo "============================================"
echo "Collection complete. Top-tier saved under: outputs/$OUTPUT_TAG/"
echo ""
echo "Next: run loop a to promote top leaders:"
echo "============================================"
