#!/bin/bash
# Script to download SAT Competition 2025 benchmarks from track_main_2025.uri
# Downloads .cnf.xz files, extracts them, and removes the compressed versions

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# LLM-SAT root is one level up from scripts/
LLM_SAT_ROOT="$(dirname "$SCRIPT_DIR")"

URI_FILE="$LLM_SAT_ROOT/track_main_2025.uri"
OUTPUT_DIR="$LLM_SAT_ROOT/data/benchmarks/satcomp2025"

# Check that the URI file exists
if [ ! -f "$URI_FILE" ]; then
    echo "Error: $URI_FILE not found"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Downloading SAT Competition 2025 benchmarks..."
echo "URI file: $URI_FILE"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Download files into the output directory
cd "$OUTPUT_DIR"
wget --content-disposition -i "$URI_FILE"

echo ""
echo "Extracting .xz files..."

# Decompress all .xz files and remove the compressed versions
for xz_file in *.cnf.xz; do
    if [ -f "$xz_file" ]; then
        echo "Extracting: $xz_file"
        xz -d "$xz_file"
    fi
done

echo ""
echo "Done! CNF files are now in: $OUTPUT_DIR"
echo "Total files: $(ls -1 *.cnf 2>/dev/null | wc -l) CNF files"
