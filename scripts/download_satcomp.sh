#!/bin/bash
# Download SAT Competition main-track benchmarks for a given year.
# Generalizes scripts/download_satcomp2025.sh (same naming/decompression
# conventions: wget --content-disposition names files <md5>-<origname>.cnf.xz,
# which decompress to <md5>-<origname>.cnf).
#
# Usage:
#   scripts/download_satcomp.sh <year> [--sample N]
#
#   <year>       e.g. 2024, 2025, 2026. Reads track_main_<year>.uri from the
#                repo root or data/benchmarks/ (one benchmark-database.de URL
#                per line) and downloads into data/benchmarks/satcomp<year>/.
#   --sample N   Only download the first N instances of the list.
#
# Resumable: an instance is skipped if a decompressed <md5>-*.cnf for its
# hash already exists in the output directory. Partial/compressed leftovers
# are re-downloaded cleanly (each download happens in a temp dir and is only
# moved into place after successful decompression and a DIMACS header check).

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# LLM-SAT root is one level up from scripts/
LLM_SAT_ROOT="$(dirname "$SCRIPT_DIR")"

usage() {
    echo "Usage: $0 <year> [--sample N]" >&2
    exit 1
}

YEAR="${1:-}"
[[ "$YEAR" =~ ^[0-9]{4}$ ]] || usage
shift

SAMPLE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --sample)
            SAMPLE="${2:-}"
            [[ "$SAMPLE" =~ ^[0-9]+$ ]] || usage
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

# Locate the URI file: repo root first (2025 legacy location), then data/benchmarks/
URI_FILE=""
for cand in "$LLM_SAT_ROOT/track_main_${YEAR}.uri" \
            "$LLM_SAT_ROOT/data/benchmarks/track_main_${YEAR}.uri"; do
    if [ -f "$cand" ]; then
        URI_FILE="$cand"
        break
    fi
done
if [ -z "$URI_FILE" ]; then
    echo "Error: track_main_${YEAR}.uri not found in $LLM_SAT_ROOT/ or $LLM_SAT_ROOT/data/benchmarks/" >&2
    exit 1
fi

OUTPUT_DIR="$LLM_SAT_ROOT/data/benchmarks/satcomp${YEAR}"
mkdir -p "$OUTPUT_DIR"

# Read URLs; tolerant of missing trailing newline, blank lines, and comments.
mapfile -t URLS < <(grep -Eo 'https?://[^[:space:]]+' "$URI_FILE")
TOTAL=${#URLS[@]}
if [ "$TOTAL" -eq 0 ]; then
    echo "Error: no URLs found in $URI_FILE" >&2
    exit 1
fi
if [ "$SAMPLE" -gt 0 ] && [ "$SAMPLE" -lt "$TOTAL" ]; then
    URLS=("${URLS[@]:0:$SAMPLE}")
fi
COUNT=${#URLS[@]}

echo "Downloading SAT Competition $YEAR benchmarks..."
echo "URI file:         $URI_FILE ($TOTAL instances, downloading $COUNT)"
echo "Output directory: $OUTPUT_DIR"
echo ""

# First non-comment, non-blank line must be a DIMACS 'p cnf' header.
check_dimacs() {
    awk '/^[[:space:]]*c/ { next } NF == 0 { next } { exit ($1 == "p" && $2 == "cnf") ? 0 : 1 }' "$1"
}

skipped=0
downloaded=0
failed=0
failed_hashes=()

for i in "${!URLS[@]}"; do
    url="${URLS[$i]}"
    hash="${url##*/}"
    idx=$((i + 1))

    # Already have the decompressed CNF for this hash -> skip (resumability).
    if compgen -G "$OUTPUT_DIR/${hash}-*.cnf" > /dev/null || [ -f "$OUTPUT_DIR/${hash}.cnf" ]; then
        echo "[$idx/$COUNT] $hash: already present, skipping"
        skipped=$((skipped + 1))
        continue
    fi

    tmpdir="$(mktemp -d "$OUTPUT_DIR/.download.${hash}.XXXXXX")"
    echo "[$idx/$COUNT] $hash: downloading..."
    if ! (cd "$tmpdir" && wget -q --content-disposition "$url"); then
        echo "[$idx/$COUNT] $hash: DOWNLOAD FAILED" >&2
        rm -rf "$tmpdir"
        failed=$((failed + 1))
        failed_hashes+=("$hash")
        continue
    fi

    # Exactly one file lands in tmpdir, named via Content-Disposition
    # (e.g. <md5>-<origname>.cnf.xz). Decompress by extension.
    file="$(find "$tmpdir" -maxdepth 1 -type f | head -n 1)"
    ok=1
    case "$file" in
        *.xz)   xz -d "$file"       || ok=0 ;;
        *.lzma) xz -d "$file"       || ok=0 ;;
        *.bz2)  bzip2 -d "$file"    || ok=0 ;;
        *.gz)   gzip -d "$file"     || ok=0 ;;
        *.cnf)  : ;;  # already plain
        *)      echo "[$idx/$COUNT] $hash: unknown extension: $(basename "$file")" >&2; ok=0 ;;
    esac

    if [ "$ok" -eq 1 ]; then
        cnf="$(find "$tmpdir" -maxdepth 1 -type f | head -n 1)"
        if check_dimacs "$cnf"; then
            mv "$cnf" "$OUTPUT_DIR/"
            downloaded=$((downloaded + 1))
        else
            echo "[$idx/$COUNT] $hash: INVALID DIMACS header in $(basename "$cnf")" >&2
            ok=0
        fi
    fi

    if [ "$ok" -eq 0 ]; then
        failed=$((failed + 1))
        failed_hashes+=("$hash")
    fi
    rm -rf "$tmpdir"
done

echo ""
echo "Done. Downloaded: $downloaded, skipped (already present): $skipped, failed: $failed"
echo "CNF files in $OUTPUT_DIR: $(ls -1 "$OUTPUT_DIR"/*.cnf 2>/dev/null | wc -l)"
if [ "$failed" -gt 0 ]; then
    echo "Failed hashes:" >&2
    printf '  %s\n' "${failed_hashes[@]}" >&2
    exit 1
fi
