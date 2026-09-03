#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
FAMILY="${1:-edge-matching}"
MANIFEST="$REPO_ROOT/experiment/benchmarks/$FAMILY.tsv"
DESTINATION="$REPO_ROOT/data/benchmarks/formula-families/$FAMILY"

if [ ! -f "$MANIFEST" ]; then
  echo "No comparison benchmark manifest for family: $FAMILY" >&2
  exit 2
fi
command -v curl >/dev/null || { echo "curl is required" >&2; exit 3; }
command -v xz >/dev/null || { echo "xz is required" >&2; exit 3; }
command -v sha256sum >/dev/null || { echo "sha256sum is required" >&2; exit 3; }

mkdir -p "$DESTINATION"
while IFS=$'\t' read -r filename expected_sha256 url; do
  [ -n "$filename" ] || continue
  case "$filename" in \#*) continue ;; esac

  output="$DESTINATION/$filename"
  if [ -f "$output" ] && printf '%s  %s\n' "$expected_sha256" "$output" | sha256sum -c - >/dev/null; then
    echo "verified $filename"
    continue
  fi

  archive=$(mktemp "${TMPDIR:-/tmp}/llmsat-benchmark.XXXXXX.xz")
  extracted=$(mktemp "${TMPDIR:-/tmp}/llmsat-benchmark.XXXXXX.cnf")
  cleanup() { rm -f -- "$archive" "$extracted"; }
  trap cleanup EXIT
  curl --fail --location --retry 3 --output "$archive" "$url"
  xz --decompress --stdout "$archive" > "$extracted"
  printf '%s  %s\n' "$expected_sha256" "$extracted" | sha256sum -c - >/dev/null
  mv -- "$extracted" "$output"
  rm -f -- "$archive"
  trap - EXIT
  echo "downloaded and verified $filename"
done < "$MANIFEST"
