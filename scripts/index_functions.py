#!/usr/bin/env python3
"""
Index target functions in the base Kissat solver to generate a function registry.

This script locates function definitions in C source files and records their
file paths and line ranges for use by the code injection pipeline.

Usage:
    # Index a single function
    python scripts/index_functions.py kissat_restarting

    # Index multiple functions
    python scripts/index_functions.py kissat_restarting kissat_decide kissat_analyze

    # Specify custom solver path and output
    python scripts/index_functions.py kissat_restarting \
        --solver solvers/base \
        --output solvers/base/function_registry.yaml

    # Append to existing registry (default behavior if registry exists)
    python scripts/index_functions.py kissat_decide

    # Overwrite existing registry
    python scripts/index_functions.py kissat_restarting --overwrite
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import yaml


def find_function_bounds(source: str, func_name: str) -> tuple[int, int, str] | None:
    """
    Find a function definition by name using regex + brace matching.

    Returns (start_line, end_line, signature) or None if not found.
    Line numbers are 1-indexed.
    """
    lines = source.splitlines(keepends=True)

    # Pattern to detect start of a function definition
    # Handles: return types, pointers, static, inline, etc.
    # Looks for: <type> <func_name>(<params>) {
    func_pattern = rf'\b{re.escape(func_name)}\s*\('

    for i, line in enumerate(lines):
        if not re.search(func_pattern, line):
            continue

        # Include possible return-type lines above the function name, e.g.:
        #   static char
        #   rephase_conflict (kissat * solver)
        sig_start = i
        while sig_start > 0:
            prev = lines[sig_start - 1].strip()
            if not prev:
                break
            if prev.startswith('#'):
                break
            if prev.endswith(';') or prev.endswith('{') or prev.endswith('}'):
                break
            sig_start -= 1

        # Accumulate lines to handle multi-line signatures
        accumulated = ""
        signature_end_line = i

        for j in range(sig_start, min(i + 10, len(lines))):  # Look ahead up to 10 lines
            accumulated += lines[j]
            signature_end_line = j

            # Check if we've reached the opening brace
            if '{' in accumulated:
                # Verify this looks like a function definition (not a call or declaration)
                # Should have return type before function name
                before_func = accumulated.split(func_name)[0]
                if not re.search(r'\b(void|bool|int|unsigned|char|double|float|struct|enum|\w+_t)\s*\*?\s*$', before_func.strip()):
                    # Might be a function call, not definition
                    if not re.search(r'^\s*(static\s+)?(inline\s+)?(const\s+)?[\w\s\*]+\s+' + re.escape(func_name), accumulated):
                        break

                # Extract signature (everything before the opening brace)
                sig_match = re.search(
                    rf'((?:static\s+)?(?:inline\s+)?[\w\s\*]+\b{re.escape(func_name)}\s*\([^)]*\))',
                    accumulated,
                    re.DOTALL
                )
                signature = sig_match.group(1).strip() if sig_match else f"{func_name}(...)"
                # Clean up whitespace in signature
                signature = re.sub(r'\s+', ' ', signature)

                # Find the closing brace using brace matching
                start_line = sig_start + 1  # 1-indexed
                brace_count = 0
                started = False

                for k in range(i, len(lines)):
                    for char in lines[k]:
                        if char == '{':
                            brace_count += 1
                            started = True
                        elif char == '}':
                            brace_count -= 1
                            if started and brace_count == 0:
                                end_line = k + 1  # 1-indexed
                                return (start_line, end_line, signature)

                # Opening brace found but no matching close - malformed
                return None

    return None


def find_function_file(solver_path: Path, func_name: str) -> tuple[Path, str] | None:
    """
    Search src/ directory for a file containing the function definition.

    Returns (file_path, source_content) or None if not found.
    """
    src_dir = solver_path / "src"

    if not src_dir.exists():
        print(f"Error: src directory not found at {src_dir}", file=sys.stderr)
        return None

    # First pass: quick grep for function name
    candidates = []
    for c_file in src_dir.glob("*.c"):
        content = c_file.read_text()
        if func_name in content:
            candidates.append((c_file, content))

    # Second pass: verify it's actually a definition
    for c_file, content in candidates:
        result = find_function_bounds(content, func_name)
        if result:
            return (c_file, content)

    return None


def load_registry(registry_path: Path) -> dict:
    """Load existing registry or return empty structure."""
    if registry_path.exists():
        with open(registry_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def save_registry(registry: dict, registry_path: Path) -> None:
    """Save registry to YAML file."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "w") as f:
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False)


def index_function(solver_path: Path, func_name: str) -> dict | None:
    """
    Index a single function.

    Returns dict with function info or None if not found.
    """
    result = find_function_file(solver_path, func_name)
    if result is None:
        return None

    file_path, content = result
    bounds = find_function_bounds(content, func_name)
    if bounds is None:
        return None

    start_line, end_line, signature = bounds
    rel_path = file_path.relative_to(solver_path)

    return {
        "file": str(rel_path),
        "start_line": start_line,
        "end_line": end_line,
        "signature": signature,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Index target functions in the base Kissat solver.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "functions",
        nargs="+",
        help="Function names to index (e.g., kissat_restarting kissat_decide)",
    )
    parser.add_argument(
        "--solver",
        default="solvers/base",
        help="Path to base solver (default: solvers/base)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output registry path (default: <solver>/function_registry.yaml)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing registry instead of appending",
    )

    args = parser.parse_args(argv)

    solver_path = Path(args.solver)
    if not solver_path.exists():
        print(f"Error: Solver path does not exist: {solver_path}", file=sys.stderr)
        return 1

    registry_path = Path(args.output) if args.output else solver_path / "function_registry.yaml"

    # Load or initialize registry
    if args.overwrite:
        registry = {
            "version": "kissat-base",
            "generated": str(date.today()),
            "functions": {},
        }
    else:
        registry = load_registry(registry_path)
        if not registry:
            registry = {
                "version": "kissat-base",
                "generated": str(date.today()),
                "functions": {},
            }
        if "functions" not in registry:
            registry["functions"] = {}

    # Index each function
    success_count = 0
    for func_name in args.functions:
        print(f"Indexing {func_name}...", end=" ")

        info = index_function(solver_path, func_name)
        if info is None:
            print("NOT FOUND")
            continue

        registry["functions"][func_name] = info
        print(f"found in {info['file']}: lines {info['start_line']}-{info['end_line']}")
        success_count += 1

    # Update timestamp
    registry["generated"] = str(date.today())

    # Save registry
    save_registry(registry, registry_path)
    print(f"\nRegistry saved to {registry_path}")
    print(f"Indexed {success_count}/{len(args.functions)} functions")

    return 0 if success_count == len(args.functions) else 1


if __name__ == "__main__":
    sys.exit(main())
