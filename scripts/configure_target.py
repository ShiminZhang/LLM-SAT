#!/usr/bin/env python3
"""
Configure LLM-SAT to target a different C function for optimization.

Automates the tedious manual steps of switching the target function:
  1. Index the function in the base solver
  2. Update function_registry.yaml (repo root + base solver copy)
  3. Rewrite leader_prompt_testing.txt with the new target name
  4. Rewrite coder_prompt_testing.txt with updated signature + embedded source

Usage:
    python scripts/configure_target.py restart_mab
    python scripts/configure_target.py kissat_restarting --file src/restart.c
    python scripts/configure_target.py my_func --solver /path/to/solver
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import yaml

# Import from sibling script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from index_functions import (
    find_function_bounds,
    find_function_file,
    index_function,
    load_registry,
    save_registry,
)


def find_repo_root() -> Path:
    """Walk up from this script to find the repo root (.git directory)."""
    d = Path(__file__).resolve().parent
    for d in [d] + list(d.parents):
        if (d / ".git").exists():
            return d
    raise FileNotFoundError("Could not find repo root (.git directory)")


def find_path_config() -> Path:
    """Walk up from cwd to find path_config.yaml (same pattern as llmsat/config.py)."""
    cwd = Path.cwd()
    for d in [cwd] + list(cwd.parents):
        candidate = d / "path_config.yaml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "path_config.yaml not found. Copy path_config.template.yaml to "
        "path_config.yaml and fill in your paths."
    )


def resolve_solver_path(override: str | None) -> Path:
    """Get base solver path from --solver flag or path_config.yaml."""
    if override:
        p = Path(override).expanduser().resolve()
    else:
        import os
        cfg = yaml.safe_load(open(find_path_config()))
        p = Path(os.path.expanduser(cfg["base_solver"])).resolve()
    if not p.exists():
        print(f"Error: Solver path does not exist: {p}", file=sys.stderr)
        sys.exit(1)
    return p


def parse_signature(signature: str) -> tuple[str, str]:
    """
    Extract return type and parameter list from a C function signature.

    Returns (return_type, params_with_parens).
    E.g. "void restart_mab(kissat *solver)" -> ("void", "(kissat *solver)")
    """
    # Strip static/inline qualifiers
    sig = re.sub(r'\b(static|inline)\s+', '', signature).strip()
    # Split at the opening paren
    m = re.match(r'(.+?)\s+\w+\s*(\(.*\))$', sig, re.DOTALL)
    if m:
        return_type = m.group(1).strip()
        params = m.group(2).strip()
        return return_type, params
    return "void", "(kissat *solver)"


def update_registry(repo_root: Path, solver_path: Path, func_name: str, info: dict) -> list[str]:
    """Update function_registry.yaml at repo root and copy to base solver."""
    updated = []

    # Repo root registry
    repo_registry_path = repo_root / "function_registry.yaml"
    registry = load_registry(repo_registry_path)
    if not registry:
        registry = {"version": "1.0", "generated": "", "functions": {}}
    if "functions" not in registry:
        registry["functions"] = {}
    registry["functions"][func_name] = info
    from datetime import date
    registry["generated"] = str(date.today())
    save_registry(registry, repo_registry_path)
    updated.append(str(repo_registry_path.relative_to(repo_root)))

    # Base solver registry (copy)
    solver_registry_path = solver_path / "function_registry.yaml"
    shutil.copy2(repo_registry_path, solver_registry_path)
    try:
        rel = solver_registry_path.relative_to(repo_root)
        updated.append(str(rel))
    except ValueError:
        updated.append(str(solver_registry_path))

    return updated


def rewrite_leader_prompt(repo_root: Path, func_name: str) -> None:
    """Replace the target function name in leader_prompt_testing.txt."""
    path = repo_root / "data" / "prompts" / "leader_prompt_testing.txt"
    text = path.read_text()

    # Pattern: "### Target Function\n\n<old_name>"
    new_text = re.sub(
        r'(### Target Function\n\n)\S+',
        rf'\g<1>{func_name}',
        text,
    )
    if new_text == text:
        print("  Warning: Could not find '### Target Function' marker in leader prompt",
              file=sys.stderr)
    path.write_text(new_text)


def rewrite_coder_prompt(
    repo_root: Path,
    solver_path: Path,
    func_name: str,
    info: dict,
    signature: str,
) -> str:
    """
    Update coder_prompt_testing.txt:
      - Update example signature return type + params
      - Replace embedded baseline source file

    Returns the source filename used for the summary.
    """
    path = repo_root / "data" / "prompts" / "coder_prompt_testing.txt"
    text = path.read_text()

    # Parse signature for return type and params
    return_type, params = parse_signature(signature)

    # 1. Update the example signature in the output format section
    #    Current pattern: "<return_type> FUNCTION_NAME_PLACEHOLDER(<params>) {"
    text = re.sub(
        r'^(\s*)\S+\s+FUNCTION_NAME_PLACEHOLDER\s*\([^)]*\)\s*\{',
        rf'\g<1>{return_type} FUNCTION_NAME_PLACEHOLDER{params} {{',
        text,
        count=1,
        flags=re.MULTILINE,
    )

    # 2. Replace everything after "### Baseline Reference"
    source_file = info["file"]  # e.g. "src/restart.c"
    source_path = solver_path / source_file
    source_content = source_path.read_text()
    filename = Path(source_file).name

    baseline_section = (
        f"### Baseline Reference\n"
        f"\n"
        f"Here is the entire {filename}. Your generated FUNCTION_NAME_PLACEHOLDER "
        f"function will replace the existing one in this file, so ensure it compiles "
        f"with the rest of the code. Make sure you faithfully implement the algorithm "
        f"you are given.\n"
        f"\n"
        f"```c\n"
        f"{source_content}"
    )
    # Ensure source content ends with newline before closing fence
    if not baseline_section.endswith("\n"):
        baseline_section += "\n"
    baseline_section += "```\n"

    # Split at the marker and replace
    marker = "### Baseline Reference"
    idx = text.find(marker)
    if idx == -1:
        print("  Warning: Could not find '### Baseline Reference' marker in coder prompt",
              file=sys.stderr)
    else:
        text = text[:idx] + baseline_section

    path.write_text(text)
    return source_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Configure LLM-SAT to target a different C function.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "function_name",
        help="C function to target (e.g. restart_mab, kissat_restarting)",
    )
    parser.add_argument(
        "--solver",
        default=None,
        help="Override base solver path (default: from path_config.yaml)",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Skip auto-detection, specify source file (e.g. src/restart.c)",
    )

    args = parser.parse_args(argv)
    func_name = args.function_name

    # 1. Resolve paths
    repo_root = find_repo_root()
    solver_path = resolve_solver_path(args.solver)

    # 2. Index the function
    if args.file:
        # Use specific file
        source_path = solver_path / args.file
        if not source_path.exists():
            print(f"Error: Source file not found: {source_path}", file=sys.stderr)
            return 1
        content = source_path.read_text()
        bounds = find_function_bounds(content, func_name)
        if bounds is None:
            print(f"Error: Function '{func_name}' not found in {args.file}", file=sys.stderr)
            return 1
        start_line, end_line, signature = bounds
        rel_path = args.file
        info = {
            "file": str(rel_path),
            "start_line": start_line,
            "end_line": end_line,
            "signature": signature,
        }
    else:
        # Auto-detect
        info = index_function(solver_path, func_name)
        if info is None:
            print(f"Error: Function '{func_name}' not found in {solver_path}/src/", file=sys.stderr)
            return 1
        signature = info["signature"]

    print(f"Configured LLM-SAT to target: {func_name}")
    print(f"  Found in: {info['file']} (lines {info['start_line']}-{info['end_line']})")
    print(f"  Signature: {info['signature']}")
    print()

    # 3. Update registries
    updated_registries = update_registry(repo_root, solver_path, func_name, info)

    # 4. Rewrite leader prompt
    rewrite_leader_prompt(repo_root, func_name)

    # 5. Rewrite coder prompt
    source_file = rewrite_coder_prompt(repo_root, solver_path, func_name, info, info["signature"])

    # 6. Print summary
    print("Updated:")
    for reg in updated_registries:
        print(f"  [OK] {reg}")
    print(f"  [OK] data/prompts/leader_prompt_testing.txt")
    print(f"  [OK] data/prompts/coder_prompt_testing.txt (embedded {source_file})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
