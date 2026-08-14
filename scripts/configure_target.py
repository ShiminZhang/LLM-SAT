#!/usr/bin/env python3
"""
Configure LLM-SAT to target a different C function for optimization.

Automates the tedious manual steps of switching the target function:
  1. Index the function in the base solver
  2. Update function_registry.yaml (repo root + base solver copy)
  3. Rewrite leader_prompt_testing.txt with the new target name + embedded source
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


def rewrite_leader_prompt(
    repo_root: Path,
    solver_path: Path,
    func_name: str,
    info: dict,
) -> None:
    """Replace the target function name and embed baseline source in leader_prompt_testing.txt."""
    path = repo_root / "data" / "prompts" / "leader_prompt_testing.txt"
    text = path.read_text()

    # Pattern: "### Target Function\n\n<old_name>"
    new_text = re.sub(
        r'(### Target Function\n\n)\S+',
        rf'\g<1>{func_name}',
        text,
    )
    if not re.search(r'### Target Function\n\n\S+', text):
        print("  Warning: Could not find '### Target Function' marker in leader prompt",
              file=sys.stderr)
    path.write_text(new_text)

    _embed_baseline_reference(path, solver_path, info, LEADER_BASELINE_INTRO)


def _install_function_prompt(repo_root: Path, func_name: str) -> bool:
    """
    If a function-specific prompt exists in function_prompts/, copy it to
    coder_prompt_testing.txt.  Returns True if a prompt was installed.
    """
    func_prompt = repo_root / "data" / "prompts" / "function_prompts" / f"{func_name}.txt"
    if not func_prompt.exists():
        return False
    dest = repo_root / "data" / "prompts" / "coder_prompt_testing.txt"
    shutil.copy2(func_prompt, dest)
    return True


CODER_BASELINE_INTRO = (
    "Here is the entire {filename}. Your generated FUNCTION_NAME_PLACEHOLDER "
    "function will replace the existing one in this file, so ensure it compiles "
    "with the rest of the code. Make sure you faithfully implement the algorithm "
    "you are given."
)

LEADER_BASELINE_INTRO = (
    "For context, here is the current implementation of the target function in "
    "{filename}. Ground your algorithm design in this real code structure, but "
    "describe your algorithm conceptually (not as code) per the output format above."
)


def _embed_baseline_reference(
    prompt_path: Path,
    solver_path: Path,
    info: dict,
    intro: str,
) -> str:
    """
    Replace (or append) the '### Baseline Reference' section in a prompt file
    with the full source file contents.  Returns the source filename.
    """
    text = prompt_path.read_text()

    source_file = info["file"]  # e.g. "src/restart.c"
    source_content = (solver_path / source_file).read_text()
    filename = Path(source_file).name

    section = (
        f"### Baseline Reference\n"
        f"\n"
        f"{intro.format(filename=filename)}\n"
        f"\n"
        f"```c\n"
        f"{source_content}"
    )
    if not section.endswith("\n"):
        section += "\n"
    section += "```\n"

    marker = "### Baseline Reference"
    idx = text.find(marker)
    if idx == -1:
        text = text.rstrip() + "\n\n" + section
    else:
        text = text[:idx] + section

    prompt_path.write_text(text)
    return source_file


def rewrite_coder_prompt(
    repo_root: Path,
    solver_path: Path,
    func_name: str,
    info: dict,
    signature: str,
) -> tuple[str, bool]:
    """
    Update coder_prompt_testing.txt:
      - If a function-specific prompt exists in function_prompts/, install it
        (skip signature rewrite since those prompts have correct signatures).
      - Otherwise update the example signature return type + params.
      - Always replace the embedded baseline source file with the full file.

    Returns (source_filename, used_function_prompt).
    """
    used_function_prompt = _install_function_prompt(repo_root, func_name)

    if not used_function_prompt:
        # No function-specific prompt — update signature in the generic prompt
        path = repo_root / "data" / "prompts" / "coder_prompt_testing.txt"
        text = path.read_text()

        return_type, params = parse_signature(signature)
        text = re.sub(
            r'^(\s*)\S+\s+FUNCTION_NAME_PLACEHOLDER\s*\([^)]*\)\s*\{',
            rf'\g<1>{return_type} FUNCTION_NAME_PLACEHOLDER{params} {{',
            text,
            count=1,
            flags=re.MULTILINE,
        )
        path.write_text(text)

    # Always replace baseline reference with full source file
    path = repo_root / "data" / "prompts" / "coder_prompt_testing.txt"
    source_file = _embed_baseline_reference(path, solver_path, info, CODER_BASELINE_INTRO)
    return source_file, used_function_prompt


def update_experience_pool_data_root(config_path: Path, func_name: str) -> None:
    """Write function-specific experience_pool_data_root into path_config.yaml."""
    cfg = yaml.safe_load(config_path.read_text()) or {}
    cfg["experience_pool_data_root"] = f"src/experience_pool/data/{func_name}"
    config_path.write_text(yaml.dump(cfg, default_flow_style=False, allow_unicode=True))


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
    rewrite_leader_prompt(repo_root, solver_path, func_name, info)

    # 5. Rewrite coder prompt
    source_file, used_function_prompt = rewrite_coder_prompt(
        repo_root, solver_path, func_name, info, info["signature"],
    )

    # 6. Update path_config.yaml with function-specific experience pool root
    config_path = find_path_config()
    update_experience_pool_data_root(config_path, func_name)

    # 7. Print summary
    print("Updated:")
    for reg in updated_registries:
        print(f"  [OK] {reg}")
    print(f"  [OK] data/prompts/leader_prompt_testing.txt")
    if used_function_prompt:
        print(f"  [OK] data/prompts/coder_prompt_testing.txt "
              f"(from function_prompts/{func_name}.txt, embedded {source_file})")
    else:
        print(f"  [OK] data/prompts/coder_prompt_testing.txt (embedded {source_file})")
    print(f"  [OK] path_config.yaml (experience_pool_data_root = src/experience_pool/data/{func_name})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
