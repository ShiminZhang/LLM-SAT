#!/usr/bin/env python3
"""
Generate JSONL batch inputs for OpenAI's Responses API from prompt files.

Usage examples:

1) Single prompt file -> one JSONL line (stdout):
   python scripts/generate_openai_batch.py \
     -i data/prompt.txt \
     --system "You are an AI researcher specialising in SAT solver heuristics." \
     --model gpt-4.1 --temperature 0.7

2) Directory of prompts (*.txt) -> many JSONL lines (to file):
   python scripts/generate_openai_batch.py \
     -i prompts/ --glob "*.txt" -o data/batch.jsonl \
     --custom-id-template "req-{stem}-{index:04d}"

3) Split a file by delimiter into multiple prompts:
   python scripts/generate_openai_batch.py \
     -i data/multiprompt.txt --delimiter "\n\n---\n\n" \
     --custom-id-template "req-heuristic-{index:04d}"

Notes:
- The output JSONL lines follow the exact shape required by OpenAI batches:
  {"custom_id": str, "method": "POST", "url": "/v1/responses", "body": {"model": str, "input": [...], "temperature": float}}
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid as uuidlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class GeneratorConfig:
    input_path: Optional[Path]
    output_path: Optional[Path]
    system_message: str
    system_file: Optional[Path]
    model: str
    temperature: float
    method: str
    url: str
    custom_id_template: str
    start_index: int
    glob_pattern: str
    sort: bool
    delimiter: Optional[str]
    per_line: bool
    target_count: int

def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_system_message(cfg: GeneratorConfig) -> str:
    if cfg.system_file:
        return read_text_file(cfg.system_file).strip("\n")
    return cfg.system_message


def iter_prompts_from_file(path: Path, delimiter: Optional[str], per_line: bool) -> List[str]:
    content = read_text_file(path)
    if delimiter is not None:
        parts = content.split(delimiter)
        prompts = [part.strip("\n") for part in parts if part.strip()]
        return prompts
    if per_line:
        prompts = [line.strip() for line in content.splitlines() if line.strip()]
        return prompts
    return [content.strip("\n")]


def iter_prompts_from_dir(directory: Path, glob_pattern: str, delimiter: Optional[str], per_line: bool, sort: bool) -> List[tuple[str, Path]]:
    files = list(directory.glob(glob_pattern))
    if sort:
        files.sort()
    results: List[tuple[str, Path]] = []
    for file_path in files:
        if not file_path.is_file():
            continue
        prompts = iter_prompts_from_file(file_path, delimiter, per_line)
        for prompt in prompts:
            results.append((prompt, file_path))
    return results


def render_custom_id(template: str, index: int, stem: Optional[str]) -> str:
    # Provide a few convenient variables into the template namespace
    namespace = {
        "index": index,       # 1-based index
        "index0": index - 1,  # 0-based index
        "i": index,           # alias
        "stem": stem or "",
        "uuid": str(uuidlib.uuid4()),
    }
    try:
        return template.format(**namespace)
    except Exception:
        # Fallback to a safe default if formatting fails
        return f"req-{index:04d}"


def build_record(system_message: str, user_prompt: str, model: str, temperature: float, method: str, url: str, custom_id: str) -> dict:
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]
    return {
        "custom_id": custom_id,
        "method": method,
        "url": url,
        "body": {
            "model": model,
            "input": messages,
            "temperature": temperature,
        },
    }


def write_jsonl(records: Iterable[dict], output: Optional[Path]) -> None:
    if output is None:
        out_fh = sys.stdout
        close_needed = False
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        out_fh = output.open("w", encoding="utf-8")
        close_needed = True
    try:
        for record in records:
            out_fh.write(json.dumps(record, ensure_ascii=False))
            out_fh.write("\n")
        out_fh.flush()
    finally:
        if close_needed:
            out_fh.close()


def generate_records(cfg: GeneratorConfig) -> List[dict]:
    system_message = load_system_message(cfg)

    records: List[dict] = []
    index = cfg.start_index

    if cfg.input_path is None:
        # Read entire stdin as a single prompt
        user_prompt = sys.stdin.read().strip("\n")
        custom_id = render_custom_id(cfg.custom_id_template, index, stem=None)
        records.append(
            build_record(system_message, user_prompt, cfg.model, cfg.temperature, cfg.method, cfg.url, custom_id)
        )
        return records

    if cfg.input_path.is_file():
        prompts = iter_prompts_from_file(cfg.input_path, cfg.delimiter, cfg.per_line)
        for prompt in prompts:
            custom_id = render_custom_id(cfg.custom_id_template, index, stem=cfg.input_path.stem)
            records.append(
                build_record(system_message, prompt, cfg.model, cfg.temperature, cfg.method, cfg.url, custom_id)
            )
            index += 1
        return records

    if cfg.input_path.is_dir():
        pairs = iter_prompts_from_dir(cfg.input_path, cfg.glob_pattern, cfg.delimiter, cfg.per_line, cfg.sort)
        for prompt, path in pairs:
            custom_id = render_custom_id(cfg.custom_id_template, index, stem=path.stem)
            records.append(
                build_record(system_message, prompt, cfg.model, cfg.temperature, cfg.method, cfg.url, custom_id)
            )
            index += 1
        return records

    raise FileNotFoundError(f"Input path not found: {cfg.input_path}")


def parse_args(argv: Optional[List[str]] = None) -> GeneratorConfig:
    parser = argparse.ArgumentParser(description="Generate JSONL batch inputs for OpenAI Responses API")

    parser.add_argument("-i", "--input", dest="input_path", type=str, default="data/prompt_heuristic.txt",
                        help="Input file or directory. If omitted, read a single prompt from stdin.")
    parser.add_argument("-o", "--output", dest="output_path", type=str, default=None,
                        help="Output JSONL path. Defaults to stdout.")

    parser.add_argument("--system", dest="system_message", type=str, default="You are an AI researcher specialising in SAT solver heuristics.",
                        help="System message string to prepend to each request.")
    parser.add_argument("-S", "--system-file", dest="system_file", type=str, default=None,
                        help="Path to a file containing the system message.")
    parser.add_argument("-n", "--target-count", dest="target_count", type=int, default=500)
    parser.add_argument("-m", "--model", dest="model", type=str, default="gpt-4.1", 
                        help="Model id (e.g., gpt-4.1)")
    parser.add_argument("-t", "--temperature", dest="temperature", type=float, default=0.7,
                        help="Sampling temperature.")
    parser.add_argument("--method", dest="method", type=str, default="POST",
                        help="HTTP method for batch entries (default: POST)")
    parser.add_argument("--url", dest="url", type=str, default="/v1/responses",
                        help="API path for batch entries (default: /v1/responses)")

    parser.add_argument("--custom-id-template", dest="custom_id_template", type=str, default="req-{index:04d}",
                        help="Python format template for custom_id. Variables: {index}, {index0}, {i}, {stem}, {uuid}")
    parser.add_argument("--start-index", dest="start_index", type=int, default=1,
                        help="Starting index for {index} (default: 1)")

    parser.add_argument("--glob", dest="glob_pattern", type=str, default="*.txt",
                        help="Glob pattern when input is a directory (default: *.txt)")
    parser.add_argument("--sort", dest="sort", action="store_true", default=False,
                        help="Sort directory files by name before processing.")

    parser.add_argument("--delimiter", dest="delimiter", type=str, default=None,
                        help="If set, split file content on this exact delimiter into multiple prompts.")
    parser.add_argument("--per-line", dest="per_line", action="store_true", default=False,
                        help="If set, treat each non-empty line in a file as a single prompt.")

    args = parser.parse_args(argv)
    if args.output_path is None:
        args.output_path = args.input_path.replace(".txt", "batchinput.jsonl")
    input_path = Path(args.input_path) if args.input_path else None
    output_path = Path(args.output_path) if args.output_path else None
    system_file = Path(args.system_file) if args.system_file else None

    return GeneratorConfig(
        input_path=input_path,
        output_path=output_path,
        system_message=args.system_message,
        system_file=system_file,
        model=args.model,
        temperature=args.temperature,
        method=args.method,
        url=args.url,
        custom_id_template=args.custom_id_template,
        start_index=args.start_index,
        glob_pattern=args.glob_pattern,
        sort=args.sort,
        delimiter=args.delimiter,
        per_line=args.per_line,
        target_count=args.target_count,
    )


def main(argv: Optional[List[str]] = None) -> int:
    cfg = parse_args(argv)
    try:
        # write exactly 1k records to one file (replicate if needed)
        base_records = generate_records(cfg)

        target_count = cfg.target_count
        if len(base_records) >= target_count:
            write_jsonl(base_records[:target_count], cfg.output_path)
        else:
            replicated: List[dict] = []
            total_written = 0
            index = cfg.start_index

            # infer stem when single file input; otherwise leave empty
            stem: Optional[str] = None
            if cfg.input_path is not None and cfg.input_path.exists() and cfg.input_path.is_file():
                stem = cfg.input_path.stem

            while total_written < target_count:
                for record in base_records:
                    if total_written >= target_count:
                        break
                    system_message = record["body"]["input"][0]["content"]
                    user_prompt = record["body"]["input"][1]["content"]
                    custom_id = render_custom_id(cfg.custom_id_template, index, stem)
                    replicated.append(
                        build_record(
                            system_message,
                            user_prompt,
                            cfg.model,
                            cfg.temperature,
                            cfg.method,
                            cfg.url,
                            custom_id,
                        )
                    )
                    index += 1
                    total_written += 1

            write_jsonl(replicated, cfg.output_path)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


