#!/usr/bin/env python3
"""Generate a local JSONL batch of diverse leader/member strategies.

This is a lightweight, file-based alternative to the DB-backed pipeline.
It is designed for the "diversity + compile-rate" experiments.

Outputs JSONL records with linkage:
- leaders:  type="leader", leader_id=id
- members:  type="member", leader_id=<leader id>

By default this script generates:
- N leaders using data/prompts/ae_prompt_restart.txt
- M members per leader using data/prompts/variant_prompt.txt

Notes:
- Leaders are required to include target_function and we enforce it.
- Members omit target_function in the prompt; we add it to the saved record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

from llmsat.data.algorithm_parse import parse_algorithm_spec_json
from llmsat.utils.chatgpt_helper import get_response_from_chatgpt


TARGET_FUNCTION = "kissat_restarting"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _extract_first_json_object(text: str) -> str:
    """Extract the first JSON object from an LLM response.

    Handles:
    - Pure JSON output
    - ```json ... ``` fenced blocks
    - Extra pre/post text

    Returns a JSON object string.
    """
    s = text.strip()
    if s.startswith("{"):
        return s

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    # Fallback: find first {...} using a greedy-ish approach.
    # This isn't a full JSON parser, but works well for typical LLM outputs.
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1].strip()

    raise ValueError("Could not locate a JSON object in model output")


def _parse_member_spec(text: str) -> dict:
    obj_str = _extract_first_json_object(text)
    try:
        obj = json.loads(obj_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid member JSON: {exc}") from exc

    if not isinstance(obj, dict):
        raise ValueError("Member spec must be a JSON object")

    name = obj.get("name")
    algorithm = obj.get("algorithm")
    reason = obj.get("reason", "")

    if not isinstance(name, str) or not name.strip():
        raise ValueError("Member 'name' must be a non-empty string")
    if not isinstance(algorithm, str) or not algorithm.strip():
        raise ValueError("Member 'algorithm' must be a non-empty string")
    if reason is None:
        reason = ""
    if not isinstance(reason, str):
        raise ValueError("Member 'reason' must be a string")

    return {"name": name.strip(), "algorithm": algorithm.strip(), "reason": reason.strip()}


def _strategy_text(spec: dict) -> str:
    parts = [spec.get("name", ""), spec.get("algorithm", ""), spec.get("reason", "")]
    return "\n\n".join([p for p in parts if isinstance(p, str) and p.strip()]).strip()


def _render_variant_prompt(template: str, leader_algorithm_text: str, seed: str) -> str:
    # Avoid .format() (templates contain many literal braces elsewhere in repo).
    prompt = template.replace("{leader_algorithm}", leader_algorithm_text)
    prompt += f"\n\nSeed: {seed}\n"
    return prompt


def _render_leader_prompt(template: str, seed: str) -> str:
    # No placeholders expected here, but we still append a seed.
    return template.rstrip() + f"\n\nSeed: {seed}\n"


@dataclass(frozen=True)
class GenerationConfig:
    leader_prompt_path: Path
    member_prompt_path: Path
    out_path: Path
    model: Optional[str]
    system_message: Optional[str]
    leaders: int
    members_per_leader: int
    min_temperature: float
    max_temperature: float
    dry_run: bool
    resume: bool
    tag: Optional[str]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_jsonl_line(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(_stable_json_dumps(obj) + "\n")


def _generate_one_leader(cfg: GenerationConfig, leader_prompt_template: str, leader_index: int) -> dict:
    seed = f"leader-{leader_index}-{_sha256_str(_utc_now_iso() + str(random.random()))[:12]}"
    temp = random.uniform(cfg.min_temperature, cfg.max_temperature)

    prompt = _render_leader_prompt(leader_prompt_template, seed=seed)

    if cfg.dry_run:
        # Minimal deterministic placeholder.
        spec = {
            "name": f"DRYRUN Leader {leader_index}",
            "algorithm": f"Dry-run leader algorithm seed={seed}",
            "reason": "dry-run",
            "target_function": TARGET_FUNCTION,
        }
        raw = ""
    else:
        raw = get_response_from_chatgpt(
            prompt=prompt,
            system_message=cfg.system_message,
            model=cfg.model,
            temperature=temp,
        )
        spec, target_function = parse_algorithm_spec_json(_extract_first_json_object(raw))
        if target_function != TARGET_FUNCTION:
            raise ValueError(
                f"Leader target_function mismatch: expected '{TARGET_FUNCTION}', got '{target_function}'"
            )

    strategy_text = _strategy_text(spec)
    leader_id = _sha256_str(_stable_json_dumps({"type": "leader", "spec": spec}))

    return {
        "id": leader_id,
        "type": "leader",
        "leader_id": leader_id,
        "target_function": TARGET_FUNCTION,
        "spec": spec,
        "strategy_text": strategy_text,
        "meta": {
            "tag": cfg.tag,
            "seed": seed,
            "leader_index": leader_index,
            "model": cfg.model or os.environ.get("OPENAI_MODEL"),
            "temperature": temp,
            "created_at": _utc_now_iso(),
        },
        "raw": raw,
    }


def _generate_one_member(
    cfg: GenerationConfig,
    member_prompt_template: str,
    leader_record: dict,
    member_index: int,
) -> dict:
    leader_algorithm_text = leader_record["spec"]["algorithm"]
    seed = f"member-{leader_record['id'][:8]}-{member_index}-{_sha256_str(_utc_now_iso() + str(random.random()))[:12]}"
    temp = random.uniform(cfg.min_temperature, cfg.max_temperature)

    prompt = _render_variant_prompt(member_prompt_template, leader_algorithm_text, seed=seed)

    if cfg.dry_run:
        spec = {
            "name": f"DRYRUN Member {member_index}",
            "algorithm": f"Dry-run member variant seed={seed}",
            "reason": "dry-run",
        }
        raw = ""
    else:
        raw = get_response_from_chatgpt(
            prompt=prompt,
            system_message=cfg.system_message,
            model=cfg.model,
            temperature=temp,
        )
        spec = _parse_member_spec(raw)

    spec = {**spec, "target_function": TARGET_FUNCTION}
    strategy_text = _strategy_text(spec)
    member_id = _sha256_str(
        _stable_json_dumps({"type": "member", "leader_id": leader_record["id"], "spec": spec})
    )

    return {
        "id": member_id,
        "type": "member",
        "leader_id": leader_record["id"],
        "target_function": TARGET_FUNCTION,
        "spec": spec,
        "strategy_text": strategy_text,
        "meta": {
            "tag": cfg.tag,
            "seed": seed,
            "member_index": member_index,
            "model": cfg.model or os.environ.get("OPENAI_MODEL"),
            "temperature": temp,
            "created_at": _utc_now_iso(),
        },
        "raw": raw,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leaders", type=int, default=10, help="Number of leaders to generate")
    ap.add_argument("--members-per-leader", type=int, default=3, help="Members per leader")
    ap.add_argument(
        "--leader-prompt-path",
        type=Path,
        default=Path("data/prompts/ae_prompt_restart.txt"),
    )
    ap.add_argument(
        "--member-prompt-path",
        type=Path,
        default=Path("data/prompts/variant_prompt.txt"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/diverse_batch/strategies.jsonl"),
        help="Output JSONL path",
    )
    ap.add_argument("--model", type=str, default=None, help="OpenAI model (defaults to OPENAI_MODEL)")
    ap.add_argument("--system-message", type=str, default=None, help="Optional system message")
    ap.add_argument("--min-temperature", type=float, default=0.6)
    ap.add_argument("--max-temperature", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true", help="Do not call OpenAI; emit placeholder JSONL")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Append to existing output instead of refusing to overwrite",
    )
    ap.add_argument("--tag", type=str, default=None, help="Tag to include in output meta")

    args = ap.parse_args()

    cfg = GenerationConfig(
        leader_prompt_path=args.leader_prompt_path,
        member_prompt_path=args.member_prompt_path,
        out_path=args.out,
        model=args.model,
        system_message=args.system_message,
        leaders=args.leaders,
        members_per_leader=args.members_per_leader,
        min_temperature=float(args.min_temperature),
        max_temperature=float(args.max_temperature),
        dry_run=bool(args.dry_run),
        resume=bool(args.resume),
        tag=args.tag,
    )

    if cfg.out_path.exists() and not cfg.resume:
        raise SystemExit(f"Refusing to overwrite existing file (use --resume): {cfg.out_path}")

    leader_prompt_template = _read_text(cfg.leader_prompt_path)
    member_prompt_template = _read_text(cfg.member_prompt_path)

    random.seed()  # Use system entropy

    pbar = tqdm(total=cfg.leaders * (1 + cfg.members_per_leader), desc="Generating")

    for i in range(cfg.leaders):
        leader = _generate_one_leader(cfg, leader_prompt_template, leader_index=i)
        _write_jsonl_line(cfg.out_path, leader)
        pbar.update(1)

        for j in range(cfg.members_per_leader):
            member = _generate_one_member(cfg, member_prompt_template, leader, member_index=j)
            _write_jsonl_line(cfg.out_path, member)
            pbar.update(1)

    pbar.close()

    print(f"Wrote strategies to: {cfg.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
