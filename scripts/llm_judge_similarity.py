#!/usr/bin/env python3
"""LLM-judge similarity sanity check (pair sampling + JSONL output).

This script provides a qualitative check alongside cosine similarity.
Instead of relying on embeddings, it asks an LLM whether two strategies are
"the same idea" vs materially different, producing a score in [0, 1] plus a
brief rationale.

When to use:
- Spot-check runs where cosine is high/compressed and you want a stronger signal.
- Validate that "within-team" pairs look more similar than "cross-team" pairs.
- Escalate suspicious near-duplicates before investing in compilation/evals.

Prompt:
Uses a template file (default: data/prompts/judge_similarity_prompt.txt) with
placeholders like {{A_NAME}}, {{A_ALGORITHM}}, {{B_NAME}}, {{B_ALGORITHM}}.

Outputs:
Writes JSONL where each line includes:
- pair identifiers (pair_id / ids / kind)
- judge fields: similarity in [0,1], same_family boolean, rationale

Notes:
- This is *not* a replacement for embedding-based metrics; treat it as a sanity check.
- Keep temperature low (0.0–0.2) for consistency.
- Use --resume to append new samples without duplicating previously judged pairs.

Examples:
Judge sampled pairs from a team batch directory:
    python scripts/llm_judge_similarity.py \
        --team-batch-dir outputs/controlled_mutation/batch_batch_<...> \
        --out outputs/controlled_mutation/judge/judged_pairs.jsonl \
        --per-kind 50 --seed 0 --resume \
        --model gpt-5.2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llmsat.utils.chatgpt_helper import get_response_from_chatgpt


def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_jsonl_line(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(_stable_json_dumps(obj) + "\n")


def _pair_id(a: str, b: str) -> str:
    x, y = (a, b) if a <= b else (b, a)
    return hashlib.sha256(f"{x}|{y}".encode("utf-8")).hexdigest()


def _extract_first_json_object(text: str) -> str:
    s = text.strip()
    if s.startswith("{"):
        return s
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1]
    raise ValueError("Could not locate JSON object in judge output")


def _render_prompt_template(template: str, a: dict, b: dict) -> str:
    a_spec = a.get("spec") or {}
    b_spec = b.get("spec") or {}

    # Intentionally avoid str.format(): prompts often contain braces for JSON.
    out = template
    out = out.replace("{{A_NAME}}", str(a_spec.get("name", "")))
    out = out.replace("{{A_ALGORITHM}}", str(a_spec.get("algorithm", "")))
    out = out.replace("{{B_NAME}}", str(b_spec.get("name", "")))
    out = out.replace("{{B_ALGORITHM}}", str(b_spec.get("algorithm", "")))
    return out


def _parse_judge_response(text: str) -> dict:
    obj_str = _extract_first_json_object(text)
    obj = json.loads(obj_str)
    if not isinstance(obj, dict):
        raise ValueError("Judge response must be a JSON object")
    sim = obj.get("similarity")
    if not isinstance(sim, (int, float)):
        raise ValueError("Judge response missing numeric 'similarity'")
    sim = float(sim)
    if sim < 0.0:
        sim = 0.0
    if sim > 1.0:
        sim = 1.0
    same_family = obj.get("same_family")
    if not isinstance(same_family, bool):
        # default heuristic
        same_family = sim >= 0.7
    rationale = obj.get("rationale")
    if not isinstance(rationale, str):
        rationale = ""
    return {"similarity": sim, "same_family": same_family, "rationale": rationale.strip()}


@dataclass(frozen=True)
class Pair:
    kind: str
    a_id: str
    b_id: str


def _sample_pairs(rows: List[dict], per_kind: int, seed: int) -> List[Pair]:
    rng = random.Random(seed)
    leaders = [r for r in rows if r.get("type") == "leader"]
    members = [r for r in rows if r.get("type") == "member"]

    by_id = {str(r.get("id")): r for r in rows}

    pairs: List[Pair] = []

    # leader-leader
    leader_ids = [str(r.get("id")) for r in leaders]
    if len(leader_ids) >= 2:
        for _ in range(per_kind):
            a, b = rng.sample(leader_ids, 2)
            pairs.append(Pair(kind="leader_leader", a_id=a, b_id=b))

    # leader-member within team
    members_by_leader: Dict[str, List[str]] = {}
    for m in members:
        lid = str(m.get("leader_id"))
        members_by_leader.setdefault(lid, []).append(str(m.get("id")))
    for _ in range(per_kind):
        if not members_by_leader:
            break
        lid = rng.choice(list(members_by_leader.keys()))
        mids = members_by_leader[lid]
        if not mids or lid not in by_id:
            continue
        mid = rng.choice(mids)
        pairs.append(Pair(kind="leader_member_within", a_id=lid, b_id=mid))

    # member-member cross team
    if len(members) >= 2:
        member_ids = [str(m.get("id")) for m in members]
        for _ in range(per_kind):
            a, b = rng.sample(member_ids, 2)
            if str(by_id.get(a, {}).get("leader_id")) == str(by_id.get(b, {}).get("leader_id")):
                continue
            pairs.append(Pair(kind="member_member_cross", a_id=a, b_id=b))

    # Deduplicate by unordered pair id
    seen = set()
    out: List[Pair] = []
    for p in pairs:
        pid = _pair_id(p.a_id, p.b_id)
        if pid in seen:
            continue
        seen.add(pid)
        out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", type=Path, required=False)
    ap.add_argument(
        "--team-batch-dir",
        type=Path,
        default=None,
        help="Load strategies from outputs/<tag>/batch_<leader_batch_id>/ produced by generate_team_data",
    )
    ap.add_argument("--out", type=Path, default=Path("outputs/llm_judge_similarity/judgements.jsonl"))
    ap.add_argument(
        "--prompt",
        type=Path,
        default=Path("data/prompts/judge_similarity_prompt.txt"),
        help="Prompt template file (uses {{A_NAME}}, {{A_ALGORITHM}}, {{B_NAME}}, {{B_ALGORITHM}})",
    )
    ap.add_argument("--per-kind", type=int, default=25, help="Sample size per pair kind")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--system-message", type=str, default=None)
    ap.add_argument("--resume", action="store_true", help="Skip pairs already in output")

    args = ap.parse_args()

    if args.team_batch_dir is None and args.strategies is None:
        raise SystemExit("Must pass --strategies JSONL or --team-batch-dir")
    if args.team_batch_dir is not None and args.strategies is not None:
        raise SystemExit("Pass only one of --strategies or --team-batch-dir")

    if args.team_batch_dir is not None:
        from llmsat.utils.team_batch_io import load_team_strategies_from_batch_dir

        rows = load_team_strategies_from_batch_dir(args.team_batch_dir)
    else:
        rows = _read_jsonl(args.strategies)
    by_id = {str(r.get("id")): r for r in rows}

    template = _read_text(args.prompt)

    pairs = _sample_pairs(rows, per_kind=int(args.per_kind), seed=int(args.seed))
    if not pairs:
        raise SystemExit("No pairs could be sampled")

    done = set()
    if args.resume and args.out.exists():
        for r in _read_jsonl(args.out):
            pid = r.get("pair_id")
            if isinstance(pid, str):
                done.add(pid)

    for p in pairs:
        pid = _pair_id(p.a_id, p.b_id)
        if pid in done:
            continue
        a = by_id.get(p.a_id)
        b = by_id.get(p.b_id)
        if a is None or b is None:
            continue

        prompt = _render_prompt_template(template, a, b)
        raw = get_response_from_chatgpt(
            prompt=prompt,
            system_message=args.system_message,
            model=args.model,
            temperature=float(args.temperature),
        )
        judged = _parse_judge_response(raw)

        record = {
            "pair_id": pid,
            "kind": p.kind,
            "a_id": p.a_id,
            "b_id": p.b_id,
            "similarity": judged["similarity"],
            "same_family": judged["same_family"],
            "rationale": judged["rationale"],
            "model": args.model or os.environ.get("OPENAI_MODEL"),
        }
        _write_jsonl_line(args.out, record)
        done.add(pid)

    print(f"Wrote judgements to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
