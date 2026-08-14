"""Build a strategies.jsonl for `analyze_diversity.py --in` from a generation tag.

Reads solvers/<tag>/{leaders,members}/algorithm_<id>/<id>.json and emits one row
per algorithm in the schema analyze_diversity expects:

  {"id", "type": "leader"|"member", "leader_id", "target_function",
   "spec": {"name", "algorithm"}, "meta": {...}}

Usage:
  python scripts/build_strategies_jsonl.py <generation_tag> --out strategies.jsonl
  python scripts/build_strategies_jsonl.py kissat_evolve_iter1 --out kev1.jsonl --leaders-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("generation_tag")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--leaders-only", action="store_true")
    ap.add_argument("--max-leaders", type=int, default=None,
                    help="If set, keep at most this many leader teams (and their members)")
    args = ap.parse_args()

    base = REPO / "solvers" / args.generation_tag
    if not base.is_dir():
        raise SystemExit(f"No such generation tag: {base}")

    leaders_dir = base / "leaders"
    members_dir = base / "members"

    rows = []
    leader_ids = set()

    # Leaders first
    if leaders_dir.is_dir():
        team_dirs = sorted(leaders_dir.glob("algorithm_*"))
        if args.max_leaders:
            team_dirs = team_dirs[: args.max_leaders]
        for d in team_dirs:
            algo_id = d.name.split("algorithm_", 1)[1]
            j = d / f"{algo_id}.json"
            if not j.exists():
                continue
            data = json.loads(j.read_text())
            rows.append({
                "id": algo_id,
                "type": "leader",
                "leader_id": algo_id,  # leader is its own team's leader
                "target_function": data.get("function_name"),
                "spec": {
                    "name": (data.get("description") or "").split(":", 1)[0][:80] or "(no name)",
                    "algorithm": data.get("description") or "",
                },
                "strategy_text": data.get("description") or "",
                "meta": {
                    "model": data.get("prompt", {}).get("model") if isinstance(data.get("prompt"), dict) else None,
                    "mutation_step": data.get("mutation_step"),
                },
            })
            leader_ids.add(algo_id)

    # Members
    if not args.leaders_only and members_dir.is_dir():
        for d in sorted(members_dir.glob("algorithm_*")):
            algo_id = d.name.split("algorithm_", 1)[1]
            j = d / f"{algo_id}.json"
            if not j.exists():
                continue
            data = json.loads(j.read_text())
            parent = data.get("parent_id") or []
            if isinstance(parent, list) and parent:
                parent_id = parent[0]
            else:
                parent_id = None
            # Filter members to only those whose leader was kept (when --max-leaders)
            if args.max_leaders and parent_id not in leader_ids:
                continue
            rows.append({
                "id": algo_id,
                "type": "member",
                "leader_id": parent_id,
                "target_function": data.get("function_name"),
                "spec": {
                    "name": (data.get("description") or "").split(":", 1)[0][:80] or "(no name)",
                    "algorithm": data.get("description") or "",
                },
                "strategy_text": data.get("description") or "",
                "meta": {
                    "model": data.get("prompt", {}).get("model") if isinstance(data.get("prompt"), dict) else None,
                    "mutation_step": data.get("mutation_step"),
                },
            })

    n_leader = sum(1 for r in rows if r["type"] == "leader")
    n_member = sum(1 for r in rows if r["type"] == "member")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} rows to {args.out} ({n_leader} leaders, {n_member} members)")


if __name__ == "__main__":
    main()
