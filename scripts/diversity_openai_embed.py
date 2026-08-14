"""Compute OpenAI-embedding cosine similarity diversity over a strategies.jsonl.

Designed as a sibling to analyze_diversity.py — same input + output JSON shape,
but uses OpenAI's text-embedding-3-large (or -small) instead of TF-IDF/Qwen3.

API key must be in OPENAI_API_KEY (never printed).

Usage:
  OPENAI_API_KEY=... python scripts/diversity_openai_embed.py \
      --in outputs/diversity/kev_iter1_strategies.jsonl \
      --out-dir outputs/diversity/kev_iter1_openai \
      --model text-embedding-3-large
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", default="text-embedding-3-large",
                    choices=["text-embedding-3-large", "text-embedding-3-small"])
    ap.add_argument("--duplicate-threshold", type=float, default=0.95)
    ap.add_argument("--max-duplicate-pairs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-chars", type=int, default=8000,
                    help="Truncate strategy text to this many chars before embedding")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY env var must be set", file=sys.stderr)
        return 1

    from openai import OpenAI
    client = OpenAI()

    rows = []
    with args.in_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        raise SystemExit("No rows in JSONL")

    leaders = [r for r in rows if r.get("type") == "leader"]
    members = [r for r in rows if r.get("type") == "member"]

    texts = [(r.get("strategy_text") or "")[: args.max_chars] for r in rows]
    if any(not t for t in texts):
        n_empty = sum(1 for t in texts if not t)
        print(f"WARNING: {n_empty} rows have empty strategy_text", file=sys.stderr)

    # Batch-embed
    print(f"Embedding {len(texts)} strategies via {args.model} (batch size {args.batch_size})")
    vectors: List[List[float]] = []
    for i in range(0, len(texts), args.batch_size):
        batch = texts[i : i + args.batch_size]
        # API rejects empty strings — pad with a single space
        batch = [t if t else " " for t in batch]
        resp = client.embeddings.create(model=args.model, input=batch)
        for d in resp.data:
            vectors.append(d.embedding)
        print(f"  embedded {min(i + args.batch_size, len(texts))}/{len(texts)}")

    M = np.array(vectors, dtype=np.float32)
    # Normalize (cosine = dot of L2-normalized vectors)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    M_n = M / np.clip(norms, 1e-12, None)
    sim_all = M_n @ M_n.T

    # Build index maps
    id_list = [r["id"] for r in rows]
    idx_by_id = {sid: i for i, sid in enumerate(id_list)}

    leader_ids = [r["id"] for r in leaders]
    member_ids = [r["id"] for r in members]
    leader_idx = [idx_by_id[i] for i in leader_ids if i in idx_by_id]
    member_idx = [idx_by_id[i] for i in member_ids if i in idx_by_id]

    def upper_triangle(submat):
        n = submat.shape[0]
        if n < 2:
            return []
        iu = np.triu_indices(n, k=1)
        return [float(x) for x in submat[iu]]

    # Group similarities
    leader_leader_vals = upper_triangle(sim_all[np.ix_(leader_idx, leader_idx)])
    member_member_all_vals = upper_triangle(sim_all[np.ix_(member_idx, member_idx)])

    members_by_leader: Dict[str, List[str]] = defaultdict(list)
    for m in members:
        members_by_leader[m.get("leader_id")].append(m["id"])

    leader_member_vals: List[float] = []
    member_member_within_vals: List[float] = []
    for lid in leader_ids:
        if lid not in idx_by_id:
            continue
        li = idx_by_id[lid]
        team_mids = [mid for mid in members_by_leader.get(lid, []) if mid in idx_by_id]
        for mid in team_mids:
            leader_member_vals.append(float(sim_all[li, idx_by_id[mid]]))
        team_idx = [idx_by_id[mid] for mid in team_mids]
        if team_idx:
            member_member_within_vals.extend(upper_triangle(sim_all[np.ix_(team_idx, team_idx)]))

    member_member_within_set = set()
    for lid in leader_ids:
        team_mids = [mid for mid in members_by_leader.get(lid, []) if mid in idx_by_id]
        for i, m1 in enumerate(team_mids):
            for m2 in team_mids[i + 1 :]:
                member_member_within_set.add(tuple(sorted([m1, m2])))

    member_member_cross_vals: List[float] = []
    for i in range(len(member_idx)):
        for j in range(i + 1, len(member_idx)):
            pair = tuple(sorted([member_ids[i], member_ids[j]]))
            if pair in member_member_within_set:
                continue
            member_member_cross_vals.append(float(sim_all[member_idx[i], member_idx[j]]))

    def describe(vals):
        if not vals:
            return {"count": 0, "mean": None, "median": None, "min": None, "max": None, "p10": None, "p90": None}
        a = np.array(vals)
        return {
            "count": int(len(vals)),
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "p10": float(np.percentile(a, 10)),
            "p90": float(np.percentile(a, 90)),
        }

    similarity = {
        "leader_leader": describe(leader_leader_vals),
        "leader_member_within_team": describe(leader_member_vals),
        "member_member_within_team": describe(member_member_within_vals),
        "member_member_cross_team": describe(member_member_cross_vals),
        "member_member_all": describe(member_member_all_vals),
    }

    # Find duplicates above threshold
    dup_pairs = []
    n = len(id_list)
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim_all[i, j])
            if s >= args.duplicate_threshold:
                dup_pairs.append({
                    "id_a": id_list[i],
                    "id_b": id_list[j],
                    "type_a": rows[i]["type"],
                    "type_b": rows[j]["type"],
                    "leader_a": rows[i].get("leader_id"),
                    "leader_b": rows[j].get("leader_id"),
                    "similarity": s,
                })
    dup_pairs.sort(key=lambda p: -p["similarity"])
    dup_pairs = dup_pairs[: args.max_duplicate_pairs]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "diversity_report.json"
    out.write_text(json.dumps({
        "counts": {"leaders": len(leaders), "members": len(members), "total": len(rows)},
        "embedding": {"dim": int(M.shape[1]), "method": f"openai/{args.model}"},
        "input": str(args.in_path),
        "duplicates": {"threshold": args.duplicate_threshold, "pairs": dup_pairs},
        "similarity": similarity,
    }, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
