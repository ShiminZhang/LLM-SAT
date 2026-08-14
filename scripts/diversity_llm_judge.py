"""LLM-judge for mechanistic equivalence between heuristic algorithm pairs.

Uses an OpenAI chat model to read two algorithm descriptions and decide whether
they encode mechanistically equivalent strategies. Outputs per-pair verdicts +
aggregate stats.

API key from OPENAI_API_KEY env var.

Usage:
  OPENAI_API_KEY=... python scripts/diversity_llm_judge.py \
      --in outputs/diversity/kev_iter1_strategies.jsonl \
      --out outputs/diversity/kev_iter1_llm_judge.json \
      --model gpt-5-mini \
      --pairs leader_leader  # all 27C2=351 leader pairs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import List


SYSTEM_PROMPT = """You are an expert in SAT solver heuristics. Given two natural-language descriptions of CDCL solver heuristic strategies, decide whether they describe mechanistically equivalent algorithms.

"Mechanistically equivalent" means: they would behave identically (or near-identically) at the algorithm level — same inputs read, same decisions made, same state updated, same control flow — even if their wording or constants differ.

"Distinct" means: they differ in at least one meaningful structural component (different signals consumed, different decision rule, different update mechanism, different control flow), such that they could measurably diverge on at least some SAT instances.

Reply ONLY with a JSON object: {"verdict": "equivalent" | "distinct", "reason": "<one sentence>"}"""

USER_TEMPLATE = """Algorithm A:
{algo_a}

Algorithm B:
{algo_b}

Are these mechanistically equivalent or distinct?"""


def call_judge(client, model: str, algo_a: str, algo_b: str) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(algo_a=algo_a, algo_b=algo_b)},
        ],
        response_format={"type": "json_object"},
    )
    txt = resp.choices[0].message.content
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return {"verdict": "parse_error", "reason": txt[:200]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--pairs", choices=["leader_leader"], default="leader_leader")
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--max-pairs", type=int, default=None,
                    help="If set, judge at most this many pairs (sampled random)")
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
            if line:
                rows.append(json.loads(line))

    leaders = [r for r in rows if r.get("type") == "leader"]
    print(f"Loaded {len(leaders)} leaders")

    pairs = list(combinations(range(len(leaders)), 2))
    if args.max_pairs and len(pairs) > args.max_pairs:
        import random
        random.seed(42)
        pairs = random.sample(pairs, args.max_pairs)
    print(f"Judging {len(pairs)} pairs with {args.model}...")

    results = []
    n_eq = n_dist = n_err = 0
    for k, (i, j) in enumerate(pairs):
        a = leaders[i].get("strategy_text", "")[: args.max_chars]
        b = leaders[j].get("strategy_text", "")[: args.max_chars]
        try:
            verdict = call_judge(client, args.model, a, b)
        except Exception as e:
            verdict = {"verdict": "api_error", "reason": str(e)[:200]}
        v = verdict.get("verdict", "unknown")
        if v == "equivalent":
            n_eq += 1
        elif v == "distinct":
            n_dist += 1
        else:
            n_err += 1
        results.append({
            "id_a": leaders[i]["id"],
            "id_b": leaders[j]["id"],
            "verdict": v,
            "reason": verdict.get("reason", ""),
        })
        if (k + 1) % 25 == 0 or k + 1 == len(pairs):
            print(f"  judged {k+1}/{len(pairs)}  eq={n_eq} dist={n_dist} err={n_err}")

    out = {
        "model": args.model,
        "pair_kind": args.pairs,
        "n_leaders": len(leaders),
        "n_pairs_judged": len(pairs),
        "counts": {
            "equivalent": n_eq,
            "distinct": n_dist,
            "errors": n_err,
        },
        "fraction_equivalent": n_eq / len(pairs) if pairs else 0.0,
        "fraction_distinct": n_dist / len(pairs) if pairs else 0.0,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print()
    print(f"=== Summary ===")
    print(f"  total pairs:     {len(pairs)}")
    print(f"  equivalent:      {n_eq}  ({100*n_eq/len(pairs):.1f}%)")
    print(f"  distinct:        {n_dist}  ({100*n_dist/len(pairs):.1f}%)")
    print(f"  parse/api err:   {n_err}")
    print(f"  output: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
