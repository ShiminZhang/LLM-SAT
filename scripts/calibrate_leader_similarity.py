#!/usr/bin/env python3
"""Calibrate leader–leader cosine similarity (optionally with LLM judging).

This script is a *leader-only* diagnostic:
- Embed leader strategies with a chosen embedding model (typically Qwen3-Embedding-8B).
- Compute leader–leader cosine similarities.
- Report extrema (most similar / most different) and top-k lists.
- Optionally send selected pairs to an LLM judge (recommended model: GPT-5.2) to
    sanity-check whether high cosine actually means “same family”.

Use this before (or alongside) diversity analysis to understand cosine
"compression" for a run tag. It helps answer questions like:
- Are leaders truly distinct, or just paraphrases?
- For this embedding model, what cosine range is “meaningfully different”?

How many judge rows should I expect?
If there are N leaders, there are at most N*(N-1)/2 unique leader pairs.
The number of judged rows depends on:
- --judge (enabled/disabled)
- --judge-scope (which pair sets to judge)
- --top-k (how many pairs are included in the top lists)

Example: N=5 leaders => 10 total pairs, so "scope=all" can only yield 10 rows.
Example: N=10 leaders => 45 total pairs; with top-k=10, you may see ~20 judged
pairs (10 most similar + 10 most different), not all 45.

Usage:
GPU (recommended for 8B, via Slurm wrapper):
    sbatch scripts/slurm_calibrate_qwen8b_leaders.sh \
        outputs/<tag>/batch_batch_<...> <tag> \
        --include-distribution \
        --judge --judge-model gpt-5.2 --judge-scope all --top-k 10

CPU (slow for 8B; mainly for debugging):
    python scripts/calibrate_leader_similarity.py \
        --team-batch-dir outputs/<tag>/batch_batch_<...> \
        --allow-cpu --top-k 5

Output:
Writes a JSON report (see --out) containing cosines, selected pairs, and any
LLM judge results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_stable_json_dumps(obj) + "\n", encoding="utf-8")


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
    sim = max(0.0, min(1.0, sim))

    same_family = obj.get("same_family")
    if not isinstance(same_family, bool):
        same_family = sim >= 0.7

    rationale = obj.get("rationale")
    if not isinstance(rationale, str):
        rationale = ""

    return {"similarity": sim, "same_family": same_family, "rationale": rationale.strip()}


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def _embed_qwen3_hf(
    texts: List[str],
    model_id: str,
    batch_size: int,
    max_length: int,
    dtype: str,
    trust_remote_code: bool,
    require_cuda: bool,
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Run on a GPU node or pass --allow-cpu.")

    torch_dtype: Any
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float32":
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=trust_remote_code, torch_dtype=torch_dtype)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    vecs: List[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            last = out.last_hidden_state  # (b, t, h)
            mask = enc.get("attention_mask")
            if mask is None:
                pooled = last.mean(dim=1)
            else:
                mask_f = mask.unsqueeze(-1).to(last.dtype)
                summed = (last * mask_f).sum(dim=1)
                denom = mask_f.sum(dim=1).clamp(min=1e-6)
                pooled = summed / denom
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vecs.append(pooled.detach().cpu().numpy())

    mat = np.concatenate(vecs, axis=0)
    # already normalized, but keep robust
    return _normalize_rows(mat.astype(np.float32, copy=False))


def _leader_text(row: dict, field: str) -> str:
    spec = row.get("spec") or {}
    if field == "algorithm":
        return str(spec.get("algorithm") or "")
    if field == "strategy_text":
        return str(row.get("strategy_text") or "")
    if field == "name+algorithm":
        name = str(spec.get("name") or "")
        alg = str(spec.get("algorithm") or "")
        return (name + "\n\n" + alg).strip()
    raise ValueError(f"Unknown text field: {field}")


def _upper_triangle_extrema(sim: np.ndarray) -> tuple[tuple[int, int, float], tuple[int, int, float]]:
    """Return (max_pair, min_pair) where each is (i, j, sim)."""
    n = sim.shape[0]
    best = (-1, -1, float("-inf"))
    worst = (-1, -1, float("inf"))
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim[i, j])
            if s > best[2]:
                best = (i, j, s)
            if s < worst[2]:
                worst = (i, j, s)
    return best, worst


def _topk_pairs(sim: np.ndarray, k: int, largest: bool) -> List[tuple[int, int, float]]:
    n = sim.shape[0]
    pairs: List[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j, float(sim[i, j])))
    pairs.sort(key=lambda t: t[2], reverse=largest)
    return pairs[:k]


def _all_pairs(sim: np.ndarray) -> List[tuple[int, int, float]]:
    n = sim.shape[0]
    pairs: List[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j, float(sim[i, j])))
    return pairs


def _cosine_percentiles(values: List[float], percentiles: List[float]) -> Dict[str, float]:
    if not values:
        return {str(p): float("nan") for p in percentiles}
    arr = np.asarray(values, dtype=np.float64)
    qs = np.quantile(arr, np.asarray(percentiles, dtype=np.float64) / 100.0, method="linear")
    return {str(p): float(q) for p, q in zip(percentiles, qs)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--team-batch-dir",
        type=Path,
        required=True,
        help="outputs/<tag>/batch_<leader_batch_id>/ directory",
    )
    ap.add_argument(
        "--text-field",
        choices=["algorithm", "strategy_text", "name+algorithm"],
        default="algorithm",
        help="What to embed for each leader",
    )
    ap.add_argument("--qwen-model", type=str, default="Qwen/Qwen3-Embedding-8B")
    ap.add_argument("--qwen-batch-size", type=int, default=2)
    ap.add_argument("--qwen-max-length", type=int, default=512)
    ap.add_argument(
        "--qwen-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    ap.add_argument(
        "--qwen-trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the Qwen model",
    )
    ap.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow running on CPU (slow); by default, requires CUDA",
    )

    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write a JSON report to this path (default: outputs/<tag>/calibration/qwen8b_leaders.json)",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Include top-k most similar and top-k most different leader pairs",
    )
    ap.add_argument(
        "--include-distribution",
        action="store_true",
        help="Include full leader-leader cosine distribution summary (percentiles, mean/std) in the report",
    )
    ap.add_argument(
        "--dump-all-pairs",
        action="store_true",
        help="Include every leader-leader pair (ids/names/cosine) in the report, sorted by cosine desc",
    )
    ap.add_argument(
        "--dump-all-pairs-max",
        type=int,
        default=5000,
        help="Safety limit: only dump all pairs if num_pairs <= this (default: 5000)",
    )
    ap.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("LEADER_ID_A", "LEADER_ID_B"),
        help="Manually request a specific leader pair to be reported (repeatable)",
    )

    # Optional judge escalation
    ap.add_argument(
        "--judge",
        action="store_true",
        help="If set, run an LLM judge on selected pairs",
    )
    ap.add_argument(
        "--judge-model",
        type=str,
        default="gpt-5.2",
        help="Judge model name",
    )
    ap.add_argument(
        "--judge-prompt",
        type=Path,
        default=Path("data/prompts/judge_similarity_prompt.txt"),
        help="Judge prompt template",
    )
    ap.add_argument(
        "--escalate-above",
        type=float,
        default=0.8,
        help="Only judge pairs with cosine similarity >= this threshold",
    )
    ap.add_argument(
        "--judge-scope",
        choices=["all", "top", "extrema", "suspect"],
        default="all",
        help=(
            "Which pairs to send to the judge when --judge is set. "
            "'all' judges top-k most similar + top-k most different + extrema + manual pairs; "
            "'top' judges both top-k lists + manual; "
            "'extrema' judges most-similar + most-different + manual; "
            "'suspect' preserves old behavior (only most-similar + manual with cosine >= --escalate-above)."
        ),
    )
    ap.add_argument("--judge-temperature", type=float, default=0.0)
    ap.add_argument("--judge-system-message", type=str, default=None)

    args = ap.parse_args()

    from llmsat.utils.team_batch_io import load_team_strategies_from_batch_dir

    rows = load_team_strategies_from_batch_dir(args.team_batch_dir)
    leaders = [r for r in rows if r.get("type") == "leader"]
    if len(leaders) < 2:
        raise SystemExit("Need at least 2 leaders")

    leader_ids = [str(r.get("id")) for r in leaders]
    by_id = {str(r.get("id")): r for r in leaders}

    texts = [_leader_text(by_id[lid], field=str(args.text_field)) for lid in leader_ids]

    emb = _embed_qwen3_hf(
        texts=texts,
        model_id=str(args.qwen_model),
        batch_size=int(args.qwen_batch_size),
        max_length=int(args.qwen_max_length),
        dtype=str(args.qwen_dtype),
        trust_remote_code=bool(args.qwen_trust_remote_code),
        require_cuda=not bool(args.allow_cpu),
    )

    # cosine similarity since embeddings are normalized
    sim = emb @ emb.T

    best, worst = _upper_triangle_extrema(sim)
    top_sim = _topk_pairs(sim, k=int(args.top_k), largest=True)
    top_diff = _topk_pairs(sim, k=int(args.top_k), largest=False)

    def pack_pair(i: int, j: int, s: float) -> dict:
        a = leader_ids[i]
        b = leader_ids[j]
        return {
            "pair_id": _pair_id(a, b),
            "a_id": a,
            "b_id": b,
            "cosine": float(s),
            "a_name": str((by_id[a].get("spec") or {}).get("name", "")),
            "b_name": str((by_id[b].get("spec") or {}).get("name", "")),
        }

    report: Dict[str, Any] = {
        "team_batch_dir": str(args.team_batch_dir),
        "qwen_model": str(args.qwen_model),
        "text_field": str(args.text_field),
        "counts": {"leaders": len(leaders)},
        "extrema": {
            "most_similar": pack_pair(*best),
            "most_different": pack_pair(*worst),
        },
        "top_pairs": {
            "most_similar": [pack_pair(i, j, s) for (i, j, s) in top_sim],
            "most_different": [pack_pair(i, j, s) for (i, j, s) in top_diff],
        },
        "manual_pairs": [],
        "distribution": {
            "included": bool(args.include_distribution),
            "dump_all_pairs": bool(args.dump_all_pairs),
            "dump_all_pairs_max": int(args.dump_all_pairs_max),
        },
        "judge": {
            "enabled": bool(args.judge),
            "model": str(args.judge_model),
            "scope": str(args.judge_scope),
            "escalate_above": float(args.escalate_above),
        },
    }

    # Optional: include distribution summary and/or all pairs
    if args.include_distribution or args.dump_all_pairs:
        all_pairs = _all_pairs(sim)
        cosines = [s for (_, _, s) in all_pairs]

        if args.include_distribution:
            pct_list = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
            report["distribution"].update(
                {
                    "pair_count": int(len(cosines)),
                    "min": float(np.min(cosines)) if cosines else float("nan"),
                    "max": float(np.max(cosines)) if cosines else float("nan"),
                    "mean": float(np.mean(cosines)) if cosines else float("nan"),
                    "std": float(np.std(cosines)) if cosines else float("nan"),
                    "percentiles": _cosine_percentiles(cosines, pct_list),
                }
            )

        if args.dump_all_pairs:
            if len(cosines) <= int(args.dump_all_pairs_max):
                packed = [pack_pair(i, j, s) for (i, j, s) in all_pairs]
                packed.sort(key=lambda p: float(p.get("cosine", 0.0)), reverse=True)
                report["distribution"]["all_pairs_sorted_by_cosine_desc"] = packed
            else:
                report["distribution"]["all_pairs_skipped"] = True
                report["distribution"]["all_pairs_skip_reason"] = (
                    f"pair_count={len(cosines)} exceeds dump_all_pairs_max={int(args.dump_all_pairs_max)}"
                )

    if args.pair:
        for a_id, b_id in args.pair:
            if a_id not in by_id or b_id not in by_id:
                report["manual_pairs"].append({
                    "a_id": a_id,
                    "b_id": b_id,
                    "error": "unknown leader id",
                })
                continue
            i = leader_ids.index(a_id)
            j = leader_ids.index(b_id)
            report["manual_pairs"].append(pack_pair(min(i, j), max(i, j), float(sim[i, j])))

    # Optional judge pass
    if args.judge:
        from llmsat.utils.chatgpt_helper import get_response_from_chatgpt

        template = _read_text(args.judge_prompt)

        def should_judge(pair: dict) -> bool:
            if str(args.judge_scope) == "suspect":
                return float(pair.get("cosine", 0.0)) >= float(args.escalate_above)
            return True

        def judge_pair(pair: dict) -> dict:
            a = by_id[pair["a_id"]]
            b = by_id[pair["b_id"]]
            prompt = _render_prompt_template(template, a, b)
            raw = get_response_from_chatgpt(
                prompt=prompt,
                system_message=args.judge_system_message,
                model=str(args.judge_model),
                temperature=float(args.judge_temperature),
            )
            judged = _parse_judge_response(raw)
            return {
                "pair_id": pair["pair_id"],
                "similarity": judged["similarity"],
                "same_family": judged["same_family"],
                "rationale": judged["rationale"],
                "model": str(args.judge_model),
            }

        # Build a unique set of candidate pairs to judge.
        # Deduping avoids spending tokens twice when a pair appears in both extrema and top-k.
        candidates_by_id: Dict[str, dict] = {}

        def add_candidate(pair: dict) -> None:
            pid = pair.get("pair_id")
            if not isinstance(pid, str) or not pid:
                return
            candidates_by_id[pid] = pair

        scope = str(args.judge_scope)
        if scope in {"all", "top", "suspect"}:
            for pair in report["top_pairs"]["most_similar"]:
                add_candidate(pair)
        if scope in {"all", "top"}:
            for pair in report["top_pairs"]["most_different"]:
                add_candidate(pair)
        if scope in {"all", "extrema"}:
            add_candidate(report["extrema"]["most_similar"])
            add_candidate(report["extrema"]["most_different"])

        if scope in {"all", "top", "extrema", "suspect"}:
            for pair in report.get("manual_pairs", []):
                if "cosine" not in pair:
                    continue
                add_candidate(pair)

        # Judge in a stable order for reproducibility.
        candidates = list(candidates_by_id.values())
        candidates.sort(key=lambda p: (-float(p.get("cosine", 0.0)), str(p.get("pair_id", ""))))

        judged_records: List[dict] = []
        for pair in candidates:
            if not should_judge(pair):
                continue
            judged_records.append(judge_pair(pair))

        report["judge"]["results"] = judged_records

    # default output path
    if args.out is None:
        tag = args.team_batch_dir.parent.name
        out = Path("outputs") / tag / "calibration" / "qwen8b_leaders.json"
    else:
        out = args.out

    _write_json(out, report)
    print(f"Wrote calibration report to: {out}")
    print(f"Most similar leaders cosine: {report['extrema']['most_similar']['cosine']:.4f}")
    print(f"Most different leaders cosine: {report['extrema']['most_different']['cosine']:.4f}")

    if report.get("distribution", {}).get("included"):
        p = report["distribution"].get("percentiles") or {}
        if "50" in p and "90" in p and "99" in p:
            print(
                "Leader-leader cosine percentiles: "
                f"p50={float(p['50']):.4f} p90={float(p['90']):.4f} p99={float(p['99']):.4f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
