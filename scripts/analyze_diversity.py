#!/usr/bin/env python3
"""Analyze embedding/cosine diversity of generated strategies.

Use this when you have a set of generated strategies (leaders + members) and you
want to quantify how similar they are under an embedding model.

Inputs:
Exactly one of:
- --team-batch-dir: a directory like outputs/<tag>/batch_<batch_id>/ produced by
    the team generation pipeline (leaders_output.txt / member_output_batch_*.txt).
- --in: a JSONL file of strategy rows (legacy mode).

Embeddings:
--embedding chooses the backend:
- auto: prefer sentence-transformers, fall back to TF-IDF
- st: sentence-transformers (fast on CPU)
- tfidf: TF-IDF baseline (fast on CPU; good smoke test)
- qwen3: Hugging Face Qwen3 embedding model (Qwen3-Embedding-8B typically needs GPU)

Metrics produced:
Computes cosine similarity distributions for:
- leader_leader
- leader_member_within_team
- member_member_within_team
- member_member_cross_team
- member_member_all
and reports near-duplicate pairs above --duplicate-threshold.

Outputs:
Writes two files under --out-dir:
- diversity_report.json : summary stats + duplicate pairs
- strategy_summary.csv  : one row per strategy (without full text)

Examples:
CPU smoke test (fast, no GPU):
    python scripts/analyze_diversity.py \
        --team-batch-dir outputs/controlled_mutation/batch_batch_<...> \
        --out-dir outputs/controlled_mutation/diversity_tfidf \
        --embedding tfidf

GPU run (Qwen3-Embedding-8B) on Slurm:
    sbatch scripts/slurm_analyze_diversity_qwen8b.sh \
        outputs/controlled_mutation/batch_batch_<...> \
        outputs/controlled_mutation/diversity_qwen8b \
        --duplicate-threshold 0.97
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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


def _describe(values: List[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p10": None,
            "p90": None,
        }
    arr = np.array(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


@dataclass
class EmbeddingResult:
    method: str
    matrix: np.ndarray  # shape (n, d)


def _embed_texts(texts: List[str], prefer_sentence_transformers: bool = True) -> EmbeddingResult:
    if prefer_sentence_transformers:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            model = SentenceTransformer("all-MiniLM-L6-v2")
            mat = np.array(model.encode(texts, normalize_embeddings=True, show_progress_bar=True))
            return EmbeddingResult(method="sentence-transformers/all-MiniLM-L6-v2", matrix=mat)
        except Exception:
            pass

    vec = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        lowercase=True,
        stop_words=None,
    )
    X = vec.fit_transform(texts)
    # Normalize to unit length for cosine via dot.
    norms = np.sqrt(X.multiply(X).sum(axis=1)).A1
    norms[norms == 0] = 1.0
    Xn = X.multiply(1.0 / norms[:, None])
    return EmbeddingResult(method="tfidf(1-2gram,max=5000)", matrix=Xn.toarray())


def _embed_qwen3(
    texts: List[str],
    model_id: str,
    batch_size: int = 16,
    max_length: int = 512,
    trust_remote_code: bool = False,
    dtype: str = "float32",
    require_cuda: bool = False,
) -> EmbeddingResult:
    """Embed texts using a Qwen3 embedding model via Hugging Face transformers.

    This expects a model that can be loaded with AutoModel and returns last_hidden_state.
    Embedding is computed by mean pooling over tokens (masked) and then L2-normalized.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Run on a GPU node or omit --qwen-require-cuda.")

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

    all_vecs: List[np.ndarray] = []
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
            all_vecs.append(pooled.detach().cpu().numpy())

    mat = np.concatenate(all_vecs, axis=0)
    return EmbeddingResult(method=f"qwen3:{model_id}", matrix=mat)


def _upper_triangle_pairs(sim: np.ndarray) -> List[float]:
    n = sim.shape[0]
    out: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            out.append(float(sim[i, j]))
    return out


def _top_pairs(
    ids: List[str],
    sim: np.ndarray,
    threshold: float,
    max_pairs: int,
    allow_same: bool = False,
) -> List[dict]:
    pairs: List[Tuple[float, str, str]] = []
    n = sim.shape[0]
    for i in range(n):
        for j in range(n):
            if not allow_same and j <= i:
                continue
            s = float(sim[i, j])
            if s >= threshold:
                pairs.append((s, ids[i], ids[j]))
    pairs.sort(reverse=True, key=lambda t: t[0])
    out = [{"similarity": s, "a": a, "b": b} for s, a, b in pairs[:max_pairs]]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path, required=False, help="Input strategies.jsonl")
    ap.add_argument(
        "--team-batch-dir",
        type=Path,
        default=None,
        help="Load strategies from outputs/<tag>/batch_<leader_batch_id>/ produced by generate_team_data",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/diversity"),
        help="Output directory",
    )
    ap.add_argument("--duplicate-threshold", type=float, default=0.95)
    ap.add_argument("--max-duplicate-pairs", type=int, default=50)
    ap.add_argument(
        "--embedding",
        choices=["auto", "tfidf", "st", "qwen3"],
        default="auto",
        help="Embedding backend for cosine similarity",
    )
    ap.add_argument(
        "--qwen-model",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Hugging Face model id/path for Qwen3 embedding",
    )
    ap.add_argument("--qwen-batch-size", type=int, default=16)
    ap.add_argument("--qwen-max-length", type=int, default=512)
    ap.add_argument(
        "--qwen-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float32",
        help="Torch dtype for Qwen3 embeddings (use float16/bfloat16 for large models like 8B)",
    )
    ap.add_argument(
        "--qwen-require-cuda",
        action="store_true",
        help="Fail fast if CUDA is not available (recommended for Qwen3-Embedding-8B)",
    )
    ap.add_argument(
        "--qwen-trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the Qwen model",
    )

    args = ap.parse_args()

    if args.team_batch_dir is None and args.in_path is None:
        raise SystemExit("Must pass --in strategies.jsonl or --team-batch-dir")
    if args.team_batch_dir is not None and args.in_path is not None:
        raise SystemExit("Pass only one of --in or --team-batch-dir")

    if args.team_batch_dir is not None:
        from llmsat.utils.team_batch_io import load_team_strategies_from_batch_dir

        rows = load_team_strategies_from_batch_dir(args.team_batch_dir)
    else:
        rows = _read_jsonl(args.in_path)
    if not rows:
        raise SystemExit("No rows read from input")

    leaders = [r for r in rows if r.get("type") == "leader"]
    members = [r for r in rows if r.get("type") == "member"]

    # Build per-strategy table
    all_rows = []
    for r in rows:
        spec = r.get("spec") or {}
        meta = r.get("meta") or {}
        all_rows.append(
            {
                "id": r.get("id"),
                "type": r.get("type"),
                "leader_id": r.get("leader_id"),
                "target_function": r.get("target_function"),
                "name": spec.get("name"),
                "temperature": meta.get("temperature"),
                "model": meta.get("model"),
                "created_at": meta.get("created_at"),
                "strategy_text": r.get("strategy_text") or "",
            }
        )

    df = pd.DataFrame(all_rows)
    df["strategy_text"] = df["strategy_text"].fillna("")

    texts = df["strategy_text"].tolist()
    if args.embedding == "tfidf":
        emb = _embed_texts(texts, prefer_sentence_transformers=False)
    elif args.embedding == "st":
        emb = _embed_texts(texts, prefer_sentence_transformers=True)
        if not emb.method.startswith("sentence-transformers/"):
            raise SystemExit(
                "sentence-transformers embedding requested, but it is not available. Install 'sentence-transformers' or use --embedding tfidf/qwen3."
            )
    elif args.embedding == "qwen3":
        emb = _embed_qwen3(
            texts,
            model_id=str(args.qwen_model),
            batch_size=int(args.qwen_batch_size),
            max_length=int(args.qwen_max_length),
            trust_remote_code=bool(args.qwen_trust_remote_code),
            dtype=str(args.qwen_dtype),
            require_cuda=bool(args.qwen_require_cuda),
        )
    else:
        # auto
        emb = _embed_texts(texts, prefer_sentence_transformers=True)

    sim_all = cosine_similarity(emb.matrix)

    # Similarities by group
    id_list = df["id"].astype(str).tolist()
    idx_by_id = {sid: i for i, sid in enumerate(id_list)}

    leader_ids = [str(r["id"]) for r in leaders]
    member_ids = [str(r["id"]) for r in members]

    leader_idx = [idx_by_id[i] for i in leader_ids if i in idx_by_id]
    member_idx = [idx_by_id[i] for i in member_ids if i in idx_by_id]

    leader_leader_vals: List[float] = []
    if leader_idx:
        leader_sim = sim_all[np.ix_(leader_idx, leader_idx)]
        leader_leader_vals = _upper_triangle_pairs(leader_sim)

    # leader-member within team
    leader_member_vals: List[float] = []
    members_by_leader: Dict[str, List[str]] = defaultdict(list)
    for m in members:
        members_by_leader[str(m["leader_id"])].append(str(m["id"]))

    for lid in leader_ids:
        if lid not in idx_by_id:
            continue
        li = idx_by_id[lid]
        for mid in members_by_leader.get(lid, []):
            mi = idx_by_id.get(mid)
            if mi is None:
                continue
            leader_member_vals.append(float(sim_all[li, mi]))

    # member-member across teams
    member_member_cross_vals: List[float] = []
    # Build list of (member_id, leader_id)
    member_pairs = [(str(m["id"]), str(m["leader_id"])) for m in members]
    for i in range(len(member_pairs)):
        mid_a, lid_a = member_pairs[i]
        ia = idx_by_id.get(mid_a)
        if ia is None:
            continue
        for j in range(i + 1, len(member_pairs)):
            mid_b, lid_b = member_pairs[j]
            if lid_a == lid_b:
                continue
            ib = idx_by_id.get(mid_b)
            if ib is None:
                continue
            member_member_cross_vals.append(float(sim_all[ia, ib]))

    # member-member within the same team (same leader)
    member_member_within_vals: List[float] = []
    for lid, mids in members_by_leader.items():
        idxs: List[int] = []
        for mid in mids:
            mi = idx_by_id.get(mid)
            if mi is not None:
                idxs.append(mi)
        if len(idxs) < 2:
            continue
        sub = sim_all[np.ix_(idxs, idxs)]
        member_member_within_vals.extend(_upper_triangle_pairs(sub))

    # member-member overall (includes within + cross)
    member_member_all_vals: List[float] = []
    if member_idx and len(member_idx) >= 2:
        member_sim = sim_all[np.ix_(member_idx, member_idx)]
        member_member_all_vals = _upper_triangle_pairs(member_sim)

    # Duplicates on full set
    dup_pairs = _top_pairs(
        ids=id_list,
        sim=sim_all,
        threshold=float(args.duplicate_threshold),
        max_pairs=int(args.max_duplicate_pairs),
        allow_same=False,
    )

    report = {
        "input": str(args.in_path) if args.in_path is not None else None,
        "team_batch_dir": str(args.team_batch_dir) if args.team_batch_dir is not None else None,
        "counts": {
            "total": int(len(rows)),
            "leaders": int(len(leaders)),
            "members": int(len(members)),
        },
        "embedding": {"method": emb.method, "dim": int(emb.matrix.shape[1])},
        "similarity": {
            "leader_leader": _describe(leader_leader_vals),
            "leader_member_within_team": _describe(leader_member_vals),
            "member_member_within_team": _describe(member_member_within_vals),
            "member_member_cross_team": _describe(member_member_cross_vals),
            "member_member_all": _describe(member_member_all_vals),
        },
        "duplicates": {
            "threshold": float(args.duplicate_threshold),
            "pairs": dup_pairs,
        },
    }

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "diversity_report.json"
    report_path.write_text(_stable_json_dumps(report) + "\n", encoding="utf-8")

    csv_path = out_dir / "strategy_summary.csv"
    df.drop(columns=["strategy_text"]).to_csv(csv_path, index=False)

    print(f"Wrote: {report_path}")
    print(f"Wrote: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
