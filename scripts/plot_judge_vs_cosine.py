#!/usr/bin/env python3

"""Visualize correspondence between embedding cosine similarity and LLM judge similarity.

Use this after running scripts/calibrate_leader_similarity.py with --judge.
It turns the calibrator JSON output into quick, presentation-ready plots.

Expected input:
--in should point to a JSON file that contains:
- Per-pair cosine similarity keyed by "pair_id" (typically under extrema/top_pairs)
- LLM judge results under report["judge"]["results"], where each result has:
    {"pair_id": ..., "similarity": <0..1>, "same_family": <bool>, ...}

Outputs:
Writes to --out-dir:
- hist_judge_similarity.png           (distribution of LLM judge similarity)
- hist_cosine.png                    (distribution of embedding cosine, for judged pairs)
- scatter_cosine_vs_judge.png         (cosine vs judge, with y=x reference line)
- hist_delta_cos_minus_judge.png      (distribution of cosine - judge)
- judge_vs_cosine.csv                 (tidy table: pair_id, cosine, judge_similarity, ...)
- summary.json                        (percentiles + correlation stats)

Interpretation tips:
- Cosine scores from large embedding models can be "compressed" (many pairs near 0.9+).
    The scatter + delta plots make it obvious when cosine overestimates similarity.
- The number of judged rows is not necessarily "all pairs"; it depends on how the
    calibrator selected pairs (top-k/extrema/etc) and on how many leaders exist.
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PairDatum:
    pair_id: str
    cosine: float


def _safe_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except Exception:
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _iter_pair_like(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _iter_pair_like(v)
        return
    if isinstance(obj, list):
        for item in obj:
            yield from _iter_pair_like(item)
        return


def _extract_pair_cosines(report: Dict[str, Any]) -> Dict[str, float]:
    """Best-effort extraction of {pair_id -> cosine} from a calibrator JSON.

    This walks the JSON and collects any dict that looks like
    {"pair_id": str, "cosine": number}.
    """

    pair_to_cosine: Dict[str, float] = {}
    for item in _iter_pair_like(report):
        if not isinstance(item, dict):
            continue
        pair_id = item.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            continue
        cosine = _safe_float(item.get("cosine"))
        if cosine is None:
            continue
        # Keep the first seen value; if multiple disagree, prefer max (more conservative for "similar").
        if pair_id in pair_to_cosine:
            pair_to_cosine[pair_id] = max(pair_to_cosine[pair_id], cosine)
        else:
            pair_to_cosine[pair_id] = cosine

    return pair_to_cosine


def _extract_judge_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    judge = report.get("judge")
    if not isinstance(judge, dict):
        return []
    results = judge.get("results")
    if not isinstance(results, list):
        return []

    rows: List[Dict[str, Any]] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        pair_id = r.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            continue
        similarity = _safe_float(r.get("similarity"))
        if similarity is None:
            continue
        rows.append(
            {
                "pair_id": pair_id,
                "judge_similarity": similarity,
                "same_family": bool(r.get("same_family")) if "same_family" in r else None,
                "model": r.get("model"),
            }
        )

    return rows


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x0 = x - x.mean()
    y0 = y - y.mean()
    denom = float(np.sqrt((x0**2).sum()) * np.sqrt((y0**2).sum()))
    if denom == 0:
        return float("nan")
    return float((x0 * y0).sum() / denom)


def _spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return _pearsonr(xr, yr)


def _percentiles(values: np.ndarray, percentiles: List[int]) -> Dict[str, float]:
    if len(values) == 0:
        return {str(p): float("nan") for p in percentiles}
    out: Dict[str, float] = {}
    for p in percentiles:
        out[str(p)] = float(np.percentile(values, p))
    return out


def _plot_hist(values: np.ndarray, *, title: str, xlabel: str, out_path: str, bins: int = 20) -> None:
    plt.figure(figsize=(7, 4.2), dpi=160)
    plt.hist(values, bins=bins, color="#4C78A8", alpha=0.9, edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_scatter(
    cosine: np.ndarray,
    judge: np.ndarray,
    *,
    title: str,
    out_path: str,
    same_family: Optional[np.ndarray] = None,
) -> None:
    plt.figure(figsize=(5.6, 5.2), dpi=160)

    if same_family is None:
        plt.scatter(cosine, judge, s=28, alpha=0.85)
    else:
        same_family = same_family.astype("object")
        mask_true = same_family == True
        mask_false = same_family == False
        if mask_true.any():
            plt.scatter(cosine[mask_true], judge[mask_true], s=30, alpha=0.85, label="same_family=True")
        if mask_false.any():
            plt.scatter(cosine[mask_false], judge[mask_false], s=30, alpha=0.85, label="same_family=False")
        if (mask_true.any() or mask_false.any()):
            plt.legend(loc="lower right")

    lo = float(min(cosine.min(), judge.min()))
    hi = float(max(cosine.max(), judge.max()))
    pad = 0.02
    lo = max(0.0, lo - pad)
    hi = min(1.0, hi + pad)

    xs = np.linspace(lo, hi, 200)
    plt.plot(xs, xs, linestyle="--", linewidth=1.2, color="#888888", label="y=x")

    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.title(title)
    plt.xlabel("cosine (Qwen3-Embedding-8B)")
    plt.ylabel("LLM judge similarity")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot correspondence between embedding cosine and LLM judge similarity from a calibrator output JSON."
    )
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSON (e.g. qwen8b_leaders_judged_all.json)")
    ap.add_argument("--out-dir", required=True, help="Directory to write plots + CSV/JSON summary")
    ap.add_argument("--bins", type=int, default=20, help="Histogram bins")
    ap.add_argument("--title", default=None, help="Optional plot title prefix")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.in_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    pair_to_cosine = _extract_pair_cosines(report)
    judge_rows = _extract_judge_rows(report)

    df = pd.DataFrame(judge_rows)
    if df.empty:
        raise SystemExit("No judge results found in input JSON.")

    df["cosine"] = df["pair_id"].map(pair_to_cosine)
    missing = df["cosine"].isna().sum()
    if missing:
        # Still plot the judge distribution even if cosine mapping is incomplete.
        pass

    df_full = df.dropna(subset=["cosine"]).copy()
    df_full["delta_cos_minus_judge"] = df_full["cosine"] - df_full["judge_similarity"]

    title_prefix = (args.title.strip() + " — ") if isinstance(args.title, str) and args.title.strip() else ""

    # 1) Distribution of judge score
    _plot_hist(
        df["judge_similarity"].to_numpy(dtype=float),
        title=f"{title_prefix}LLM judge similarity distribution (n={len(df)})",
        xlabel="judge similarity",
        out_path=os.path.join(args.out_dir, "hist_judge_similarity.png"),
        bins=args.bins,
    )

    # 2) If we can map to cosine, add more plots
    if not df_full.empty:
        _plot_hist(
            df_full["cosine"].to_numpy(dtype=float),
            title=f"{title_prefix}Cosine similarity distribution (n={len(df_full)})",
            xlabel="cosine",
            out_path=os.path.join(args.out_dir, "hist_cosine.png"),
            bins=args.bins,
        )

        _plot_hist(
            df_full["delta_cos_minus_judge"].to_numpy(dtype=float),
            title=f"{title_prefix}Cosine − judge similarity (n={len(df_full)})",
            xlabel="delta (cosine - judge)",
            out_path=os.path.join(args.out_dir, "hist_delta_cos_minus_judge.png"),
            bins=args.bins,
        )

        same_family_arr: Optional[np.ndarray] = None
        if "same_family" in df_full.columns:
            # Keep None values as None.
            same_family_arr = df_full["same_family"].to_numpy()

        _plot_scatter(
            df_full["cosine"].to_numpy(dtype=float),
            df_full["judge_similarity"].to_numpy(dtype=float),
            title=f"{title_prefix}Cosine vs LLM judge (n={len(df_full)})",
            out_path=os.path.join(args.out_dir, "scatter_cosine_vs_judge.png"),
            same_family=same_family_arr,
        )

    # Save a tidy CSV and a small numeric summary JSON.
    csv_path = os.path.join(args.out_dir, "judge_vs_cosine.csv")
    df.to_csv(csv_path, index=False)

    summary: Dict[str, Any] = {
        "input": os.path.relpath(args.in_path),
        "n_judged": int(len(df)),
        "n_with_cosine": int(len(df_full)),
        "judge_similarity": {
            "percentiles": _percentiles(df["judge_similarity"].to_numpy(dtype=float), [0, 5, 10, 25, 50, 75, 90, 95, 100]),
            "mean": float(df["judge_similarity"].mean()),
        },
    }

    if not df_full.empty:
        x = df_full["cosine"].to_numpy(dtype=float)
        y = df_full["judge_similarity"].to_numpy(dtype=float)
        d = df_full["delta_cos_minus_judge"].to_numpy(dtype=float)
        summary.update(
            {
                "cosine": {
                    "percentiles": _percentiles(x, [0, 5, 10, 25, 50, 75, 90, 95, 100]),
                    "mean": float(x.mean()),
                },
                "delta_cos_minus_judge": {
                    "percentiles": _percentiles(d, [0, 5, 10, 25, 50, 75, 90, 95, 100]),
                    "mean": float(d.mean()),
                },
                "correlation": {
                    "pearson": _pearsonr(x, y),
                    "spearman": _spearmanr(x, y),
                },
            }
        )

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"Wrote plots + data to: {args.out_dir}")
    print(f"Rows judged: {len(df)}; with cosine mapped: {len(df_full)}")


if __name__ == "__main__":
    main()
