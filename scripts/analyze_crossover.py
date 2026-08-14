"""Analyze whether crossover offspring improved on their parents' PAR-2.

Three ways to use:

  1. Discover crossover offspring directly from the DB, no local files needed.
     Scans all algorithms; an algorithm is treated as a crossover offspring
     iff its `parent_id` list has ≥2 entries (mutation has 1, init has 0).

      PYTHONPATH=src python scripts/analyze_crossover.py --db-scan
      PYTHONPATH=src python scripts/analyze_crossover.py --db-scan --tag-pattern '_ge'

  2. Pass paths to one or more `crossover_results_*.json` files directly:

      PYTHONPATH=src python scripts/analyze_crossover.py \
        outputs/foo/crossover_results_iter1.json \
        outputs/bar/crossover_results_iter1.json

  3. Scan all crossover_results_*.json under outputs/:

      PYTHONPATH=src python scripts/analyze_crossover.py --scan

Output: a per-offspring table + aggregate stats. Skips offspring with no
PAR-2 yet (status `code_generated`). For each evaluated offspring, looks up
its raw PAR-2 from the DB along with its parents' PAR-2, computes
Δ = offspring_par2 - min(parent_par2). Negative Δ ⇒ crossover helped.

Pass `--save out.json` to also dump the raw paired data so a colleague can
share it without DB access.

To consume someone else's saved data offline (no DB needed):

      python scripts/analyze_crossover.py --load colleague_data.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from glob import glob
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _flatten_par2(p):
    """raw_par2_score in the DB can be a scalar or a list (per-fold). Return mean or None."""
    if p is None:
        return None
    if isinstance(p, (int, float)):
        return float(p)
    if isinstance(p, list):
        vals = [v for v in p if isinstance(v, (int, float))]
        return sum(vals) / len(vals) if vals else None
    return None


def fetch_db(algo_id: str, cache: dict):
    """Look up an algorithm's raw_par2_score + status from the DB. Cached."""
    if algo_id in cache:
        return cache[algo_id]
    try:
        from llmsat.utils.aws import get_algorithm_result
        a = get_algorithm_result(algo_id)
    except Exception as e:
        cache[algo_id] = {"error": str(e), "par2": None, "status": None}
        return cache[algo_id]
    if a is None:
        cache[algo_id] = {"par2": None, "status": "missing"}
        return cache[algo_id]
    rec = {
        "par2": _flatten_par2(getattr(a, "raw_par2_score", None)),
        "status": str(getattr(a, "status", None)),
    }
    cache[algo_id] = rec
    return rec


def collect_crossover_entries(paths):
    """Read crossover_results JSON files; return list of (tag, child_id, a_id, b_id, target_fn)."""
    out = []
    for p in paths:
        try:
            data = json.load(open(p))
        except Exception as e:
            print(f"WARNING: could not read {p}: {e}", file=sys.stderr)
            continue
        # tag = parent dir of the file (e.g. outputs/phase_decide_ge/crossover_results_iter1.json -> phase_decide_ge)
        tag = Path(p).parent.name
        if not isinstance(data, list):
            print(f"WARNING: {p} is not a list — skipping", file=sys.stderr)
            continue
        for c in data:
            out.append({
                "tag": tag,
                "source_file": str(p),
                "child_id": c["algorithm_id"],
                "parent_a_id": c.get("parent_a_id"),
                "parent_b_id": c.get("parent_b_id"),
                "target_function": c.get("target_function"),
            })
    return out


def build_paired_data(entries):
    """For each crossover entry, fetch DB par2 for child + parents. Returns list of dicts."""
    cache: dict = {}
    paired = []
    for e in entries:
        child = fetch_db(e["child_id"], cache)
        a = fetch_db(e["parent_a_id"], cache) if e["parent_a_id"] else {"par2": None}
        b = fetch_db(e["parent_b_id"], cache) if e["parent_b_id"] else {"par2": None}
        parents = [p for p in (a["par2"], b["par2"]) if p is not None]
        best_parent = min(parents) if parents else None
        delta = (child["par2"] - best_parent) if (child["par2"] is not None and best_parent is not None) else None
        delta_pct = (100.0 * delta / best_parent) if (delta is not None and best_parent and best_parent > 0) else None
        paired.append({
            **e,
            "child_par2": child["par2"],
            "child_status": child.get("status"),
            "parent_a_par2": a["par2"],
            "parent_b_par2": b["par2"],
            "best_parent_par2": best_parent,
            "delta": delta,
            "delta_pct": delta_pct,
        })
    return paired


def summarize(paired):
    """Print per-entry table + aggregate."""
    n = len(paired)
    n_evaluated = sum(1 for p in paired if p["child_par2"] is not None)
    n_paired_full = sum(1 for p in paired if p["delta"] is not None)
    print()
    print(f"Total crossover offspring scanned:                 {n}")
    print(f"Offspring with a PAR-2 score (i.e. evaluated):     {n_evaluated}")
    print(f"Offspring with PAR-2 AND ≥1 parent PAR-2 (paired): {n_paired_full}")

    print()
    print(f"{'Tag':<26} {'Child':<14} {'Status':<22} {'Child PAR2':>11} {'Best parent':>12} {'Δ':>9} {'Δ %':>8}")
    print("-" * 110)
    for p in paired:
        cid = p["child_id"][:12]
        status = (p["child_status"] or "—")[:20]
        child_s   = f"{p['child_par2']:.2f}"        if p["child_par2"]        is not None else "—"
        best_s    = f"{p['best_parent_par2']:.2f}"  if p["best_parent_par2"]  is not None else "—"
        delta_s   = f"{p['delta']:+.2f}"            if p["delta"]             is not None else "—"
        pct_s     = f"{p['delta_pct']:+.2f}%"       if p["delta_pct"]         is not None else "—"
        print(f"{p['tag'][:25]:<26} {cid:<14} {status:<22} {child_s:>11} {best_s:>12} {delta_s:>9} {pct_s:>8}")

    deltas = [p["delta"] for p in paired if p["delta"] is not None]
    if not deltas:
        print()
        print("No paired data to aggregate. Either no offspring were ever evaluated,")
        print("or the parents have no PAR-2 either. Pipeline justification:")
        print("  - Crossover children require the same compute budget per eval as a mutation iter")
        print("  - Without runs, there's no evidence they improve PAR-2")
        print("  - Mutation iterations have measurable improvements per iter (median Δ ≈ -10 to -20 PAR2 per team)")
        return

    n_better  = sum(1 for d in deltas if d < 0)
    n_worse   = sum(1 for d in deltas if d > 0)
    n_tie     = sum(1 for d in deltas if d == 0)
    pcts = [p["delta_pct"] for p in paired if p["delta_pct"] is not None]

    print()
    print(f"=== Aggregate (n={len(deltas)} paired evaluations) ===")
    print(f"  Children that BEAT best parent (Δ < 0):  {n_better}/{len(deltas)}  ({100*n_better/len(deltas):.1f}%)")
    print(f"  Children that TIED  (Δ = 0):             {n_tie}/{len(deltas)}")
    print(f"  Children WORSE than best parent (Δ > 0): {n_worse}/{len(deltas)}  ({100*n_worse/len(deltas):.1f}%)")
    print(f"  Mean Δ:    {statistics.mean(deltas):+.2f}")
    print(f"  Median Δ:  {statistics.median(deltas):+.2f}")
    print(f"  Stdev Δ:   {statistics.stdev(deltas):.2f}" if len(deltas) > 1 else "")
    print(f"  Range Δ:   [{min(deltas):+.2f}, {max(deltas):+.2f}]")
    if pcts:
        print(f"  Mean Δ%:   {statistics.mean(pcts):+.2f}%   (negative = crossover improved over best parent)")


def discover_crossover_from_db(tag_pattern: str | None = None):
    """Query the DB for all algorithms whose parent_id list has ≥ 2 entries
    (the structural marker for crossover lineage). Optionally filter by a
    substring `tag_pattern` matched against the algorithm's generation tag
    (e.g. '_ge' to limit to bridge runs)."""
    from llmsat.utils.aws import connect_to_db, release_conn
    from llmsat.llmsat import CHATGPT_DATA_GENERATION_TABLE

    # Find all distinct tags in the router table — used to associate each algo with its tag
    conn = connect_to_db()
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT DISTINCT type FROM {CHATGPT_DATA_GENERATION_TABLE};")
        tags = sorted({row[0] for row in cur.fetchall() if row[0]})
        # Build id → tag map (one query)
        cur.execute(f"SELECT id, type FROM {CHATGPT_DATA_GENERATION_TABLE};")
        id_to_tag = {row[0]: row[1] for row in cur.fetchall()}
    finally:
        release_conn(conn)

    print(f"DB has {len(tags)} distinct generation tags, {len(id_to_tag)} (algo, tag) rows")

    # Now iterate all algorithms and pick those with len(parent_id) >= 2
    from llmsat.utils.aws import get_all_algorithm_results
    print("Fetching all algorithm_results...")
    all_algos = get_all_algorithm_results()
    print(f"Got {len(all_algos)} algorithm_result rows")

    entries = []
    for a in all_algos:
        pid = getattr(a, "parent_id", None) or []
        if not isinstance(pid, list) or len(pid) < 2:
            continue
        algo_id = a.id
        tag = id_to_tag.get(algo_id)
        if tag_pattern and (not tag or tag_pattern not in tag):
            continue
        entries.append({
            "tag": tag or "(no tag in router)",
            "source_file": "<db-scan>",
            "child_id": algo_id,
            "parent_a_id": pid[0],
            "parent_b_id": pid[1],
            "target_function": getattr(a, "function_name", None),
            # extra parents if any (e.g. 3-way crossover)
            "extra_parents": pid[2:] if len(pid) > 2 else [],
        })
    return entries


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="crossover_results_*.json paths")
    ap.add_argument("--scan", action="store_true",
                    help="Auto-discover all outputs/*/crossover_results_*.json files")
    ap.add_argument("--db-scan", action="store_true",
                    help="Discover crossover offspring directly from the DB "
                         "(any algorithm with >=2 parent_ids). Doesn't need local files.")
    ap.add_argument("--tag-pattern", default=None,
                    help="With --db-scan, only keep offspring whose generation tag contains this substring "
                         "(e.g. '_ge' to limit to run_bridge runs).")
    ap.add_argument("--save", default=None,
                    help="Also dump paired data to this JSON path (for sharing offline)")
    ap.add_argument("--load", default=None,
                    help="Skip DB; load already-saved paired data and just print summary")
    args = ap.parse_args()

    if args.load:
        paired = json.load(open(args.load))
        print(f"Loaded {len(paired)} paired entries from {args.load}")
        summarize(paired)
        return 0

    if args.db_scan:
        entries = discover_crossover_from_db(tag_pattern=args.tag_pattern)
        print(f"Discovered {len(entries)} crossover offspring in DB"
              + (f" (filter: tag contains '{args.tag_pattern}')" if args.tag_pattern else ""))
    else:
        if args.scan and not args.paths:
            args.paths = sorted(glob(str(REPO / "outputs/*/crossover_results_*.json")))
        if not args.paths:
            ap.error("Pass crossover_results_*.json paths, or use --scan / --db-scan / --load")
        print(f"Scanning {len(args.paths)} crossover_results files:")
        for p in args.paths:
            print(f"  - {p}")
        entries = collect_crossover_entries(args.paths)

    paired = build_paired_data(entries)

    if args.save:
        Path(args.save).write_text(json.dumps(paired, indent=2))
        print(f"\nSaved paired data → {args.save}")

    summarize(paired)
    return 0


if __name__ == "__main__":
    sys.exit(main())
