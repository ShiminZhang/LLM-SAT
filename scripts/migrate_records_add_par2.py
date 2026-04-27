"""Migrate mutation pool records.json files to include leader/member Par2Scores.

Looks up each record's leader_algorithm_id and member_algorithm_id in the AWS
algorithm_results table and writes the par2 breakdown back into the JSON
record under `leader_par2` / `member_par2`. Also seeds an empty `extra: {}`
field for forward compatibility.

Existing record_id keys are preserved verbatim (par2 fields are excluded from
the SHA256 identity hash, so keys remain canonical and the FAISS index/id_map
stay in sync).

Usage:
    source scripts/activate_llmsat_conda.sh
    DB_PASS="..." python scripts/migrate_records_add_par2.py [--dry-run] [--data-root <path>]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

# Make sibling `experience_pool` importable when run from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experience_pool.pools import _par2_from_raw_list  # noqa: E402
from llmsat.utils.aws import get_algorithm_result  # noqa: E402

DEFAULT_DATA_ROOT = SRC_DIR / "experience_pool" / "data"


def _lookup_par2(
    algo_id: Optional[str],
    cache: Dict[str, Optional[dict]],
) -> Tuple[Optional[dict], str]:
    """Return (par2_dict_or_None, miss_reason).

    miss_reason is "" on hit, otherwise one of: "id_null", "db_miss", "shape".
    """

    if not isinstance(algo_id, str) or not algo_id:
        return None, "id_null"

    if algo_id in cache:
        cached = cache[algo_id]
        if cached is None:
            return None, "db_miss_or_shape"
        return cached, ""

    try:
        algo = get_algorithm_result(algo_id)
    except Exception as exc:
        print(f"  [WARN] DB lookup failed for {algo_id[:12]}…: {exc}")
        cache[algo_id] = None
        return None, "db_miss_or_shape"

    if algo is None:
        cache[algo_id] = None
        return None, "db_miss"

    par2 = _par2_from_raw_list(algo.raw_par2_score)
    if par2 is None:
        cache[algo_id] = None
        return None, "shape"

    par2_dict = asdict(par2)
    cache[algo_id] = par2_dict
    return par2_dict, ""


def migrate_file(path: Path, dry_run: bool, cache: Dict[str, Optional[dict]]) -> dict:
    """Migrate a single records.json file. Returns per-file summary."""

    summary = {
        "path": str(path),
        "total": 0,
        "leader_hits": 0,
        "member_hits": 0,
        "leader_misses": {"id_null": 0, "db_miss_or_shape": 0, "db_miss": 0, "shape": 0},
        "member_misses": {"id_null": 0, "db_miss_or_shape": 0, "db_miss": 0, "shape": 0},
        "skipped_already_migrated": 0,
        "wrote": False,
    }
    t0 = time.time()

    raw = path.read_text(encoding="utf-8")
    records: Dict[str, dict] = json.loads(raw)
    summary["total"] = len(records)

    for record_id, rec in records.items():
        already = (
            "leader_par2" in rec
            and "member_par2" in rec
            and "extra" in rec
        )
        if already and rec.get("leader_par2") is not None and rec.get("member_par2") is not None:
            summary["skipped_already_migrated"] += 1
            continue

        leader_id = rec.get("leader_algorithm_id")
        member_id = rec.get("member_algorithm_id")

        leader_par2, leader_miss = _lookup_par2(leader_id, cache)
        member_par2, member_miss = _lookup_par2(member_id, cache)

        if leader_miss:
            summary["leader_misses"][leader_miss] = (
                summary["leader_misses"].get(leader_miss, 0) + 1
            )
        else:
            summary["leader_hits"] += 1

        if member_miss:
            summary["member_misses"][member_miss] = (
                summary["member_misses"].get(member_miss, 0) + 1
            )
        else:
            summary["member_hits"] += 1

        rec["leader_par2"] = leader_par2
        rec["member_par2"] = member_par2
        rec.setdefault("extra", {})

    summary["elapsed_s"] = round(time.time() - t0, 2)

    if dry_run:
        return summary

    bak = path.with_suffix(path.suffix + ".bak")
    if bak.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing backup at {bak}. "
            "Delete it manually if you really intend to re-run."
        )
    shutil.copy2(path, bak)
    path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary["wrote"] = True
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute par2 lookups and print summary without modifying files.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Root of experience_pool data dir (default: {DEFAULT_DATA_ROOT})",
    )
    args = parser.parse_args()

    pattern = "*/mutation/*/records.json"
    files = sorted(args.data_root.glob(pattern))
    if not files:
        print(f"No records.json files found under {args.data_root} matching {pattern}")
        return 1

    print(f"Found {len(files)} records.json file(s):")
    for f in files:
        print(f"  - {f}")
    print()

    cache: Dict[str, Optional[dict]] = {}
    overall = {
        "files": 0,
        "total": 0,
        "leader_hits": 0,
        "member_hits": 0,
        "skipped_already_migrated": 0,
    }

    for f in files:
        print(f"=== {f} ===")
        summary = migrate_file(f, dry_run=args.dry_run, cache=cache)
        overall["files"] += 1
        overall["total"] += summary["total"]
        overall["leader_hits"] += summary["leader_hits"]
        overall["member_hits"] += summary["member_hits"]
        overall["skipped_already_migrated"] += summary["skipped_already_migrated"]
        print(json.dumps(summary, indent=2))
        print()

    print("=== OVERALL ===")
    print(json.dumps(overall, indent=2))
    if args.dry_run:
        print("\n(dry-run: no files modified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
