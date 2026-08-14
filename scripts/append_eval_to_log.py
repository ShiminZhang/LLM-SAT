"""Append a row to outputs/full_eval_log.csv from a completed eval.

Usage:
  python scripts/append_eval_to_log.py \
    --solver B1 --run-label simul-2026-05-01-pair2 \
    --result-dir solvers/.../result_simul_run2 \
    --job-id 12345678 \
    [--generation-tag kissat_evolve_iter1] \
    [--algorithm-id-short cee194034a6f] \
    [--code-id-short 462fc8f76750]

Appends one row with: run_date (today), run_label, solver, algorithm_id_short,
code_id_short, generation_tag, job_id, result_dir, n_instances, solved,
timeouts, par2.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOG = REPO / "outputs/full_eval_log.csv"
PENALTY = 10000.0
FIELDS = ["run_date","run_label","solver","algorithm_id_short","code_id_short",
          "generation_tag","job_id","result_dir","n_instances","solved","timeouts","par2"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solver", required=True)
    ap.add_argument("--run-label", required=True)
    ap.add_argument("--result-dir", required=True, help="Path to the result dir containing solving_times.json (or solving_times_full.json)")
    ap.add_argument("--job-id", default="")
    ap.add_argument("--generation-tag", default="—")
    ap.add_argument("--algorithm-id-short", default="—")
    ap.add_argument("--code-id-short", default="—")
    ap.add_argument("--run-date", default=date.today().isoformat(), help="YYYY-MM-DD; defaults to today")
    args = ap.parse_args()

    rd = Path(args.result_dir)
    if not rd.is_absolute():
        rd = REPO / rd
    candidates = [rd / "solving_times.json", rd / "solving_times_full.json"]
    times_file = next((c for c in candidates if c.exists()), None)
    if not times_file:
        raise SystemExit(f"No solving_times[_full].json found under {rd}")

    times = json.load(open(times_file))
    n = len(times)
    solved = sum(1 for t in times.values() if t < PENALTY)
    par2 = round(sum(times.values()) / n, 2)

    LOG.parent.mkdir(parents=True, exist_ok=True)
    new_file = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        try:
            rd_rel = str(rd.relative_to(REPO))
        except ValueError:
            rd_rel = str(rd)
        w.writerow({
            "run_date": args.run_date,
            "run_label": args.run_label,
            "solver": args.solver,
            "algorithm_id_short": args.algorithm_id_short,
            "code_id_short": args.code_id_short,
            "generation_tag": args.generation_tag,
            "job_id": args.job_id,
            "result_dir": rd_rel,
            "n_instances": n,
            "solved": solved,
            "timeouts": n - solved,
            "par2": par2,
        })
    print(f"Appended {args.solver} ({args.run_label}) → solved={solved}/{n} timeouts={n-solved} par2={par2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
