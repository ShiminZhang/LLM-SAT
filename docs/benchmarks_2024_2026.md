# SAT Competition 2024 / 2026 Main-Track Benchmarks

Status as of 2026-08-14. Both benchmark sets are published and downloadable
(400 instances each, same format as the existing satcomp2025 set).

## Sources and endpoints

Every instance is hosted by the Global Benchmark Database (GBD,
benchmark-database.de) and addressed by MD5 hash:
`http://benchmark-database.de/file/<md5>` (301-redirects to https; served with
`Content-Disposition: attachment; filename=<md5>-<origname>.cnf.xz`).

Instance lists were obtained from:

- **2024**: GBD list endpoint (returns `text/uri-list`, one URL per line):

  ```
  https://benchmark-database.de/getinstances?query=track%3Dmain_2024&context=cnf
  ```

  400 URLs. The SAT Competition 2024 site's Downloads section is still a "tbd"
  placeholder (https://satcompetition.github.io/2024/downloads.html), so GBD is
  the only source for 2024. Browse UI:
  `https://benchmark-database.de/?track=main_2024&context=cnf`.

- **2026**: official competition file (SC2026 concluded; solver sources and
  instance-wise results are published on the same page):

  ```
  https://satcompetition.github.io/2026/downloads/track_main_2026.uri
  ```

  400 URLs. Cross-checked: the GBD query
  `getinstances?query=track%3Dmain_2026&context=cnf` returns the **identical**
  hash set.

- **Endpoint validation**: `getinstances?query=track%3Dmain_2025&context=cnf`
  returns exactly the hash set of the repo's existing `track_main_2025.uri`,
  confirming the GBD endpoint is the provenance of the repo's format.

## Files in this repo

- `data/benchmarks/track_main_2024.uri` — 400 URLs (from GBD, sorted by hash)
- `data/benchmarks/track_main_2026.uri` — 400 URLs (official SC2026 file)
- `scripts/download_satcomp.sh` — generalized downloader (see below)

(The 2025 list stays at the repo root, `track_main_2025.uri`; the downloader
checks both locations.)

Set overlap (shared hashes): 2024∩2025 = 7, 2026∩2025 = 8, 2024∩2026 = 19.

## Downloading

```
bash scripts/download_satcomp.sh <year> [--sample N]
```

Reads `track_main_<year>.uri` (repo root or `data/benchmarks/`), downloads into
`data/benchmarks/satcomp<year>/`, decompresses, and names files
`<md5>-<origname>.cnf` — identical conventions to
`scripts/download_satcomp2025.sh` / the existing `satcomp2025/` directory.
Improvements over the 2025 script:

- **Resumable**: instances whose decompressed `<md5>-*.cnf` already exists are
  skipped; interrupted downloads never leave partial files in the output dir
  (each file is fetched into a temp dir and moved only after successful
  decompression).
- Handles `.xz`/`.bz2`/`.gz`/`.lzma`/plain (a HEAD sweep of all 800 URLs
  confirmed every 2024 and 2026 instance is currently served as `.cnf.xz`).
- Verifies each decompressed file has a DIMACS `p cnf` header before accepting
  it; failures are listed and make the script exit non-zero.

Full downloads (not yet run):

```
bash scripts/download_satcomp.sh 2024     # 4.04 GiB compressed download
bash scripts/download_satcomp.sh 2026     # 2.09 GiB compressed download
```

Sizes (compressed totals are exact, from summing `Content-Length` over all 400
URLs per year; decompressed sizes are estimates using satcomp2025's measured
ratio — 5.04 GiB compressed -> 53 GB on disk, ~10.5x):

| year | instances | compressed | est. decompressed | largest single (compressed) |
|------|-----------|------------|-------------------|------------------------------|
| 2024 | 400       | 4.04 GiB   | ~35–45 GB         | 334 MiB                      |
| 2025 | 400       | 5.04 GiB   | 53 GB (measured)  | 836 MiB                      |
| 2026 | 400       | 2.09 GiB   | ~18–25 GB         | 189 MiB                      |

Validation performed (2026-08-14): `--sample 8` for both years — all 16 files
downloaded, decompressed, and start with a valid `p cnf` header (2024 samples:
56 KB–801 MB; 2026 samples: 156 KB–60 MB). A re-run correctly skipped all
existing files. Those 16 files remain in `data/benchmarks/satcomp2024/` (1.1 GB)
and `data/benchmarks/satcomp2026/` (93 MB); the full download resumes past them.

## Making the new sets usable in the evaluation pipeline

Downloading is not enough: the pipeline needs per-instance baseline data and a
quick subset for each year. Current blockers are hardcoded `satcomp2025` paths.

1. **Full download** (above).

2. **Baseline solving times** — `scripts/run_base_solver_eval.sh` submits a
   SLURM array (account `def-vganesh`, 4G, 01:30:00 wall, 5000 s solver
   timeout, one task per unsolved CNF) and, with `--collect`, parses the
   `.solving.log`s into `solvers/base/solving_times.json` (PAR-2, penalty
   10000). It hardcodes `BENCHMARK_PATH="data/benchmarks/satcomp2025"`
   (line 26), `RESULT_DIR="solvers/base/result"`, and the quick list — it must
   be parameterized by year (or copied per year) so results land in per-year
   result dirs. Then:

   ```
   bash scripts/run_base_solver_eval.sh --full            # submit ~400 tasks
   bash scripts/run_base_solver_eval.sh --collect --full  # after jobs finish
   ```

   Budget: worst case 400 x 5000 s ≈ 555 CPU-hours per year.
   `sbatch --test-only` with exactly these flags (`--account=def-vganesh
   --mem=4G --time=01:30:00 --array=0-399%1000`) was verified on this cluster
   on 2026-08-14: the scheduler accepted it (would start immediately on
   partition `cpubase_bycore_b1`). No job was actually submitted.

3. **Quick subset** — `scripts/generate_benchmark_subset.py` reads
   `solvers/base/solving_times.json` and writes a stratified 50-instance list
   to `data/benchmarks/satcomp2025_quick50.txt` (tiers: <10 s, 10–1000 s,
   1000–5000 s, timeout; seed 42). Input/output paths are hardcoded constants
   at the top of the script — point them at the per-year solving_times.json
   and e.g. `satcomp2024_quick50.txt`.

4. **Instance categories** — `scripts/generate_instance_categories.py` parses
   SAT/UNSAT from baseline `.solving.log`s (in `solvers/baseline/result`,
   `solvers/base/result`, `solvers/base/result_quick`) plus
   `data/results/baseline/baseline_solving_times.json`, and writes
   `data/benchmarks/instance_categories.json` (key = filename without `.cnf`,
   fields `satisfiability` and `baseline_time`). The result-dir list and the
   single global output path are hardcoded — needs per-year variants, or the
   new years' entries merged into the existing JSON (hash-prefixed keys cannot
   collide across years except for the 7/8/19 overlapping instances, which
   would get identical entries anyway).

5. **Other hardcoded `satcomp2025` references** that must be generalized before
   pointing the evolution loop at a new year:
   `scripts/run_base_solver_eval.sh`, `scripts/generate_benchmark_subset.py`,
   `scripts/ice_scripts/benchmark_evaluate.py`,
   `scripts/ice_scripts/run_baseline.py`, `src/llmsat/llmsat.py`,
   `src/llmsat/pipelines/evaluation.py`.
