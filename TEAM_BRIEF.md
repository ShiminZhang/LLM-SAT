# KissatEvolve Overhaul — Team Brief (2026-08-14)

One-page summary for the team meeting. Full decision log with commit hashes: [IMPROVEMENTS.md](IMPROVEMENTS.md). Everything is on branch **`claude/overhaul`** (local only, nothing pushed; `main` untouched; the pre-overhaul tree is preserved verbatim on `snapshot/pre-overhaul-2026-08-14`).

## Headline numbers

| Metric | Before | After |
|---|---|---|
| Candidate build time | ~5–10 s (full copy + configure + make) | **1.05 s** (hardlink clone + incremental make; base-tree integrity proven by full sha256) |
| SAT answer validation | none (marker-absence only) | **real model checking** (streaming C verifier; validated on a 6M-clause instance from a live run) |
| Controllability (paper §3.2) | **not implemented** | implemented end-to-end (`TARGET_SUBCATEGORY=easy\|hard\|sat\|unsat`) |
| Max measurable improvement | PAR-2 floor at 400 auto-penalized anything better (baseline 429.82!) | uncapped; suspicious scores gated by answer validation instead |
| Partial-eval scoring | optimistic (averaged whatever logs existed) | missing instances count as penalties |
| Cross-cluster PAR-2 | CC measured CPU time, NERSC wall-clock | unified on CPU time |
| Benchmarks | SC2025 only | + **SC2024 and SC2026 fully downloaded** (400 each) |
| Models | 5 conflicting hardcoded defaults, Gemini+OpenAI | role-based config (`generation/coder/analysis`), + Claude routing, verified live against Aug-2026 model lineup |
| Dead weight | — | ~2,400 LOC deleted, 6,300 LOC attic'd, 5 loop scripts deduped onto one lib (−535 lines) |
| Tests | none runnable | **34 passing** (injector safety, instance keys, controlled retrieval, checkmodel) |

## Bugs that mattered (all fixed, each verified)

1. `run_loop_eval_success.sh` — the full-benchmark re-eval of best solvers — could not run at all (two committed IndentationErrors).
2. `run_loop_a.sh`'s SLURM pollers were broken as committed (quote-truncated embedded Python, silenced by `2>/dev/null` → infinite poll loops).
3. A solver printing `SATISFIABLE` with a wrong model passed "validation".
4. PAR-2 < 400 was auto-replaced with the timeout penalty — best evolved candidates sit at ~469, so only 15% headroom remained before the pipeline would destroy legitimate winners.
5. NERSC and CC PAR-2 measured different physical quantities (wall vs CPU).
6. Stale registry line numbers spliced code into the wrong region silently (`kissat_analyze` was 76 lines off); injection now verifies+relocates or refuses.
7. Proof jobs from reuse/eval_success were misdirected on NERSC (hardcoded CC account); bridge polled *all* the user's SLURM jobs.
8. `aws.py` silently dropped every `code_results` row (dict-row length guard).
9. DB password was hardcoded in three committed scripts (scrubbed; **rotation still needed — see flags**).
10. The team's uncommitted base-solver re-baselining (MAB stripped + CURE layer) existed only as local working-tree state — now preserved in git.

## Validated live today

- 50-instance quick eval submitted, run, and collected on fir through the new wrapper/collect path (30 solved, 22 UNSAT proofs kept, OOMs correctly penalized).
- checkmodel verified a real 154 MB / 6M-clause SAT answer from that run.
- All three LLM provider routes exercised with real calls, including the gpt-5.5+ temperature-rejection retry.

## Decisions needed from the team (also in IMPROVEMENTS.md flags)

1. **Rotate the Postgres credential** (it is in git history on GitHub).
2. **~600 GB reclaimable** in old run archives (`scripts/prune_run_artifacts.sh`, dry-run ready) — needs a go-ahead.
3. **Baseline is stale and eval noise is large**: three same-day repeats of the unmodified base solver measured PAR-2 **564.69 / 791.28 / 789.68** vs the configured 429.82 (raw data in `solvers/base/result_quick_rep{1,2,3}/`). Normalized scores are currently meaningless on this cluster, and ~40% run-to-run spread means candidate-vs-baseline comparisons need same-batch scheduling or repeats — worth a protocol decision before the next campaign.
4. **Coder-model A/B**: `gemini-3.6-flash` is ~2.7× cheaper than the current coder; one quick-eval iteration per arm would settle whether quality holds.
5. Add `ANTHROPIC_API_KEY` if you want Claude in the mix (routing is ready).
6. SC2024/SC2026 need per-year baselines + categories before the loop can target them (steps in `docs/benchmarks_2024_2026.md`).

## How to review

```bash
git log --oneline snapshot/pre-overhaul-2026-08-14..claude/overhaul   # 12 commits, each self-contained
python -m pytest tests/                                               # 34 tests
git diff main..claude/overhaul --stat
```
