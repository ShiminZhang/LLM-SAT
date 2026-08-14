# KissatEvolve Overhaul — Decision & Progress Log

Autonomous overhaul started **2026-08-14** on branch **`claude/overhaul`**. Nothing is pushed to origin; `main` is untouched.

## Git layout

- `snapshot/pre-overhaul-2026-08-14` (`60141b30`) — the working tree exactly as found, including the previously **un-versioned** base-solver re-baselining (MAB/CHB stripped + CURE `a_*.c` layer), the prompt state, and the local loop fixes. This is the reproducibility anchor for the paper's pipeline: it did not exist in git before.
- `claude/overhaul` — all improvement work, built on top of the snapshot in reviewable commits.
- `main` — untouched.

## 🚩 FLAGS — need your input (I continued working; these don't block)

1. **Rotate the Postgres credential.** The RDS password is hardcoded in three committed scripts (`scripts/ice_scripts/delete_generation_data.py`, `scripts/old_scripts/show_par2_scores.py`, `scripts/old_scripts/check_generation_status.py`) and therefore in git history on GitHub. I am removing it from *current* code (env-based), but only rotation (+ optional history rewrite) actually fixes it, and that breaks teammates who use the old password — team decision.
2. **~600 GB reclaimable** under `solvers/`: past-run candidate `build/` object trees (`kissat_evolve_iter1` 297 GB, `AE_kissat2025_MAB_clean` 265 GB). I will make the pipeline prune *future* builds automatically and provide a cleanup script for old runs, but I won't delete existing run data without your OK — the raw logs/binaries exist nowhere else.
3. **Anthropic API key**: `.env` has OpenAI + Google creds only. I'm adding a provider-agnostic LLM layer with Claude support (Claude Sonnet 5 / Opus 4.8 are strong coder models); add `ANTHROPIC_API_KEY` to `.env` if you want them live.
4. Nothing is pushed anywhere; when you're back, review `git log claude/overhaul` and this file.

## Workstreams

| # | Workstream | Status |
|---|---|---|
| 0 | Safety: snapshot branch, .gitignore fixes, this log | ✅ done |
| 1 | Correctness: committed bugs + eval semantics (SAT validation, PAR-2 floor, partial logs, CPU-time unification, key derivation, injection safety) | 🔄 in progress |
| 2 | Structure: dedupe eval/orchestration logic, delete dead code, pyproject | ⏳ |
| 3 | Efficiency: hardlink+incremental candidate builds, build-tree pruning, higher build parallelism | ⏳ |
| 4 | Models: probe available Gemini/OpenAI models, provider-agnostic client, Claude support, unify model config | 🔄 agent running |
| 5 | Benchmarks: SATCOMP 2024/2026 lists + download tooling + stratified subsets | 🔄 agent running |
| 6 | Loop upgrades: implement subcategory-controlled retrieval (paper claims it; code lacks it), SAT model checker in the proof gate | 🔄 agent running (checker) |
| 7 | Tests: pytest suite for parsers/injector/PAR-2/checker + smoke build | ⏳ |
| 8 | Docs: README/docs truth pass, team-meeting summary | ⏳ |

## Decision log

- **2026-08-14** Snapshot-first strategy: committed the as-found tree to `snapshot/pre-overhaul-2026-08-14` so no local state can be lost and the paper pipeline is reproducible from git. `.gitignore`: stopped ignoring `*.md` (docs were untrackable); excluded `google-cloud-sdk/` (861 MB vendored SDK), `KissatEvolve.pdf` (confidential reviewer copy), base build artifacts, `t.txt`.
- **2026-08-14** Working branch `claude/overhaul` created; all changes land there in small commits, never on `main`.
