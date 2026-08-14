# attic/

Retired-but-historically-real experiment code. Nothing in this directory is
maintained, imported by the live pipeline, or expected to run as-is; it is kept
for reference only.

Contents:

- `ice_scripts/` — the old GaTech-ICE marker-based evaluation + DPO finetuning
  stack (restart-only code injection, no proof/validation step, PAR2 computed
  differently from the current pipeline). Superseded by the canonical
  generation/evaluation pipeline under `src/llmsat/pipelines/`.
- `old_scripts/` — early generation/evaluation launcher scripts from before the
  current pipeline layout.
- `watch_scripts/` — one-off SLURM watcher scripts with hardcoded job IDs from
  specific past runs.
- `prompts/` — prompt files no longer referenced by any live code
  (`leader_prompt.txt` uses placeholders nothing fills; `kissat_api_reference.txt`
  is unreferenced).
- `eval_threetimes.sh` — one-off triple-evaluation script hardcoding another
  user's conda environment.
