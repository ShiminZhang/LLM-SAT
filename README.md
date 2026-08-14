# LLM-SAT (KissatEvolve)

Evolutionary search over kissat SAT-solver heuristic functions using LLMs. The pipeline generates candidate heuristic implementations (Gemini/OpenAI/Claude), evaluates them on SAT Competition benchmarks via SLURM, validates answers (DRAT proofs for UNSAT, model checking for SAT), promotes the best, and steers future mutations through a FAISS-backed experience memory bank with optional subcategory targeting.

> **Aug 2026 overhaul:** the codebase was systematically audited and improved on branch
> `claude/overhaul` — see [IMPROVEMENTS.md](IMPROVEMENTS.md) for every decision, fixed bug,
> and new capability (controlled retrieval, SAT model checker, ~1s incremental candidate
> builds, SATCOMP 2024/2026, provider-agnostic model config, 34-test suite).

## Summary Run

```bash

PYTHONPATH=src N_LEADERS=10 M_VARIANTS=10 ./run_loop_a.sh cc experiment 3 --init

# Outputs will be experiment_iter0, experiment_iter1, _iter2, experiment_iter3, which gets fed into genetic evolution next

PYTHONPATH=src TOP_K=5 PAR2_KEEP_TOP_N=10 ./run_bridge.sh cc experiment_iter3 experiment_gen1

# Now our best algos (ideally) are in experiment_gen1

# Go back to iterative member loop:
PYTHONPATH=src M_VARIANTS=5 ./run_loop_a.sh cc experiment_gen1 3

# output: 1,2, experiment_gen1_iter3
# More genetic evolution:

PYTHONPATH=src ./run_bridge.sh cc experiment_gen1_iter3 experiment_gen2


```


## Setup

### 1. Clone and install

```bash
git clone https://github.com/ShiminZhang/LLM-SAT.git
cd LLM-SAT
pip install -r requirements.txt
pip install -e .
```

### 2. Path configuration

Copy the template and fill in your cluster-specific paths:

```bash
cp path_config.template.yaml path_config.yaml
```

Edit `path_config.yaml`:

```yaml
base_solver: "/home/you/scratch/LLM-SAT/solvers/base"
python_activate: "~/your-venv/bin/activate"
```

This file is gitignored. All scripts and Python modules read paths from here, so you only set them once.

### 3. Environment variables

Create a `.env` file in the project root:

```
DB_PASS="<postgres password>"
OPENAI_API_KEY="<openai key>"
GOOGLE_PROJECT_ID="<gcp project id>"
ANTHROPIC_API_KEY="<anthropic key>"   # optional, enables claude-* models
```

`DB_PASS` is for the shared PostgreSQL database that stores algorithms, code, and scores. Gemini runs on **Vertex AI**: it needs `GOOGLE_PROJECT_ID` plus gcloud application-default credentials (`gcloud auth application-default login`), not an API key. `OPENAI_API_KEY` enables `gpt-*` models; `ANTHROPIC_API_KEY` enables `claude-*` models. Model roles (generation/coder/analysis) are set in `path_config.yaml` — see the template.

### 4. Base solver

Six kissat-family solver tarballs live in the repo root (`AE_kissat2025_MAB`, `Kissat-CURE`, `Kissat_CoRephase_CoReward`, `Kissat_MAB_CoRephase`, `kissat-pred`, `vsa`). Install any of them as the base:

```bash
bash setup_solver.sh AE_kissat2025_MAB.tar.xz
```

This extracts the solver to your configured `base_solver` path, copies the function registry, runs `./configure`, and flattens the `src/makefile` symlink. (`setup_aemab.sh` is the older AE-MAB-only variant.) Build the base once (`cd $base_solver && ./configure && make`) — candidate builds then hardlink-clone the prebuilt tree and rebuild only the injected file (~1s each).

### 5. Baseline evaluation (for PAR2 normalization)

To compare results across different clusters, run the baseline solver and record its PAR2:

```bash
# Submit baseline evaluation jobs
python scripts/evaluate_baseline.py (--quick-eval)

# After SLURM jobs complete, collect results
python scripts/evaluate_baseline.py --collect (--quick-eval)
```

The script will print the baseline PAR2. Add it to your `path_config.yaml`:

```yaml
baseline_par2: 1234.56  # Replace with your actual value
```

This enables normalized PAR2 scores (1.0 = same as baseline, lower is better).

### 6. Benchmarks

SAT Competition main tracks 2024, 2025, and 2026 are supported (400 instances each; URI lists in `data/benchmarks/track_main_*.uri` and repo root):

```bash
bash scripts/download_satcomp.sh 2025            # or 2024 / 2026
bash scripts/download_satcomp.sh 2026 --sample 8 # small sample
```

Resumable; validates DIMACS headers; extracts to `data/benchmarks/satcomp<year>/`. Note: 2024/2026 still need per-year baseline runs + `instance_categories.json` + a quick subset before the evolution loop can target them — see `docs/benchmarks_2024_2026.md` for the exact steps.

### 7. Function registry

The file `solvers/base/function_registry.yaml` tells the evaluation pipeline which C function to replace and where it lives in the source. The active target is set by `configure_target.py` (currently `kissat_bump_score_increment`); the registry lists every indexed candidate, e.g.:

```yaml
functions:
  kissat_restarting:
    file: "src/restart.c"
    start_line: 15
    end_line: 38
    signature: "bool kissat_restarting(kissat *solver)"
```

To target a different function, use `configure_target.py`:

```bash
# Switch to a different function (auto-detects source file)
python scripts/configure_target.py kissat_restarting

# Specify the source file explicitly
python scripts/configure_target.py restart_mab --file src/restart.c

# Use a different solver path
python scripts/configure_target.py my_func --solver /path/to/solver
```

This updates the function registry, rewrites the prompt templates (`leader_prompt_testing.txt` and `coder_prompt_testing.txt`) with the correct target name, signature, and embedded source file, and points `experience_pool_data_root` at the target's memory bank. Injection additionally verifies at build time that the function really is where the registry says (relocating or refusing if stale), so a missed re-index can no longer corrupt source files.

## SLURM Configuration

Update the SLURM settings in `src/llmsat/pipelines/evaluation.py` for your cluster:

## Running the Pipeline

The pipeline has two main phases that alternate:

1. **Loop A** (`run_loop_a.sh`) — Leader refinement: generates mutant variants of leaders, evaluates them, and promotes the best
2. **Bridge** (`run_bridge.sh`) — Genetic evolution: combines top leaders via LLM-guided crossover to produce new offspring

A complete evolution cycle is just 3 commands:

```bash
./run_loop_a.sh cc my_tag 3 --init       # Initialize + 3 refinement iterations
./run_bridge.sh cc my_tag_iter3 my_gen1  # Genetic crossover
./run_loop_a.sh cc my_gen1 3             # Continue refining the offspring
```

---

### Loop A: Leader Refinement (`run_loop_a.sh`)

Runs N iterations of: generate mutant variants → SLURM evaluation → collect results → promote best member to leader.

**Usage:**

```bash
./run_loop_a.sh <cc|nb|nersc> <base_tag> <n_iterations> [source_tag] [--init]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `cc\|nersc` | Cluster: `cc` uses `evaluation.py` (Compute Canada), `nersc` uses `evaluation_nersc.py` (Perlmutter, packs 128 solver runs per node) |
| `base_tag` | Base name for iteration tags (`{base_tag}_iter1`, `_iter2`, ...) |
| `n_iterations` | Number of mutate→evaluate→promote cycles |
| `source_tag` | (Optional) Tag to load initial leaders from. Defaults to `{base_tag}` |
| `--init` | (Optional) Generate initial leaders + members + code before starting iterations |

**Environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `M_VARIANTS` | 3 | Number of mutant variants per leader |
| `MODEL` | `default_model` from path_config.yaml (`gemini-3.1-pro-preview`) | LLM model for generation |
| `TARGET_SUBCATEGORY` | (unset) | Controlled retrieval: `easy`\|`hard`\|`sat`\|`unsat` steers mutation exemplars toward that subcategory |
| `QUICK_EVAL` | 1 | Quick 50-CNF eval (600s/1200 penalty) vs full 400 (5000s/10000) |
| `VERIFY_PROOFS` | 1 | DRAT-check UNSAT proofs + model-check SAT answers; gates promotion |
| `POLL_INTERVAL` | 120 | Seconds between SLURM job status checks |
| `N_LEADERS` | 5 | (Init mode only) Number of leaders to generate |
| `DESIGNER_PROMPT` | `data/prompts/leader_prompt_testing.txt` | (Init mode only) Path to leader prompt |
| `LLMSAT_BUILD_CONCURRENCY` | 8 | Concurrent candidate builds (each ~1s incremental) |

**Examples:**

```bash
# Initialize a new population and run 3 refinement iterations
./run_loop_a.sh cc gemini_trial5 3 --init

# Continue refining from existing leaders (no init)
./run_loop_a.sh cc gemini_trial5 2 gemini_trial5_iter3

# Use custom settings
N_LEADERS=10 M_VARIANTS=5 ./run_loop_a.sh cc gemini_trial5 3 --init
```

**What happens with `--init`:**

1. Generates `N_LEADERS` leader algorithms (natural language)
2. Generates `M_VARIANTS` member variants per leader
3. Translates all algorithms to C code
4. Builds solvers and submits SLURM evaluation jobs
5. Polls SLURM until all jobs complete
6. Collects PAR2 scores
7. Promotes best member to leader in each team
8. Proceeds to mutation iterations 1..N

**What happens in each iteration:**

1. Generates `M_VARIANTS` mutant variants for each leader
2. Builds and submits SLURM evaluation (skips already-evaluated leaders)
3. Polls until jobs complete
4. Collects PAR2 scores
5. Promotes best member to leader

---

### Bridge: Genetic Evolution (`run_bridge.sh`)

Promotes top offspring from a refined population, runs LLM-guided genetic crossover to combine leaders, evaluates the offspring, and collects results.

**Usage:**

```bash
./run_bridge.sh <cc|nersc> <input_tag> <output_tag>
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `cc\|nersc` | Cluster: `cc` for Compute Canada, `nersc` passes `--nersc` to `genetic_evolution.py` to use the NERSC evaluation backend |
| `input_tag` | Tag to read evaluated leaders from (scans `_iter1`, `_iter2`, ... automatically) |
| `output_tag` | Tag for the offspring population |

**Environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `TOP_K` | 10 | LLM combination proposals per minibatch |
| `MINIBATCH_SIZE` | 10 | Leaders per LLM proposal call |
| `RUBRIC_MIN` | 6.0 | Minimum proposal score to proceed |
| `RUBRIC_KEEP_TOP_N` | 50 | Keep top-N proposals after score filter (also the promote count) |
| `SHUFFLE_PASSES` | 2 | Number of shuffled minibatch passes |
| `MODEL` | `default_model` from path_config.yaml | LLM model |
| `PAR2_KEEP_TOP_N` | 50 | Keep top-N offspring by PAR2 score (run_ge_collect.sh defaults to 7) |
| `POLL_INTERVAL` | 120 | Seconds between SLURM job status checks |
| `QUICK_EVAL` | 1 | Set to `0` for full evaluation |

**Example:**

```bash
# Run genetic evolution on refined leaders from iter3
TOP_K=3 MINIBATCH_SIZE=5 ./run_bridge.sh cc my_tag_iter3 my_gen1
```

**What happens:**

1. Scans all `{input_tag}_iter*` directories for evaluated leaders
2. Selects top leaders by PAR2 score
3. LLM analyzes each leader (strengths, weaknesses, key mechanisms)
4. LLM proposes leader pairs to combine
5. LLM generates offspring algorithms via crossover
6. Translates offspring to C code
7. Builds and submits SLURM evaluation
8. Polls until all jobs complete
9. Collects PAR2 scores and keeps top offspring

---

## Full Pipeline Example

Here's a complete end-to-end run evolving SAT solver heuristics:

```bash
# 1. Initialize population with 5 leaders, 3 variants each, run 3 refinement iterations
N_LEADERS=5 M_VARIANTS=3 ./run_loop_a.sh cc experiment1 3 --init

# Output: experiment1_iter0 (initial), experiment1_iter1, _iter2, _iter3 (refined)
# Best leaders are in experiment1_iter3

# 2. Run genetic evolution to combine top leaders into new offspring
TOP_K=5 PAR2_KEEP_TOP_N=7 ./run_bridge.sh cc experiment1_iter3 experiment1_gen1

# Output: experiment1_gen1 (offspring from crossover)

# 3. Continue refining the new generation
M_VARIANTS=3 ./run_loop_a.sh cc experiment1_gen1 3

# Output: experiment1_gen1_iter1, _iter2, _iter3

# 4. Run another round of genetic evolution
./run_bridge.sh cc experiment1_gen1_iter3 experiment1_gen2

# ... and so on
```

**Results location:**

- Solver binaries and code: `solvers/<TAG>/{leaders,members}/algorithm_<ID>/code_<ID>/`
- Per-instance times: `solvers/<TAG>/<role>/algorithm_<ID>/solving_times_<code_id>.json`
- PAR2 breakdown / solver statistics: `par2_breakdown_<code_id>.json` / `solver_stats_<code_id>.json` alongside it
- Run artifacts (job ids, timing, par2 report, proof verdicts): `outputs/<TAG>/`
- Reclaim old runs' build trees: `scripts/prune_run_artifacts.sh <solvers-dir>` (dry-run by default)

**PAR2 scoring:**

The PAR2 score is the average solving time across all benchmark instances. Unsolved instances (timeout, crash, OOM) receive a penalty of 2× the timeout:
- Quick eval (`--quick-eval`): 50 CNFs, 600s timeout → 1200s penalty
- Full eval: 400 CNFs, 5000s timeout → 10000s penalty

---

## Prompts

Prompt files live in `data/prompts/`:

| File | Purpose |
|------|---------|
| `leader_prompt_testing.txt` | Designer prompt for generating leader algorithms |
| `variant_prompt.txt` | Template for generating member variants (uses `{leader_algorithm}` and `{target_step_num}` placeholders) |
| `coder_prompt_testing.txt` | Template for translating algorithms to C code (uses `ALGORITHM_PLACEHOLDER`) |

---

## Validation
To run evaluation and validation on existing best solvers from iterations:

```
PYTHONPATH=src ./run_loop_eval_success.sh <tag> <cluster> --no-clean
```

To run validation only:

```
python scripts/verify_iteration_proofs.py \
    --submit-slurm \
    --generation_tag "$generation_tag" \
    --benchmark_path data/benchmarks/satcomp2025 \
    --drat_trim "$DRAT_TRIM_CMD" \
    --check_timeout "$PROOF_CHECK_TIMEOUT" \
    --slurm-mem "$PROOF_VERIFY_MEM" \
    --slurm-time "$PROOF_VERIFY_TIME" \
    --slurm-max-concurrent "$PROOF_VERIFY_MAX_CONCURRENT"
```

`run_loop_a.sh` calls validation after each iteration automatically. Validation covers **both answer types**: UNSAT proofs are checked with drat-trim, and SAT models are verified against the CNF by `tools/checkmodel` (built automatically; without it SAT results are recorded `unverified`, which blocks promotion fail-safe).

After all validation jobs are done, there will be multiple proof_verification_xxx.json in outputs/tag/ directory. All invalid (solver,formula) runs for an iteration will be recorded in proof_verification_invalid.json. They should be empty for valid solvers.

Complete results including verification timeout and verification mem-kill warning will be in proof_verification.json.

---

## Tests

```bash
python -m pytest tests/    # 34 tests: injector safety, instance keys, controlled retrieval, checkmodel
```