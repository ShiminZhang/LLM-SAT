# LLM-SAT

Evolutionary search over SAT solver restart heuristics using LLMs. The pipeline generates candidate heuristic function implementations via Gemini, evaluates them on SAT Competition 2025 benchmarks via SLURM, promotes the best, and evolves new candidates through LLM-guided genetic crossover.

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
GOOGLE_API_KEY="<google/gemini key>"
```

`DB_PASS` is for the shared PostgreSQL database that stores algorithms, code, and scores. `OPENAI_API_KEY` is used by the genetic evolution pipeline. `GOOGLE_API_KEY` is used by data generation (Gemini batch API).

### 4. Base solver

The base solver is AE_kissatMAB. Place the `AE_kissat2025_MAB.tar.xz` tarball in the repo root and run:

```bash
bash setup_aemab.sh
```

This extracts the solver to your configured `base_solver` path, copies the function registry, and runs `./configure`.

### 5. Baseline evaluation (for PAR2 normalization)

To compare results across different clusters, run the baseline solver and record its PAR2:

```bash
# Submit baseline evaluation jobs
python scripts/evaluate_baseline.py

# After SLURM jobs complete, collect results
python scripts/evaluate_baseline.py --collect
```

The script will print the baseline PAR2. Add it to your `path_config.yaml`:

```yaml
baseline_par2: 1234.56  # Replace with your actual value
```

This enables normalized PAR2 scores (1.0 = same as baseline, lower is better).

### 6. Benchmarks

Place `track_main_2025.uri` in the repo root (the SAT Competition 2025 URI list), then:

```bash
bash scripts/download_satcomp2025.sh
```

Downloads and extracts ~400 CNF files to `data/benchmarks/satcomp2025/`.

### 7. Function registry

The file `solvers/base/function_registry.yaml` tells the evaluation pipeline which C function to replace and where it lives in the source. It currently targets `kissat_restarting` and `restart_mab` in `src/restart.c`:

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

This updates the function registry, rewrites the prompt templates (`leader_prompt_testing.txt` and `coder_prompt_testing.txt`) with the correct target name, signature, and embedded source file. Without this script you'd need to manually edit both prompts and the registry.

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
./run_loop_a.sh cc my_gen1 3 my_gen1     # Continue refining the offspring
```

---

### Loop A: Leader Refinement (`run_loop_a.sh`)

Runs N iterations of: generate mutant variants → SLURM evaluation → collect results → promote best member to leader.

**Usage:**

```bash
./run_loop_a.sh <cc|nersc> <base_tag> <n_iterations> [source_tag] [--init]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `cc\|nersc` | Cluster: `cc` uses `evaluation.py` (Compute Canada), `nersc` uses `evaluation_nersc.py` (Perlmutter, packs 128 solver runs per node) |
| `base_tag` | Base name for iteration tags (`{base_tag}_iter1`, `_iter2`, ...) |
| `n_iterations` | Number of mutate→evaluate→promote cycles |
| `source_tag` | (Optional) Tag to load initial leaders from. Defaults to `{base_tag}_iter0` |
| `--init` | (Optional) Generate initial leaders + members + code before starting iterations |

**Environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `M_VARIANTS` | 3 | Number of mutant variants per leader |
| `MODEL` | `gemini-3-flash-preview` | LLM model for generation |
| `POLL_INTERVAL` | 120 | Seconds between SLURM job status checks |
| `N_LEADERS` | 5 | (Init mode only) Number of leaders to generate |
| `DESIGNER_PROMPT` | `data/prompts/leader_prompt_testing.txt` | (Init mode only) Path to leader prompt |

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
| `TOP_K` | 5 | LLM combination proposals per minibatch |
| `MINIBATCH_SIZE` | 10 | Leaders per LLM proposal call |
| `RUBRIC_MIN` | 6.0 | Minimum proposal score to proceed |
| `RUBRIC_KEEP_TOP_N` | 10 | Keep top-N proposals after score filter |
| `SHUFFLE_PASSES` | 1 | Number of shuffled minibatch passes |
| `MODEL` | `gemini-3-flash-preview` | LLM model |
| `PAR2_KEEP_TOP_N` | 7 | Keep top-N offspring by PAR2 score |
| `POLL_INTERVAL` | 120 | Seconds between SLURM job status checks |

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
M_VARIANTS=3 ./run_loop_a.sh cc experiment1_gen1 3 experiment1_gen1

# Output: experiment1_gen1_iter1, _iter2, _iter3

# 4. Run another round of genetic evolution
./run_bridge.sh cc experiment1_gen1_iter3 experiment1_gen2

# ... and so on
```

**Results location:**

- Solver binaries and code: `solvers/<TAG>/{leaders,members}/algorithm_<ID>/code_<ID>/`
- Per-instance times: `results/solving_times_<code_id>.json`
- PAR2 breakdown: `results/par2_breakdown_<code_id>.json`
- Solver statistics: `results/solver_stats_<code_id>.json`

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
