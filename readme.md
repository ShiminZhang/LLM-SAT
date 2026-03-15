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

### 5. Benchmarks

Place `track_main_2025.uri` in the repo root (the SAT Competition 2025 URI list), then:

```bash
bash scripts/download_satcomp2025.sh
```

Downloads and extracts ~400 CNF files to `data/benchmarks/satcomp2025/`.

### 6. Function registry

The file `solvers/base/function_registry.yaml` tells the evaluation pipeline which C function to replace and where it lives in the source. It currently targets `kissat_restarting` and `restart_mab` in `src/restart.c`:

```yaml
functions:
  kissat_restarting:
    file: "src/restart.c"
    start_line: 15
    end_line: 38
    signature: "bool kissat_restarting(kissat *solver)"
```

To target a different function, add an entry with its source file path (relative to `solvers/base/`), start/end line numbers, and signature.

## SLURM Configuration

Update the SLURM settings in `src/llmsat/pipelines/evaluation.py` for your cluster:

- `SLURM_ACCOUNT` (line 48) — your allocation account (default: `def-vganesh`)

The Python venv activation path is now read from `path_config.yaml` automatically.

## Running the Pipeline

The full pipeline has three stages that repeat in a cycle. Each stage can be run standalone or chained together via the orchestration scripts.

### Data generation

Generates a population of solver heuristics organized into teams (leaders + member variants). Uses Gemini's batch API.

```bash
python src/llmsat/pipelines/gemini_data_generation.py
```

Prompt files live in `data/prompts/` (`leader_prompt.txt`, `variant_prompt.txt`, `coder_prompt.txt`).

### Evaluation

Builds each algorithm into a kissat binary, runs them against benchmarks on SLURM, and promotes the best member in each team to leader.

```bash
# Build + submit SLURM jobs
python src/llmsat/pipelines/evaluation.py \
    --run_all --generation_tag <TAG> --quick-eval --batch-mode

# Collect PAR2 scores (after SLURM jobs finish)
python src/llmsat/pipelines/evaluation.py \
    --collect_all_results --generation_tag <TAG> --quick-eval

# Promote best member to leader per team
python src/llmsat/pipelines/evaluation.py \
    --promote-leaders --generation_tag <TAG>
```

Use `--dry-run` to preview. Drop `--quick-eval` for full evaluation (400 CNFs, 5000s timeout) instead of quick (50 CNFs, 600s).

### Genetic evolution

Takes promoted leaders, analyzes strengths/weaknesses, pairs them for crossover, generates offspring, and evaluates.

```bash
python src/llmsat/pipelines/genetic_evolution.py \
    --generation_tag <TAG> \
    --output_tag <TAG>_gen1 \
    --code_prompt_path data/prompts/coder_prompt.txt \
    --evaluate --top_k 5 --rubric_min 6.0
```

## Orchestration Scripts

These scripts chain the stages above into automated loops. They handle SLURM polling, result collection, and leader promotion between iterations.

### Loop A — Leader Refinement

Iterates: generate mutants -> evaluate -> promote. The first argument selects the cluster (`cc` for Compute Canada, `nersc` for NERSC), which determines which Python scripts to run.

```bash
./run_loop_a.sh <cc|nersc> <base_tag> <n_iterations> [source_tag]

# Examples
./run_loop_a.sh cc gemini_trial5 3
./run_loop_a.sh nersc gemini_trial5 3
```

On `cc`, the loop runs `evaluation.py` and `gemini_data_generation.py`. On `nersc`, it runs the `_nersc` variants (`evaluation_nersc.py`, `gemini_data_generation_nersc.py`).

Environment variables: `POLL_INTERVAL` (default 120s), `M_VARIANTS` (default 3), `MODEL` (default `gemini-3-flash-preview`).

### Bridge — GE Offspring to New Leaders

Takes the top N offspring from a genetic evolution run and converts them into a clean leader pool under a new tag. Feed this into Loop A for further refinement.

```bash
./run_bridge.sh <ge_output_tag> <target_base_tag> [top_n]

# Example: promote top 10 offspring, then refine
./run_bridge.sh gemini_trial5_gen1_v1_iter1 gemini_trial5_ge1 10
./run_loop_a.sh cc gemini_trial5_ge1 3 gemini_trial5_ge1_iter0
```

### Genetic Evolution (shell wrapper)

`run_genetic_evolution.sh` is a convenience wrapper around `genetic_evolution.py` with preset parameters. Edit the environment variables and generation tag at the top before running.

## Typical Workflow

```
1. Data generation       (create initial population)
2. Loop A (N iters)      (mutate + evaluate + promote)
3. Genetic evolution     (crossover best leaders)
4. Bridge                (promote offspring to leaders)
5. Loop A (N iters)      (refine the new leaders)
6. Repeat from 3
```

## Multi-Cluster Support

The same codebase runs on both Compute Canada and NERSC. Differences are handled by:

1. **`path_config.yaml`** — each user sets their own solver and venv paths
2. **`run_loop_a.sh cc|nersc`** — selects the appropriate pipeline scripts per cluster
3. **NERSC script variants** (`evaluation_nersc.py`, `gemini_data_generation_nersc.py`) — contain NERSC-specific SLURM settings and job submission logic
