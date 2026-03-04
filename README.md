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

### 2. Environment variables

Create a `.env` file in the project root:

```
DB_PASS="<postgres password>"
OPENAI_API_KEY="<openai key>"
GOOGLE_API_KEY="<google/gemini key>"
```

`DB_PASS` is for the shared PostgreSQL database that stores algorithms, code, and scores. `OPENAI_API_KEY` is used by the genetic evolution pipeline. `GOOGLE_API_KEY` is used by data generation (Gemini batch API).

### 3. Base solver

The base solver is AE_kissatMAB. Place the `AE_kissat2025_MAB.tar.xz` tarball in the repo root and run:

```bash
bash setup_aemab.sh
```

This extracts the solver to `solvers/base/`, copies the function registry, and runs `./configure`.

### 4. Benchmarks

Place `track_main_2025.uri` in the repo root (the SAT Competition 2025 URI list), then:

```bash
bash scripts/download_satcomp2025.sh
```

Downloads and extracts ~400 CNF files to `data/benchmarks/satcomp2025/`.

### 5. Function registry

The file `solvers/base/function_registry.yaml` tells the evaluation pipeline which C function to replace and where it lives in the source. It currently targets `kissat_restarting` and `restart_mab` in `src/restart.c`:

```yaml
functions:
  kissat_restarting:
    file: "src/restart.c"
    start_line: 15
    end_line: 38
    signature: "bool kissat_restarting(kissat *solver)"
```

To target a different function, add an entry with its source file path (relative to `solvers/base/`), start/end line numbers, and signature. The data generation prompts must also ask for that function. The evaluation pipeline reads this registry to know where to inject generated code before compiling.

## SLURM Configuration

Before running anything, update the SLURM settings in `src/llmsat/pipelines/evaluation.py` for your cluster:

- `SLURM_ACCOUNT` (line 46) — your allocation account (default: `def-vganesh`)
- `_get_activation_cmd()` (line 110) — path to your Python virtualenv (default: `~/general/bin/activate`)

## Running the Pipeline

### Step 1: Data Generation

Generates a population of solver heuristics organized into teams. Each team has one leader (an original strategy) and several members (variants of that leader). The LLM first generates leader algorithms in natural language, then produces member variants by modifying specific steps of each leader, and finally translates all algorithms into C code. All three stages use Gemini's batch API.

Edit the `main()` function in `gemini_data_generation.py` to set your generation tag, prompt paths, number of leaders, and variants per leader, then run:

```bash
python src/llmsat/pipelines/gemini_data_generation.py
```

Prompt files live in `data/prompts/`:
- `leader_prompt.txt` — designer prompt for generating leader algorithms
- `variant_prompt.txt` — template for generating member variants (uses `{leader_algorithm}` and `{target_step_num}` placeholders)
- `coder_prompt.txt` — template for translating algorithms to C code (uses `ALGORITHM_PLACEHOLDER`)

### Step 2: Evaluation

Builds each generated algorithm into a kissat binary by injecting its C code into the base solver, then runs all binaries against the benchmark CNFs on SLURM. Each solver is timed per-instance; timeouts and crashes get a PAR2 penalty. After all jobs finish, results are collected and the best-performing member in each team is promoted to leader (swapping filesystem directories and database records).

```bash
# Build solvers and submit SLURM evaluation jobs
python src/llmsat/pipelines/evaluation.py \
    --run_all --generation_tag <TAG> --quick-eval

# After SLURM jobs complete, collect PAR2 scores
python src/llmsat/pipelines/evaluation.py \
    --collect_all_results --generation_tag <TAG> --quick-eval

# Promote best member to leader in each team
python src/llmsat/pipelines/evaluation.py \
    --promote-leaders --generation_tag <TAG>
```

Use `--dry-run` on any command to preview without submitting. Drop `--quick-eval` for full evaluation (400 CNFs, 5000s timeout) instead of the fast subset (50 CNFs, 600s timeout).

### Step 3: Genetic Evolution

Takes the promoted leaders from Step 2 and evolves them. An LLM first analyzes each leader to identify its strengths, weaknesses, and key mechanisms (causal analysis). It then proposes the most promising pairs of leaders to combine, generates offspring algorithms via crossover, translates them to C code, and evaluates them. Offspring that beat their parents' PAR2 are kept. Since SLURM evaluation is async, each invocation runs one iteration; run repeatedly to continue evolving.

```bash
python src/llmsat/pipelines/genetic_evolution.py \
    --generation_tag <TAG> \
    --code_prompt_path data/prompts/coder_prompt.txt \
    --evaluate \
    --top_k 5 \
    --rubric_min 6.0 \
    --model gpt-4.1
```

Use `--folder outputs/<TAG>` to load the population from local files instead of the database. Use `--causal_only` to run just the analysis stage without crossover. Use `--skip_causal` on subsequent runs to reuse cached reports.
