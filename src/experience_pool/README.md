# Experience Pool Integration Guide

This package provides a unified memory bank for solver evolution with 3 pools:
- **`algorithm`**: stores BAD experiences only.
- **`mutation`**: stores GOOD and BAD experiences.
- **`combination`**: stores GOOD and BAD experiences.

Use `ExperiencePoolManager` as the single entry point.

---

## Disabling the pools (env-var opt-out)

The mutation pool and combination pool can each be turned off at runtime via
environment variables, so you can A/B-test runs with and without retrieval
augmentation. Defaults are **enabled**; set the var to `0` to disable.

| Variable        | Effect when set to `0`                                                                                                                                                                                                                                |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MUTATION_POOL` | Skips mutation-pool retrieval in `parallel_orchestrator` (the `{experience_pool_section}` prompt placeholder is filled with `None`) and skips the mutation-pool update step in `scripts/update_experience_pool.py`.                                   |
| `COMB_POOL`     | Skips combination-pool retrieval in `genetic_evolution` (the experience block in the combination-proposal prompt is filled with `None`) and skips the combination-pool update step in `scripts/update_combination_experience_pool.py`.                |

The variable is parsed permissively: `0` means off; **anything else** (unset,
empty, `1`, `true`, …) means on. The variables are read inside the Python
processes, so they propagate naturally from the shell to all child invocations.

**Examples:**

```bash
# Run Loop A without the mutation experience pool (cc or nersc)
MUTATION_POOL=0 bash run_loop_a.sh cc gemini_trial5 3

# Run the bridge step without the combination experience pool
COMB_POOL=0 bash run_bridge.sh nersc gemini_trial5_gen1_v1 gemini_trial5_ge1_gen1

# Disable both pools for a fully zero-shot run
MUTATION_POOL=0 COMB_POOL=0 bash run_loop_a.sh nersc gemini_trial5 3 --init
```

When a pool is disabled, retrieval is skipped entirely (no FAISS/embedding
work) and the corresponding update script exits early with a log line
indicating it was skipped. The literal `None` substituted into the prompt
makes the ablation visible in any saved prompt dumps.

---

## Mutation pool: controllable retrieval by PAR2 subcategory

Mutation records carry fine-grained PAR2 scores (`sat`, `unsat`, `hard`,
`easy`, `overall`). By default, retrieval ranks by semantic similarity and
returns hits regardless of which subcategory each mutation actually
improved — a "good" mutation may have improved overall PAR2 while
*regressing* on, say, SAT instances.

To focus retrieval on mutations that improved one specific subcategory,
set **exactly one** of the following env vars to `1`:

| Variable | Effect when set to `1`                                                                                          |
| -------- | --------------------------------------------------------------------------------------------------------------- |
| `SAT`    | Drop any retrieved mutation whose `member_par2.sat   >= leader_par2.sat`   (kept the SAT regression).           |
| `UNSAT`  | Drop any retrieved mutation whose `member_par2.unsat >= leader_par2.unsat`.                                     |
| `HARD`   | Drop any retrieved mutation whose `member_par2.hard  >= leader_par2.hard`.                                      |
| `EASY`   | Drop any retrieved mutation whose `member_par2.easy  >= leader_par2.easy`.                                      |

Filtering happens **after** the existing similarity search, so the
ranking logic is unchanged — only the post-filter trims hits that did not
improve the chosen subcategory. Records missing PAR2 fields are also
filtered out (safe default). Counts of dropped/kept good and bad hits
are logged. Applies only to the **mutation pool** and works for both the
`cc` and `nersc` pipelines (both go through `parallel_orchestrator`).

Defaults are `0` (no filter). Setting more than one of the four to `1`
raises a `ValueError` at orchestrator init.

**Examples:**

```bash
# Focus retrieval on mutations that improved SAT instances
SAT=1 bash run_loop_a.sh cc gemini_trial5 3

# Same, on NERSC
SAT=1 bash run_loop_a.sh nersc gemini_trial5 3 --init

# Combine with pool disable: ablation, no retrieval at all
MUTATION_POOL=0 bash run_loop_a.sh cc gemini_trial5 3

# Invalid: raises ValueError before any work begins
SAT=1 UNSAT=1 bash run_loop_a.sh cc gemini_trial5 3
```

---

## 1. Initialization

To start using any of the pools, you first need to initialize the `ExperiencePoolManager`.

```python
from experience_pool import ExperiencePoolManager

# Default data root: src/experience_pool/data
manager = ExperiencePoolManager()

# Optional custom root:
# manager = ExperiencePoolManager(data_root="/tmp/exp_pool_data")
```

The general APIs you will be using from the manager are:
- `search_experience_pool(pool_name, query_text, retrieve_good_k, retrieve_bad_k, sample_good_k, sample_bad_k) -> ExperiencePoolSearchResult`
- `update(pool_name, *args, **kwargs) -> dict | None`

---

## 2. Algorithm Pool

The Algorithm Pool is used to store **BAD** algorithm experiences only. It helps the system remember algorithms that performed worse than the baseline, along with an explanation of why.

### `search_experience_pool`

For the algorithm pool, semantic retrieval is not used; instead, we randomly sample existing bad experiences to serve as prompt context.

**Example Usage:**
```python
res = manager.search_experience_pool(
    pool_name="algorithm",
    query_text="", # Not used since we only sample
    retrieve_good_k=0, # Must be 0 (outcome not supported)
    retrieve_bad_k=0, # Must be 0, semantic retrieval not used
    sample_good_k=0, # Must be 0 (outcome not supported)
    sample_bad_k=3, # Randomly sample 3 BAD experiences
)

# Access the sampled bad experiences
for hit in res.bad.unique:
    rec = hit.payload # AlgorithmExperienceRecord
    print(f"BAD Algorithm ID: {rec.algorithm_id}")
    print(f"Algorithm Description: {rec.algorithm_description}")
    print(f"Analysis: {rec.analysis}")

# Always check for section-level warnings/errors
if res.bad.error:
    print("BAD warning/error:", res.bad.error)
```

### `update`

Ingests a batch of algorithms and classifies them as BAD if their representative PAR2 score exceeds the baseline. 

**Example Usage:**
```python
summary = manager.update(
    "algorithm",
    input_dir="/path/to/run_dir",
    baseline_par2=123.45,
    baseline_code="...baseline source code...",
)
```

**Expected File Structure & Assumptions:**
- `input_dir` must contain a `leaders/` and/or `members/` directory.
- Files should be structured as `<input_dir>/leaders/algorithm_<id>/<id>.json` or `<input_dir>/members/algorithm_<id>/<id>.json`.
- Each candidate JSON must have: 
  - A valid `id`
  - A `description`
  - A valid singleton code-id list
  - A valid `raw_par2_score` list (exactly 5 finite numbers, where `raw_par2_score[4]` is the representative PAR2).

**Classification Logic:**
- **BAD**: Candidate PAR2 > `baseline_par2`.
- Otherwise: Neutral (skipped).

---

## 3. Mutation Pool

The Mutation Pool stores **GOOD** and **BAD** mutation experiences to remember successful and harmful, single-step modifications to an algorithm.

### `search_experience_pool`

Retrieval is based on matching the leader algorithm description and the intended mutation step.

**Example Usage:**
```python
# The recommended query_text format for mutation:
query = "Leader Algorithm Description: <desc>\nMutation Step: <step>"

res = manager.search_experience_pool(
    pool_name="mutation",
    query_text=query,
    retrieve_good_k=3, # Retrieve 3 semantic hits for GOOD
    retrieve_bad_k=3,  # Retrieve 3 semantic hits for BAD
    sample_good_k=2,   # Retrieve 2 random hits for GOOD
    sample_bad_k=2,    # Retrieve 2 random hits for BAD
)

# To access specific information from the result:
for hit in res.good.unique:
    rec = hit.payload # This is a MutationExperienceRecord object
    print(f"Similarity Score: {hit.score}")
    print(f"Leader ID: {rec.leader_algorithm_id}")
    print(f"Member ID: {rec.member_algorithm_id}")
    print(f"Leader Description:\n{rec.leader_algorithm_description}")
    print(f"Member Description:\n{rec.member_algorithm_description}")
    print(f"Mutation Step That Was Taken:\n{rec.step}")
    print(f"Analysis of Result:\n{rec.analysis}")
    print("-" * 40)

for hit in res.bad.unique:
    rec = hit.payload # This is a MutationExperienceRecord object
    # Just as with GOOD hits, you can access the exact same fields:
    print(f"BAD hit: score={hit.score}, record_id={hit.record_id}, member_id={rec.member_algorithm_id}")

if res.good.error:
    print("GOOD warning/error:", res.good.error)
if res.bad.error:
    print("BAD warning/error:", res.bad.error)
```

### `update`

Compares a mutated member algorithm against its parent (leader) to determine if the mutation was GOOD or BAD.

**Example Usage:**
```python
summary = manager.update(
    "mutation",
    input_dir="/path/to/run_iter0",
    top_k_good=5,   # Persist the top 5 BEST improvements
    top_k_bad=5,    # Persist the top 5 WORST degradations
    debug=False,
)
```

**Expected File Structure & Assumptions:**
- `input_dir` must contain both `leaders/` and `members/` directories.
- Leader JSON: `<input_dir>/leaders/algorithm_<id>/<id>.json`
- Member JSON: `<input_dir>/members/algorithm_<id>/<id>.json`
- Member JSON must have exactly one `parent_id` matching a leader in the `leaders/` directory.
- JSON files must include required text fields (`description`; plus `step` for members) and valid `raw_par2_score` arrays.

**Classification Logic:**
- **GOOD**: Member PAR2 < Leader PAR2.
- **BAD**: Member PAR2 > Leader PAR2.
- Equal PAR2 is Neutral.
- Candidate pairs are ranked by relative change magnitude; only the top-K good and top-K bad are persisted.

---

## 4. Combination Pool

The Combination Pool stores **GOOD** and **BAD** crossover experiences to record the results of merging two parent algorithms.

### `search_experience_pool`

Accepts a list of leader descriptions. Internally, the pool builds all possible unordered pairs of those leaders and searches for relevant past combinations.

**Example Usage:**
```python
# Pass a list of potential leaders you are considering combining.
leaders_descriptions = [
    "Alg A description...",
    "Alg B description...",
    "Alg C description...",
]

res = manager.search_experience_pool(
    pool_name="combination",
    query_text=leaders_descriptions,
    retrieve_good_k=3,
    retrieve_bad_k=3,
    sample_good_k=2,
    sample_bad_k=2,
)

# To access specific information from the result:
for hit in res.good.unique:
    rec = hit.payload # This is a CombinationExperienceRecord object
    print(f"Similarity Score: {hit.score}")
    print(f"Parent 1 ID: {rec.parent_alg1_id}")
    print(f"Parent 2 ID: {rec.parent_alg2_id}")
    print(f"New Algorithm ID (Offspring): {rec.new_algorithm_id}")
    print(f"Parent 1 Description:\n{rec.parent_alg1_description}")
    print(f"Parent 2 Description:\n{rec.parent_alg2_description}")
    print(f"New Algorithm Description:\n{rec.new_algorithm_description}")
    print(f"Analysis of Result:\n{rec.analysis}")
    print("-" * 40)

for hit in res.bad.unique:
    rec = hit.payload # This is a CombinationExperienceRecord object
    # Just as with GOOD hits, you can access the exact same fields:
    print(f"BAD combination found with score: {hit.score}, offspring id: {rec.new_algorithm_id}")
```

### `update`

Compares an offspring algorithmic approach against **both** of its parents. 

**Example Usage:**
```python
summary = manager.update(
    "combination",
    combined_dir="/path/to/genetic_output",
    parent_source_dir="/path/to/mutation_seed_output",
    threshold=0.10, # For backwards compatibility
    top_k_good=5,
    top_k_bad=5,
    debug=False,
)
```

**Expected File Structure & Assumptions:**
- Offspring Output (`combined_dir`): `<combined_dir>/members/algorithm_<id>/<id>.json`.
- Parent Source (`parent_source_dir`): both `<parent_source_dir>/leaders/...` and `<parent_source_dir>/members/...` exist.
- Offspring JSON must have exactly 2 valid `"parent_id"` values.
- Parent IDs must successfully resolve in the `parent_source_dir`.
- All involved files need `description` and valid `raw_par2_score` list fields.

**Classification Logic:**
- **GOOD**: Offspring PAR2 < **both** parents' PAR2 scores.
- **BAD**: Offspring PAR2 > **both** parents' PAR2 scores.
- Otherwise: Neutral (skipped).
- Ranked by the maximum relative change against parents; top-K persisted.

---

## 5. Error Handling & Downstream Fallbacks

### Search Errors
The `search_experience_pool` method returns an `ExperiencePoolSearchResult` object. Each outcome partition (`res.good` and `res.bad`) has an `error` attribute, which is a string describing any issues that occurred during retrieval (such as the outcome partition being unsupported, or the FAISS search throwing an exception).

**How to check:**
```python
res = manager.search_experience_pool(...)

if res.good.error:
    print(f"Failed to retrieve GOOD experiences: {res.good.error}")
if res.bad.error:
    print(f"Failed to retrieve BAD experiences: {res.bad.error}")
```

**Adapting Downstream Usage:**
If an error string is present, the corresponding `.retrieved`, `.sampled`, and `.unique` lists will safely be empty. Your downstream pipeline (like an LLM prompt builder) should gracefully fall back:
- If retrieving `good` experiences fails but `bad` succeeds, you can still inject the `bad` experiences.
- If both fail, your generation step should fallback to zero-shot (running generation without any experience pool contextual injections).
- Do not hardcode expectations of retrieving exactly `top_k` results. Always iterate over the `.unique` results list and inject dynamically.

### Update Errors
The `.update` methods return a `summary` dictionary that gives granular counts on parsing, evaluation logic, generated analyses, and insertions. 

**How to check:**
```python
summary = manager.update(...)

if summary and summary.get("errors"):
    print("Encountered the following errors during update:")
    for err in summary["errors"]:
        print(f"- {err}")
```

**Adapting Downstream Usage:**
If errors are logged to the `summary["errors"]` array, that signals non-fatal parsing or LLM-inference exceptions over some batch members (e.g. malformed JSON inputs or LLM generation issues). The system will still persist the valid ones. Nothing needs to be done if there are errors in update function.

---

## 6. Storage Layout 

Data is formally saved out under the manager configured data root (by default `src/experience_pool/data`). Inside this directory, partitions are divided by pool and outcome:

Each partition maintains three files:
- `index.faiss`: Semantic vector search index via inner-products.
- `id_map.json`: Mapping of FAISS vector indices to underlying record IDs.
- `records.json`: The raw payload map of the experiences (`AlgorithmExperienceRecord`, `MutationExperienceRecord`, `CombinationExperienceRecord` mapped by their deterministic record hash).

Layout hierarchy:
- `algorithm/bad/*`
- `mutation/good/*`
- `mutation/bad/*`
- `combination/good/*`
- `combination/bad/*`

