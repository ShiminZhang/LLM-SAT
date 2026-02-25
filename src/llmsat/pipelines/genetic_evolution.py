"""
Genetic Evolution Pipeline for LLM-SAT.

Implements population-based evolution of SAT solver heuristics using leaders only:
  1. Causal Analysis: LLM analyzes each leader to identify strengths, weaknesses,
     key mechanisms, and improvement suggestions.
  2. LLM-Proposed Combinations: Instead of enumerating all O(n²) pairs, a single LLM
     call (per minibatch) receives all leaders + their causal analyses and proposes the
     top-k most promising pairs for crossover. Pairs are scored 1-10; only pairs above
     --rubric_min proceed. For large populations, the population is divided into
     minibatches of --minibatch_size leaders; proposals are merged and deduplicated.
  3. Crossover: LLM combines each proposed pair of parents into an offspring algorithm.
  4. Code Generation: LLM translates offspring algorithms into C code.
  5. Evaluation: Build solver and evaluate on SAT benchmarks via SLURM.
  6. Selection (PAR2 only): Keep offspring whose PAR2 beats the best parent.
     Top-tier offspring are promoted back into the population for the next iteration.

The pipeline from step 2 onward is a loop until PAR2 stops improving or
max_iterations is reached.

Usage:
    python src/llmsat/pipelines/genetic_evolution.py \
        --folder outputs/gemini_trial1 \
        --code_prompt_path data/prompts/coder_prompt.txt \
        --top_k 5 \
        --minibatch_size 10 \
        --model gpt-4.1 \
        --rubric_min 6.0 \
        --rubric_keep_top_n 10 \
        --evaluate \
        --max_iterations 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import logging
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from llmsat.llmsat import (
    CHATGPT_DATA_GENERATION_TABLE,
    AlgorithmResult,
    AlgorithmStatus,
    CodeResult,
    CodeStatus,
    NOT_INITIALIZED,
    get_id,
    get_logger,
    setup_logging,
)
from llmsat.utils.aws import (
    get_algorithm_result,
    get_code_result,
    get_ids_from_router_table,
    update_algorithm_result,
    update_code_result,
    update_router_table,
)
from llmsat.utils.chatgpt_helper import get_llm_response
from llmsat.utils.paths import get_generation_output_dir, get_solver_solving_times_path
from llmsat.pipelines.gemini_data_generation import (
    parse_algorithm_response,
    parse_code_response,
)
from llmsat.pipelines.chatgpt_data_generation import (
    read_code_prompt_template,
    generate_code_prompt,
)
import glob

setup_logging(level=logging.INFO)
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Individual:
    """Represents a single individual in the population."""
    algorithm_id: str
    algorithm_json: str  # raw JSON string of the algorithm spec
    code_id: str
    code: str
    par2: float
    target_function: str = "kissat_restarting"
    parent_id: Optional[str] = None  # parent algorithm id (for team members)
    generation: int = 0              # which evolution generation this came from


@dataclass
class CausalReport:
    """Causal analysis report for an individual."""
    algorithm_id: str
    strengths: List[str]
    weaknesses: List[str]
    key_mechanisms: List[str]
    improvement_suggestions: List[str]
    raw_response: str = ""


@dataclass
class OffspringResult:
    """Result from a crossover operation."""
    parent_a_id: str
    parent_b_id: str
    algorithm_json: str
    algorithm_id: str  # SHA256 of algorithm_json
    reason: str = ""
    parent_a_strengths_used: List[str] = field(default_factory=list)
    parent_b_strengths_used: List[str] = field(default_factory=list)
    target_function: str = "kissat_restarting"


# Max leaders per LLM call when proposing combinations (minibatch threshold)
DEFAULT_MINIBATCH_SIZE = 10

# Rubric dimensions for LLM-based offspring quality evaluation
RUBRIC_DIMENSIONS = {
    "improvement_over_parents": (
        "Improvement over Parents: Does the offspring meaningfully combine and improve "
        "upon both parents' strategies? Or is it essentially a copy of one parent with "
        "minor tweaks? Score 1-10 where 10 means a clearly superior hybrid that leverages "
        "the best of both parents."
    ),
    "algorithmic_novelty": (
        "Algorithmic Novelty: Does the offspring introduce new ideas, creative combinations, "
        "or unique mechanisms not present in either parent? Score 1-10 where 10 means highly "
        "original approach that goes beyond simple concatenation of parent strategies."
    ),
    "mechanistic_soundness": (
        "Mechanistic Soundness: Are the algorithmic decisions well-justified from a SAT-solving "
        "theory perspective? Are restart conditions, clause quality metrics, threshold adaptation, "
        "and state tracking logically coherent? Score 1-10 where 10 means every design choice is "
        "rigorously motivated and mutually consistent."
    ),
    "implementation_feasibility": (
        "Implementation Feasibility: Is the algorithm description concrete and specific enough to "
        "be translated into correct, efficient C code? Are thresholds, conditions, and data "
        "structures precisely defined? Score 1-10 where 10 means unambiguous and directly "
        "implementable with no guesswork."
    ),
    "complementarity": (
        "Complementarity of Inherited Traits: Did the offspring successfully merge complementary "
        "strengths from both parents rather than combining conflicting or redundant strategies? "
        "Score 1-10 where 10 means perfect synergy between inherited traits with no conflicts."
    ),
}


@dataclass
class RubricScore:
    """Multi-dimensional rubric evaluation of an offspring algorithm."""
    offspring_id: str
    parent_a_id: str
    parent_b_id: str
    # Dimension scores (1-10)
    improvement_over_parents: float = 0.0
    algorithmic_novelty: float = 0.0
    mechanistic_soundness: float = 0.0
    implementation_feasibility: float = 0.0
    complementarity: float = 0.0
    # Aggregate
    weighted_total: float = 0.0
    # LLM justifications per dimension
    justifications: Dict[str, str] = field(default_factory=dict)
    raw_response: str = ""

    @property
    def dimension_scores(self) -> Dict[str, float]:
        return {
            "improvement_over_parents": self.improvement_over_parents,
            "algorithmic_novelty": self.algorithmic_novelty,
            "mechanistic_soundness": self.mechanistic_soundness,
            "implementation_feasibility": self.implementation_feasibility,
            "complementarity": self.complementarity,
        }

    def compute_weighted_total(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Compute weighted total score. Default weights emphasize improvement and soundness."""
        if weights is None:
            weights = {
                "improvement_over_parents": 0.30,
                "algorithmic_novelty": 0.15,
                "mechanistic_soundness": 0.25,
                "implementation_feasibility": 0.15,
                "complementarity": 0.15,
            }
        total = 0.0
        for dim, w in weights.items():
            total += w * getattr(self, dim, 0.0)
        self.weighted_total = total
        return total


@dataclass
class CombinationProposal:
    """A proposed pairwise combination of two leaders, scored by the LLM."""
    parent_a_id: str
    parent_b_id: str
    score: float  # 1-10, higher = more promising combination
    justification: str
    complementary_strengths: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 1: Load Population
# ---------------------------------------------------------------------------

def load_population(generation_tag: str) -> List[Individual]:
    """
    Load the population from the database for a given generation tag.

    For each algorithm, picks the best (lowest PAR2) evaluated code.
    Skips algorithms with no evaluated codes.
    """
    logger.info(f"Loading population for generation tag: {generation_tag}")
    algorithm_ids = get_ids_from_router_table(CHATGPT_DATA_GENERATION_TABLE, generation_tag)

    population = []
    skipped_no_code = 0
    skipped_no_par2 = 0

    for algo_id in algorithm_ids:
        algo_result = get_algorithm_result(algo_id)
        if algo_result is None:
            logger.warning(f"Algorithm {algo_id} not found in DB, skipping")
            continue

        code_ids = algo_result.code_id_list or []
        if not code_ids or (len(code_ids) == 1 and code_ids[0] == NOT_INITIALIZED):
            skipped_no_code += 1
            continue

        # Find the best evaluated code (lowest PAR2)
        best_code = None
        best_par2 = float("inf")

        for code_id in code_ids:
            code_result = get_code_result(code_id)
            if code_result is None:
                continue
            if code_result.par2 is not None and code_result.par2 < best_par2:
                best_par2 = code_result.par2
                best_code = code_result

        if best_code is None or best_par2 == float("inf"):
            skipped_no_par2 += 1
            continue

        individual = Individual(
            algorithm_id=algo_id,
            algorithm_json=algo_result.algorithm,
            code_id=best_code.id,
            code=best_code.code,
            par2=best_par2,
            target_function=algo_result.target_function,
            parent_id=algo_result.parent_id,
        )
        population.append(individual)

    logger.info(
        f"Loaded {len(population)} individuals "
        f"(skipped {skipped_no_code} with no codes, {skipped_no_par2} with no PAR2)"
    )
    return population


def _load_par2_for_algorithm(
    algorithm_id: str,
    code_id: str,
    generation_tag: Optional[str] = None,
    parent_id: Optional[str] = None,
) -> Optional[float]:
    """
    Try to get PAR2 score from multiple sources (in priority order):
      1. Local solving_times JSON file  (written by EvaluationPipeline after SLURM)
      2. Local result/ log files        (individual CNF solving logs)
      3. Database (code_result.par2)

    Returns None if no PAR2 is available yet (e.g. SLURM hasn't run).
    """
    from llmsat.utils.paths import get_solver_result_dir

    # 1. Try local solving_times JSON (primary; written after SLURM completes)
    try:
        times_path = get_solver_solving_times_path(
            algorithm_id, code_id,
            generation_tag=generation_tag,
            parent_id=parent_id,
        )
        if os.path.exists(times_path):
            with open(times_path, "r") as f:
                times = json.load(f)
            if times:
                return sum(times.values()) / len(times)
    except Exception:
        pass

    # 2. Try individual result .log files under result/ directory
    try:
        result_dir = get_solver_result_dir(
            algorithm_id, code_id,
            generation_tag=generation_tag,
            parent_id=parent_id,
        )
        log_files = glob.glob(os.path.join(result_dir, "*.log"))
        if log_files:
            PAR2_PENALTY = 10000
            times = []
            for log_path in log_files:
                t = _parse_solving_time_from_log(log_path, PAR2_PENALTY)
                times.append(t)
            if times:
                return sum(times) / len(times)
    except Exception:
        pass

    # 3. Fallback: try DB
    try:
        code_result = get_code_result(code_id)
        if code_result is not None and code_result.par2 is not None:
            return code_result.par2
    except Exception:
        pass

    return None


def _parse_solving_time_from_log(log_path: str, penalty: float = 10000) -> float:
    """Parse a single SLURM result log and return solving time (or penalty on timeout/error)."""
    try:
        with open(log_path, "r") as f:
            content = f.read()
        # Look for lines like "c elapsed: 1.23" or "s SATISFIABLE" / "s UNSATISFIABLE"
        time_match = re.search(r"c\s+elapsed[:\s]+([0-9.]+)", content)
        if time_match:
            return float(time_match.group(1))
        # Fallback: look for wall-clock time between START/END markers
        start_match = re.search(r"START_TIME=([0-9.]+)", content)
        end_match = re.search(r"END_TIME=([0-9.]+)", content)
        if start_match and end_match:
            elapsed = float(end_match.group(1)) - float(start_match.group(1))
            return elapsed
    except Exception:
        pass
    return penalty


def load_population_from_folder(
    folder: str,
    generation_tag: Optional[str] = None,
) -> List[Individual]:
    """
    Load the population directly from a Gemini outputs folder, without requiring
    the database for algorithms/codes. PAR2 scores come from local solving_times
    JSON files or (as fallback) from the database.

    Expected folder layout (e.g. "outputs/gemini_trial1"):
      outputs/{generation_tag}/
        batch_batches/{batch_id}/
          leaders_output.txt             -- leader NL algorithms (JSONL, one per line)
          member_batch_input_{algo_hash}.txt
          member_output_batches/{batch_id}.txt   -- member NL algorithms (JSONL)
          code_batch_input_{algo_hash}.txt
          code_output_batches/{batch_id}.txt     -- C code (JSONL)
          team_batch_map_{timestamp}.json        -- maps batch_ids -> algo hashes

    Corresponding PAR2 files (written by EvaluationPipeline.collect_results):
      solvers/{generation_tag}/{role}/algorithm_{algo_id}/solving_times_{code_id}.json

    Steps:
      1. Locate the batch_batches/{batch_id} sub-directory.
      2. Parse NL algorithms from leaders_output.txt and member_output_batches/.
      3. Load the latest team_batch_map JSON to map batch IDs -> algo hashes.
      4. Parse C code from code_output_batches/.
      5. For each algorithm, find the best PAR2 from local solving_times files.
      6. Return Individual objects for algorithms that have code + PAR2.
    """
    logger.info(f"Loading population from folder: {folder}")

    # ------------------------------------------------------------------
    # Locate the batch_batches sub-directory
    # ------------------------------------------------------------------
    batch_dir = None

    # Case 1: folder is "outputs/{tag}" — find batch_batches/* inside
    batch_batches_dir = os.path.join(folder, "batch_batches")
    if os.path.isdir(batch_batches_dir):
        entries = sorted(os.listdir(batch_batches_dir))
        for entry in entries:
            full = os.path.join(batch_batches_dir, entry)
            if os.path.isdir(full):
                batch_dir = full
                break
    elif os.path.isdir(folder):
        # Case 2: folder might contain batch_ directories directly
        for entry in sorted(os.listdir(folder)):
            full = os.path.join(folder, entry)
            if entry.startswith("batch_") and os.path.isdir(full):
                batch_dir = full
                break

    if batch_dir is None:
        # Case 3: folder itself is the batch directory
        if os.path.exists(os.path.join(folder, "leaders_output.txt")):
            batch_dir = folder
        else:
            logger.error(f"No batch directory found in {folder}")
            return []

    logger.info(f"Using batch directory: {batch_dir}")

    # ------------------------------------------------------------------
    # 1. Parse algorithms from leaders_output.txt
    # ------------------------------------------------------------------
    # algorithm_id -> algorithm_json_string
    algorithms: Dict[str, str] = {}
    # algorithm_id -> parent_id (None for leaders)
    parent_map: Dict[str, Optional[str]] = {}
    # algorithm_id -> target_function
    target_fn_map: Dict[str, str] = {}

    leaders_path = os.path.join(batch_dir, "leaders_output.txt")
    if os.path.exists(leaders_path):
        with open(leaders_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    resp = json.loads(line)
                    algo_str, target_fn = parse_algorithm_response(resp)
                    algo_id = get_id(algo_str)
                    algorithms[algo_id] = algo_str
                    parent_map[algo_id] = None
                    target_fn_map[algo_id] = target_fn or "kissat_restarting"
                except Exception as e:
                    logger.warning(f"Failed to parse leader line: {e}")
    logger.info(f"Parsed {len(algorithms)} leaders from {leaders_path}")

    # ------------------------------------------------------------------
    # 2. Load team_batch_map to resolve batch IDs -> algo hashes
    # ------------------------------------------------------------------
    map_files = sorted(glob.glob(os.path.join(batch_dir, "team_batch_map_*.json")))
    member_batch_to_leader: Dict[str, str] = {}  # member_batch_id -> leader algo_id
    code_batch_to_algo: Dict[str, str] = {}       # code_batch_id -> algo_id

    if map_files:
        # Use the latest map file
        with open(map_files[-1], "r") as f:
            batch_map = json.load(f)

        # member_batch_map: "batches/{batch_id}" -> algo_hash
        raw_member_map = batch_map.get("member_batch_map", {})
        for batch_key, algo_hash in raw_member_map.items():
            # Strip "batches/" prefix if present
            batch_id = batch_key.split("/")[-1]
            member_batch_to_leader[batch_id] = algo_hash  # algo_hash IS the leader algo_id

        # code_batch_map: "batches/{batch_id}" -> algo_hash
        raw_code_map = batch_map.get("code_batch_map", {})
        for batch_key, algo_hash in raw_code_map.items():
            batch_id = batch_key.split("/")[-1]
            code_batch_to_algo[batch_id] = algo_hash

    logger.info(f"Total algorithms parsed: {len(algorithms)}")

    # ------------------------------------------------------------------
    # 4. Parse code from code_output_batches/ directory
    # ------------------------------------------------------------------
    # algorithm_id -> list of (code_id, code_str)
    algo_codes: Dict[str, List[Tuple[str, str]]] = {}

    code_output_dir = os.path.join(batch_dir, "code_output_batches")
    if os.path.isdir(code_output_dir):
        for fname in sorted(os.listdir(code_output_dir)):
            if not fname.endswith(".txt"):
                continue
            batch_id = fname[:-4]  # strip .txt
            algo_id = code_batch_to_algo.get(batch_id)

            if algo_id is None:
                logger.debug(f"No algorithm mapping for code batch {batch_id}, skipping")
                continue

            code_file = os.path.join(code_output_dir, fname)
            with open(code_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        resp = json.loads(line)
                        code_str = parse_code_response(resp)
                        if not code_str:
                            continue
                        code_id = get_id(code_str)
                        if algo_id not in algo_codes:
                            algo_codes[algo_id] = []
                        algo_codes[algo_id].append((code_id, code_str))
                    except Exception as e:
                        logger.warning(f"Failed to parse code line in {fname}: {e}")

    # Also check flat code_output_batch_{batch_id}.txt files (alternative layout)
    flat_code_files = sorted(glob.glob(os.path.join(batch_dir, "code_output_batch_*.txt")))
    for code_file in flat_code_files:
        fname = os.path.basename(code_file)
        batch_id = fname.replace("code_output_", "").replace(".txt", "")
        algo_id = code_batch_to_algo.get(batch_id)
        if algo_id is None:
            continue
        with open(code_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    resp = json.loads(line)
                    code_str = parse_code_response(resp)
                    if not code_str:
                        continue
                    code_id = get_id(code_str)
                    if algo_id not in algo_codes:
                        algo_codes[algo_id] = []
                    algo_codes[algo_id].append((code_id, code_str))
                except Exception as e:
                    logger.warning(f"Failed to parse code line in {fname}: {e}")

    logger.info(f"Parsed codes for {len(algo_codes)} algorithms")

    # ------------------------------------------------------------------
    # 5. Assemble Individuals: match algorithm + best code + PAR2
    # ------------------------------------------------------------------
    population: List[Individual] = []
    skipped_no_code = 0
    no_par2_count = 0

    # Infer generation_tag from folder path if not provided
    inferred_tag = generation_tag
    if inferred_tag is None:
        folder_norm = folder.rstrip("/\\")
        parts = folder_norm.replace("\\", "/").split("/")
        for part in reversed(parts):
            if not part.startswith("batch_") and part != "outputs" and part != "batch_batches" and part:
                inferred_tag = part
                break

    for algo_id, algo_json in algorithms.items():
        codes = algo_codes.get(algo_id, [])
        if not codes:
            skipped_no_code += 1
            continue

        p_id = parent_map.get(algo_id)

        # Try to find the best PAR2 among all codes for this algorithm.
        # If no PAR2 is available (SLURM hasn't run yet), fall back to the
        # first code and set par2=inf so causal analysis can still use it.
        best_code_id = codes[0][0]
        best_code_str = codes[0][1]
        best_par2 = float("inf")

        for code_id, code_str in codes:
            par2 = _load_par2_for_algorithm(
                algo_id, code_id,
                generation_tag=inferred_tag,
                parent_id=p_id,
            )
            if par2 is not None and par2 < best_par2:
                best_par2 = par2
                best_code_id = code_id
                best_code_str = code_str

        if best_par2 == float("inf"):
            no_par2_count += 1
            logger.debug(
                f"No PAR2 for {algo_id[:16]} (SLURM not run yet); "
                f"including with par2=inf for causal analysis"
            )

        individual = Individual(
            algorithm_id=algo_id,
            algorithm_json=algo_json,
            code_id=best_code_id,
            code=best_code_str,
            par2=best_par2,
            target_function=target_fn_map.get(algo_id, "kissat_restarting"),
            parent_id=p_id,
            generation=0,
        )
        population.append(individual)

    logger.info(
        f"Loaded {len(population)} individuals from folder "
        f"(skipped {skipped_no_code} with no code, "
        f"{no_par2_count} with no PAR2 yet — using inf as placeholder)"
    )
    return population


# ---------------------------------------------------------------------------
# Stage 2: Causal Analysis
# ---------------------------------------------------------------------------

CAUSAL_ANALYSIS_SYSTEM_MESSAGE = (
    "You are an expert SAT solver researcher specializing in analyzing the causal factors "
    "that determine solver performance. You provide rigorous, specific analysis."
)


def build_causal_prompt(individual: Individual, baseline_par2: Optional[float] = None) -> str:
    """Build the prompt for causal analysis of a single individual."""
    baseline_section = ""
    if baseline_par2 is not None:
        diff = individual.par2 - baseline_par2
        direction = "better" if diff < 0 else "worse"
        baseline_section = (
            f"\n### Baseline PAR2: {baseline_par2:.2f}"
            f"\nThis solution is {abs(diff):.2f} {direction} than the baseline."
        )

    return f"""Analyze the causal factors that determine the performance of this SAT solver heuristic.

### Algorithm (Natural Language Description):
{individual.algorithm_json}

### C Code Implementation:
{individual.code}

### PAR2 Score: {individual.par2:.2f} (lower is better,  ignore PAR2 in your analysis if it shows as Inf.){baseline_section}

Your task: Identify the specific design decisions and mechanisms that CAUSE this algorithm to perform the way it does. Focus on:
- What specific algorithmic choices improve SAT solving performance and WHY they work
- What choices may hurt performance and WHY
- The core mechanisms that drive the behavior (e.g., how restart frequency adapts, what triggers restarts, how clause quality is tracked)
- Concrete suggestions for improvement

Be specific and quantitative where possible. Reference actual thresholds, conditions, and data structures from the code.

Produce your analysis as JSON:
```json
{{
    "strengths": ["Each strength: what design decision + why it improves performance"],
    "weaknesses": ["Each weakness: what design decision + why it hurts performance"],
    "key_mechanisms": ["Each mechanism: description of a core behavioral driver"],
    "improvement_suggestions": ["Each suggestion: a specific, concrete change that could improve performance"]
}}
```"""


def parse_causal_report(response: str, algorithm_id: str) -> CausalReport:
    """Parse the LLM response into a CausalReport."""
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        json_match = re.search(r"\{[^{}]*\"strengths\"[^{}]*\}", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = response

    try:
        data = json.loads(json_str)
        return CausalReport(
            algorithm_id=algorithm_id,
            strengths=data.get("strengths", []),
            weaknesses=data.get("weaknesses", []),
            key_mechanisms=data.get("key_mechanisms", []),
            improvement_suggestions=data.get("improvement_suggestions", []),
            raw_response=response,
        )
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse causal report JSON for {algorithm_id}, storing raw")
        return CausalReport(
            algorithm_id=algorithm_id,
            strengths=[],
            weaknesses=[],
            key_mechanisms=[],
            improvement_suggestions=[],
            raw_response=response,
        )


def generate_causal_reports(
    population: List[Individual],
    model: str = "gpt-4.1",
    baseline_par2: Optional[float] = None,
) -> Dict[str, CausalReport]:
    """
    Stage 1: Generate causal analysis reports for each individual.
    Returns a dict mapping algorithm_id -> CausalReport.
    """
    logger.info(f"Generating causal reports for {len(population)} individuals")
    reports: Dict[str, CausalReport] = {}

    for i, individual in enumerate(population):
        logger.info(
            f"[{i+1}/{len(population)}] Analyzing {individual.algorithm_id[:16]}... "
            f"(PAR2={individual.par2:.2f})"
        )
        prompt = build_causal_prompt(individual, baseline_par2)
        response = get_llm_response(
            prompt=prompt,
            system_message=CAUSAL_ANALYSIS_SYSTEM_MESSAGE,
            model=model,
            temperature=0.3,
        )
        report = parse_causal_report(response, individual.algorithm_id)
        reports[individual.algorithm_id] = report
        logger.info(
            f"  Strengths: {len(report.strengths)}, "
            f"Weaknesses: {len(report.weaknesses)}, "
            f"Mechanisms: {len(report.key_mechanisms)}"
        )

    return reports


# ---------------------------------------------------------------------------
# Stage 3: LLM-Proposed Combinations (replaces exhaustive pair selection + rubric filter)
# ---------------------------------------------------------------------------

COMBINATION_PROPOSAL_SYSTEM_MESSAGE = (
    "You are an expert SAT solver researcher. You analyze multiple solver algorithms "
    "and their causal analyses to identify the most promising pairwise combinations "
    "for genetic crossover. Your goal is to propose combinations that maximize "
    "the potential for producing superior offspring algorithms with lower PAR2 scores."
)


def build_combination_proposal_prompt(
    batch: List[Individual],
    causal_reports: Dict[str, CausalReport],
    top_k: int = 5,
) -> str:
    """Build a prompt asking the LLM to propose top-k combinations from a batch of leaders."""
    leaders_text = ""
    for idx, ind in enumerate(batch):
        report = causal_reports.get(ind.algorithm_id)
        if report is None:
            continue
        par2_str = f"{ind.par2:.2f}" if ind.par2 != float("inf") else "N/A (not evaluated yet)"
        leaders_text += f"""
[Index {idx}] Algorithm ID: {ind.algorithm_id[:16]}...
PAR2: {par2_str}
Algorithm:
{ind.algorithm_json}

Causal Analysis:
- Strengths: {json.dumps(report.strengths, indent=2)}
- Weaknesses: {json.dumps(report.weaknesses, indent=2)}
- Key Mechanisms: {json.dumps(report.key_mechanisms, indent=2)}
- Improvement Suggestions: {json.dumps(report.improvement_suggestions, indent=2)}
"""

    return f"""You are selecting the most promising pairs of SAT solver algorithms to combine via genetic crossover.

Below are {len(batch)} solver algorithms with their causal performance analyses.
Goal: lower PAR2 (lower is better). Ignore PAR2 in your scoring if shown as N/A.

Your task: Propose the top {top_k} most promising pairwise combinations for crossover.
For each proposed pair, assess:
- How well the two algorithms' strengths complement each other
- Whether combining them would address each other's weaknesses
- The potential for creating a superior hybrid

### Algorithms:
{leaders_text}
---

Propose the top {top_k} combinations. Score each 1-10 (10 = highest potential).
Rank by score descending (best combination first).

Requirements:
- index_a and index_b must be different indices from the list above
- Enforce index_a < index_b to avoid duplicates
- Base scores on causal analysis (strengths, weaknesses, mechanisms)

Output as JSON only:
```json
{{
    "proposed_combinations": [
        {{
            "index_a": <integer index of first parent>,
            "index_b": <integer index of second parent>,
            "score": <1.0-10.0>,
            "justification": "Why this pair is promising for crossover",
            "complementary_strengths": ["strength from A that pairs with B", "..."]
        }}
    ]
}}
```"""


def parse_combination_proposals(
    response: str,
    batch: List[Individual],
) -> List[CombinationProposal]:
    """Parse LLM response into a list of CombinationProposal objects."""
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        json_str = response.strip()

    proposals: List[CombinationProposal] = []
    try:
        data = json.loads(json_str)
        for entry in data.get("proposed_combinations", []):
            idx_a = entry.get("index_a")
            idx_b = entry.get("index_b")
            if idx_a is None or idx_b is None:
                logger.debug("Proposal missing index_a or index_b, skipping")
                continue
            if not (0 <= idx_a < len(batch)) or not (0 <= idx_b < len(batch)):
                logger.warning(f"Proposal indices ({idx_a}, {idx_b}) out of range for batch size {len(batch)}, skipping")
                continue
            if idx_a == idx_b:
                logger.debug("Proposal has index_a == index_b, skipping")
                continue
            score = float(entry.get("score", 5.0))
            score = max(1.0, min(10.0, score))
            proposals.append(CombinationProposal(
                parent_a_id=batch[idx_a].algorithm_id,
                parent_b_id=batch[idx_b].algorithm_id,
                score=score,
                justification=entry.get("justification", ""),
                complementary_strengths=entry.get("complementary_strengths", []),
            ))
    except json.JSONDecodeError:
        logger.warning("Failed to parse combination proposals JSON")
    return proposals


def propose_combinations_llm(
    population: List[Individual],
    causal_reports: Dict[str, CausalReport],
    top_k: int = 5,
    minibatch_size: int = DEFAULT_MINIBATCH_SIZE,
    combination_score_min: float = 0.0,
    model: str = "gpt-4.1",
) -> List[Tuple["CombinationProposal", Individual, Individual]]:
    """
    Ask the LLM to propose top-k pairwise combinations from the population.

    For populations larger than minibatch_size, splits into minibatches and
    aggregates proposals across all batches before returning the top results.

    Args:
        population: All current individuals (leaders) to consider.
        causal_reports: Causal analysis for each individual.
        top_k: Number of combinations to request per minibatch.
        minibatch_size: Max leaders per LLM call (threshold for splitting).
        combination_score_min: Minimum score (1-10) to keep a proposal.
        model: LLM model name.

    Returns:
        List of (CombinationProposal, parent_a, parent_b), sorted by score desc.
    """
    pop_lookup: Dict[str, Individual] = {ind.algorithm_id: ind for ind in population}

    # Split into minibatches if needed
    if len(population) <= minibatch_size:
        batches = [population]
    else:
        batches = [
            population[i: i + minibatch_size]
            for i in range(0, len(population), minibatch_size)
        ]
        logger.info(
            f"Splitting {len(population)} leaders into {len(batches)} minibatch(es) "
            f"of up to {minibatch_size}"
        )

    all_proposals: List[CombinationProposal] = []

    for batch_idx, batch in enumerate(batches):
        logger.info(
            f"Minibatch {batch_idx + 1}/{len(batches)}: requesting top-{top_k} "
            f"combinations from {len(batch)} leaders"
        )
        prompt = build_combination_proposal_prompt(batch, causal_reports, top_k=top_k)
        response = get_llm_response(
            prompt=prompt,
            system_message=COMBINATION_PROPOSAL_SYSTEM_MESSAGE,
            model=model,
            temperature=0.5,
        )
        batch_proposals = parse_combination_proposals(response, batch)
        logger.info(f"  Minibatch {batch_idx + 1}: received {len(batch_proposals)} proposals")
        all_proposals.extend(batch_proposals)

    # Sort by score descending
    all_proposals.sort(key=lambda p: p.score, reverse=True)

    # Filter by minimum score
    if combination_score_min > 0:
        before = len(all_proposals)
        all_proposals = [p for p in all_proposals if p.score >= combination_score_min]
        if before != len(all_proposals):
            logger.info(
                f"Score filter (min={combination_score_min}): "
                f"{before} -> {len(all_proposals)} proposals"
            )

    # Deduplicate (same pair regardless of order)
    seen: set = set()
    deduped: List[CombinationProposal] = []
    for p in all_proposals:
        key = tuple(sorted([p.parent_a_id, p.parent_b_id]))
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    all_proposals = deduped

    logger.info(f"Combination proposals: {len(all_proposals)} unique after dedup")

    # Resolve Individual objects
    result: List[Tuple[CombinationProposal, Individual, Individual]] = []
    for proposal in all_proposals:
        parent_a = pop_lookup.get(proposal.parent_a_id)
        parent_b = pop_lookup.get(proposal.parent_b_id)
        if parent_a is None or parent_b is None:
            logger.warning(
                f"Could not resolve individuals for proposal "
                f"{proposal.parent_a_id[:8]}... x {proposal.parent_b_id[:8]}..."
            )
            continue
        result.append((proposal, parent_a, parent_b))

    return result


def save_combination_proposals(
    proposals: List[CombinationProposal], output_dir: str, iteration: int = 0
) -> str:
    """Save combination proposals to a JSON file (per iteration)."""
    path = os.path.join(output_dir, f"combination_proposals_iter{iteration}.json")
    data = [
        {
            "parent_a_id": p.parent_a_id,
            "parent_b_id": p.parent_b_id,
            "score": p.score,
            "justification": p.justification,
            "complementary_strengths": p.complementary_strengths,
        }
        for p in proposals
    ]
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(proposals)} combination proposals to {path}")
    return path


# ---------------------------------------------------------------------------
# Stage 4: Combination (Crossover) — executed only for LLM-proposed pairs
# ---------------------------------------------------------------------------

CROSSOVER_SYSTEM_MESSAGE = (
    "You are an expert SAT solver researcher performing genetic crossover to create "
    "improved solver heuristics. You combine the strengths of parent algorithms while "
    "avoiding their weaknesses. The goal is to lower the PAR2 score. Ignore PAR2 in your analysis if it shows as Inf."
)


def build_crossover_prompt(
    parent_a: Individual,
    parent_b: Individual,
    causal_a: CausalReport,
    causal_b: CausalReport,
) -> str:
    """Build the prompt for crossover of two parents."""
    target_fn = parent_a.target_function or "kissat_restarting"
    return f"""You are performing genetic crossover of two SAT solver heuristic algorithms.
Each parent has a causal analysis identifying its strengths and weaknesses.
Your task is to design a NEW offspring algorithm that combines the strengths of both parents while avoiding their weaknesses.

### Parent A (PAR2: {parent_a.par2:.2f}):
Algorithm:
{parent_a.algorithm_json}

Causal Analysis:
- Strengths: {json.dumps(causal_a.strengths, indent=2)}
- Weaknesses: {json.dumps(causal_a.weaknesses, indent=2)}
- Key Mechanisms: {json.dumps(causal_a.key_mechanisms, indent=2)}
- Improvement Suggestions: {json.dumps(causal_a.improvement_suggestions, indent=2)}

### Parent B (PAR2: {parent_b.par2:.2f}):
Algorithm:
{parent_b.algorithm_json}

Causal Analysis:
- Strengths: {json.dumps(causal_b.strengths, indent=2)}
- Weaknesses: {json.dumps(causal_b.weaknesses, indent=2)}
- Key Mechanisms: {json.dumps(causal_b.key_mechanisms, indent=2)}
- Improvement Suggestions: {json.dumps(causal_b.improvement_suggestions, indent=2)}

Design a new offspring algorithm that:
1. Combines the key strengths identified in both parents
2. Specifically avoids the identified weaknesses of both parents
3. Incorporates the improvement suggestions where applicable
4. Is concrete and implementable (specific thresholds, conditions, state tracking)
5. Uses step-by-step format (Step 1: ..., Step 2: ...)

The offspring should target the function: {target_fn}

Output your answer as JSON only:
```json
{{
    "name": "Brief algorithm name (<=6 words)",
    "algorithm": "Complete algorithmic description. Step 1: ... Step 2: ...",
    "reason": "Why this combination improves on both parents.",
    "target_function": "{target_fn}",
    "parent_a_strengths_used": ["which strengths from Parent A were incorporated"],
    "parent_b_strengths_used": ["which strengths from Parent B were incorporated"]
}}
```"""


def parse_crossover_response(
    response: str, parent_a_id: str, parent_b_id: str, target_function: str = "kissat_restarting"
) -> Optional[OffspringResult]:
    """Parse the LLM crossover response into an OffspringResult."""
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        json_str = response.strip()

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse crossover JSON for pair ({parent_a_id[:8]}, {parent_b_id[:8]})")
        return None

    tf = data.get("target_function", target_function)
    algo_spec = {
        "name": data.get("name", "Crossover Offspring"),
        "algorithm": data.get("algorithm", ""),
        "target_function": tf,
    }
    algorithm_json = json.dumps(algo_spec, ensure_ascii=False)
    algorithm_id = get_id(algorithm_json)

    return OffspringResult(
        parent_a_id=parent_a_id,
        parent_b_id=parent_b_id,
        algorithm_json=algorithm_json,
        algorithm_id=algorithm_id,
        reason=data.get("reason", ""),
        parent_a_strengths_used=data.get("parent_a_strengths_used", []),
        parent_b_strengths_used=data.get("parent_b_strengths_used", []),
        target_function=tf,
    )


def select_pairs(
    population: List[Individual],
    top_k: Optional[int] = None,
    max_pairs: Optional[int] = None,
) -> List[Tuple[Individual, Individual]]:
    """
    Select pairs for crossover based on PAR2 ranking.

    Selects the top_k individuals by PAR2 (lowest = best), then generates
    all pairwise combinations. Limits to max_pairs if specified.
    """
    ranked = sorted(population, key=lambda ind: ind.par2)

    if top_k is not None and top_k < len(ranked):
        ranked = ranked[:top_k]

    pairs = list(combinations(ranked, 2))
    logger.info(f"Generated {len(pairs)} pairs from top {len(ranked)} individuals")

    if max_pairs is not None and len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]
        logger.info(f"Limited to {max_pairs} pairs")

    return pairs


def perform_crossover(
    pairs: List[Tuple[Individual, Individual]],
    causal_reports: Dict[str, CausalReport],
    model: str = "gpt-4.1",
) -> List[OffspringResult]:
    """
    Stage 2: Perform crossover (combination) for each pair of parents.
    Returns list of OffspringResult.
    """
    logger.info(f"Performing crossover for {len(pairs)} pairs")
    offspring_list: List[OffspringResult] = []

    for i, (parent_a, parent_b) in enumerate(pairs):
        logger.info(
            f"[{i+1}/{len(pairs)}] Crossing "
            f"{parent_a.algorithm_id[:12]}... (PAR2={parent_a.par2:.2f}) x "
            f"{parent_b.algorithm_id[:12]}... (PAR2={parent_b.par2:.2f})"
        )

        causal_a = causal_reports.get(parent_a.algorithm_id)
        causal_b = causal_reports.get(parent_b.algorithm_id)

        if causal_a is None or causal_b is None:
            logger.warning(f"  Missing causal report, skipping pair")
            continue

        prompt = build_crossover_prompt(parent_a, parent_b, causal_a, causal_b)
        response = get_llm_response(
            prompt=prompt,
            system_message=CROSSOVER_SYSTEM_MESSAGE,
            model=model,
            temperature=0.7,
        )

        target_fn = parent_a.target_function or "kissat_restarting"
        offspring = parse_crossover_response(
            response, parent_a.algorithm_id, parent_b.algorithm_id, target_function=target_fn
        )
        if offspring is not None:
            offspring_list.append(offspring)
            logger.info(f"  Offspring: {offspring.algorithm_id[:16]}...")
        else:
            logger.warning(f"  Failed to parse offspring")

    logger.info(f"Crossover produced {len(offspring_list)} offspring")
    return offspring_list


# ---------------------------------------------------------------------------
# Stage 3b: Rubric Scoring Filter (before code generation)
# ---------------------------------------------------------------------------

RUBRIC_SYSTEM_MESSAGE = (
    "You are an expert SAT solver researcher evaluating the quality of a genetically-evolved "
    "solver heuristic. You compare the offspring algorithm against its two parent algorithms "
    "and provide rigorous multi-dimensional scoring.  The goal is to lower the PAR2 score. Ignore PAR2 in your analysis if it shows as Inf."
)


def build_rubric_prompt(
    offspring: OffspringResult,
    parent_a: Individual,
    parent_b: Individual,
    causal_a: CausalReport,
    causal_b: CausalReport,
) -> str:
    """Build the prompt for rubric-based evaluation of an offspring."""
    dimensions_text = "\n".join(
        f"  {i+1}. **{dim.replace('_', ' ').title()}**: {desc}"
        for i, (dim, desc) in enumerate(RUBRIC_DIMENSIONS.items())
    )

    return f"""Evaluate the quality of the following offspring SAT solver algorithm that was produced
by combining (crossover) two parent algorithms. Score the offspring on each dimension below.

### Parent A (PAR2: {parent_a.par2:.2f}):
Algorithm:
{parent_a.algorithm_json}

Causal Analysis:
- Strengths: {json.dumps(causal_a.strengths, indent=2)}
- Weaknesses: {json.dumps(causal_a.weaknesses, indent=2)}
- Key Mechanisms: {json.dumps(causal_a.key_mechanisms, indent=2)}

### Parent B (PAR2: {parent_b.par2:.2f}):
Algorithm:
{parent_b.algorithm_json}

Causal Analysis:
- Strengths: {json.dumps(causal_b.strengths, indent=2)}
- Weaknesses: {json.dumps(causal_b.weaknesses, indent=2)}
- Key Mechanisms: {json.dumps(causal_b.key_mechanisms, indent=2)}

### Offspring Algorithm (product of crossover):
{offspring.algorithm_json}

Crossover rationale: {offspring.reason}
Strengths used from Parent A: {json.dumps(offspring.parent_a_strengths_used)}
Strengths used from Parent B: {json.dumps(offspring.parent_b_strengths_used)}

---

### Scoring Rubric — evaluate the offspring on each dimension (1-10):

{dimensions_text}

For each dimension, provide:
- A score from 1-10
- A brief justification (1-2 sentences) explaining the score

Output your evaluation as JSON only:
```json
{{
    "improvement_over_parents": {{"score": <1-10>, "justification": "..."}},
    "algorithmic_novelty": {{"score": <1-10>, "justification": "..."}},
    "mechanistic_soundness": {{"score": <1-10>, "justification": "..."}},
    "implementation_feasibility": {{"score": <1-10>, "justification": "..."}},
    "complementarity": {{"score": <1-10>, "justification": "..."}}
}}
```"""


def parse_rubric_response(
    response: str,
    offspring_id: str,
    parent_a_id: str,
    parent_b_id: str,
    weights: Optional[Dict[str, float]] = None,
) -> RubricScore:
    """Parse the LLM rubric evaluation response into a RubricScore."""
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        json_str = response.strip()

    score = RubricScore(
        offspring_id=offspring_id,
        parent_a_id=parent_a_id,
        parent_b_id=parent_b_id,
        raw_response=response,
    )

    try:
        data = json.loads(json_str)
        for dim in RUBRIC_DIMENSIONS:
            entry = data.get(dim, {})
            if isinstance(entry, dict):
                dim_score = float(entry.get("score", 0))
                justification = entry.get("justification", "")
            elif isinstance(entry, (int, float)):
                dim_score = float(entry)
                justification = ""
            else:
                dim_score = 0.0
                justification = ""
            dim_score = max(1.0, min(10.0, dim_score))
            setattr(score, dim, dim_score)
            score.justifications[dim] = justification
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse rubric JSON for offspring {offspring_id[:16]}")

    score.compute_weighted_total(weights)
    return score


def evaluate_offspring_rubric(
    offspring_list: List[OffspringResult],
    population: List[Individual],
    causal_reports: Dict[str, CausalReport],
    model: str = "gpt-4.1",
    weights: Optional[Dict[str, float]] = None,
) -> List[RubricScore]:
    """
    Stage 3b: Evaluate each offspring using the multi-dimensional rubric via LLM.

    This runs BEFORE code generation as a filter to avoid wasting computation
    on low-quality offspring designs.

    Returns list of RubricScore objects.
    """
    logger.info(f"Rubric-scoring {len(offspring_list)} offspring (pre-code-generation filter)")

    pop_lookup: Dict[str, Individual] = {ind.algorithm_id: ind for ind in population}
    rubric_scores: List[RubricScore] = []

    for i, offspring in enumerate(offspring_list):
        parent_a = pop_lookup.get(offspring.parent_a_id)
        parent_b = pop_lookup.get(offspring.parent_b_id)

        if parent_a is None or parent_b is None:
            logger.warning(
                f"[{i+1}/{len(offspring_list)}] Missing parent for offspring "
                f"{offspring.algorithm_id[:16]}, skipping rubric eval"
            )
            continue

        causal_a = causal_reports.get(offspring.parent_a_id)
        causal_b = causal_reports.get(offspring.parent_b_id)

        if causal_a is None or causal_b is None:
            logger.warning(
                f"[{i+1}/{len(offspring_list)}] Missing causal report for parents of "
                f"{offspring.algorithm_id[:16]}, skipping rubric eval"
            )
            continue

        logger.info(
            f"[{i+1}/{len(offspring_list)}] Rubric-scoring offspring "
            f"{offspring.algorithm_id[:16]}... "
            f"(parents: {parent_a.par2:.2f} x {parent_b.par2:.2f})"
        )

        prompt = build_rubric_prompt(offspring, parent_a, parent_b, causal_a, causal_b)
        response = get_llm_response(
            prompt=prompt,
            system_message=RUBRIC_SYSTEM_MESSAGE,
            model=model,
            temperature=0.3,
        )

        score = parse_rubric_response(
            response,
            offspring.algorithm_id,
            offspring.parent_a_id,
            offspring.parent_b_id,
            weights=weights,
        )
        rubric_scores.append(score)

        dim_str = ", ".join(
            f"{dim}={getattr(score, dim):.1f}" for dim in RUBRIC_DIMENSIONS
        )
        logger.info(f"  Scores: {dim_str} | weighted={score.weighted_total:.2f}")

    return rubric_scores


def filter_offspring_by_rubric(
    offspring_list: List[OffspringResult],
    rubric_scores: List[RubricScore],
    rubric_min: float = 6.0,
    keep_top_n: Optional[int] = None,
) -> List[OffspringResult]:
    """
    Filter offspring by rubric score before code generation.

    Only offspring whose weighted rubric score >= rubric_min are passed to
    code generation. Optionally limits to the top N by rubric score.

    Returns:
        Filtered list of OffspringResult objects, sorted by rubric score descending.
    """
    rubric_lookup: Dict[str, RubricScore] = {s.offspring_id: s for s in rubric_scores}

    passed: List[Tuple[OffspringResult, float]] = []
    filtered_out = 0

    for offspring in offspring_list:
        rs = rubric_lookup.get(offspring.algorithm_id)
        if rs is None:
            logger.debug(f"No rubric score for {offspring.algorithm_id[:16]}, filtering out")
            filtered_out += 1
            continue
        if rs.weighted_total < rubric_min:
            logger.info(
                f"  Filtered (rubric={rs.weighted_total:.2f} < min={rubric_min:.2f}): "
                f"{offspring.algorithm_id[:16]}"
            )
            filtered_out += 1
            continue
        passed.append((offspring, rs.weighted_total))

    # Sort by rubric score descending
    passed.sort(key=lambda x: x[1], reverse=True)

    if keep_top_n is not None and len(passed) > keep_top_n:
        passed = passed[:keep_top_n]

    result = [off for off, _ in passed]
    logger.info(
        f"Rubric filter: {len(offspring_list)} offspring -> "
        f"{len(result)} passed (rubric_min={rubric_min}"
        + (f", keep_top_n={keep_top_n}" if keep_top_n else "")
        + f"), {filtered_out} filtered out"
    )
    return result


# ---------------------------------------------------------------------------
# Stage 4: Code Generation for Offspring
# ---------------------------------------------------------------------------

def generate_offspring_code(
    offspring_list: List[OffspringResult],
    code_prompt_template_path: str,
    model: str = "gpt-4.1",
    population: Optional[List[Individual]] = None,
) -> Dict[str, str]:
    """
    Stage 4: Generate C code for each offspring algorithm that passed rubric filter.

    Uses the same coder prompt template as the main pipeline.
    If population is provided, parent C code is included in the prompt as a
    structural reference to reduce hallucinated struct fields / bad includes.
    Returns dict mapping algorithm_id -> code string.
    """
    logger.info(f"Generating code for {len(offspring_list)} rubric-filtered offspring")
    code_prompt_template = read_code_prompt_template(code_prompt_template_path)
    offspring_codes: Dict[str, str] = {}

    # Build a lookup map from algorithm_id -> Individual for parent code retrieval
    pop_map: Dict[str, Individual] = {}
    if population:
        pop_map = {ind.algorithm_id: ind for ind in population}

    for i, offspring in enumerate(offspring_list):
        logger.info(
            f"[{i+1}/{len(offspring_list)}] Generating code for {offspring.algorithm_id[:16]}..."
        )

        try:
            algo_spec = json.loads(offspring.algorithm_json)
            algorithm_text = algo_spec.get("algorithm", offspring.algorithm_json)
        except json.JSONDecodeError:
            algorithm_text = offspring.algorithm_json

        # Look up parent codes (only include if they exist and are non-empty)
        parent_a = pop_map.get(offspring.parent_a_id)
        parent_b = pop_map.get(offspring.parent_b_id)
        parent_a_code = parent_a.code if parent_a and parent_a.code else None
        parent_b_code = parent_b.code if parent_b and parent_b.code else None

        if parent_a_code or parent_b_code:
            logger.info(
                f"  Including parent code(s): A={'yes' if parent_a_code else 'no'}, "
                f"B={'yes' if parent_b_code else 'no'}"
            )

        prompt = generate_code_prompt(
            code_prompt_template,
            algorithm_text,
            parent_a_code=parent_a_code,
            parent_b_code=parent_b_code,
        )
        response = get_llm_response(
            prompt=prompt,
            system_message="You are an expert in writing C code for Kissat-based SAT Solvers.",
            model=model,
            temperature=0.5,
        )

        # Parse code from response
        # Try <code>...</code> tags first, then markdown block
        code = ""
        start = response.find("<code>")
        end = response.find("</code>")
        if start != -1 and end != -1 and end > start:
            code = response[start + len("<code>"): end].strip()
        elif "```c" in response:
            s = response.find("```c") + 4
            e = response.find("```", s)
            if e > s:
                code = response[s:e].strip()
        if not code:
            code = response

        offspring_codes[offspring.algorithm_id] = code
        logger.info(f"  Generated code ({len(code)} chars)")

    return offspring_codes


# ---------------------------------------------------------------------------
# Stage 5: Store Offspring in DB
# ---------------------------------------------------------------------------

def store_offspring(
    offspring_list: List[OffspringResult],
    offspring_codes: Dict[str, str],
    output_tag: str,
) -> List[Tuple[str, str]]:
    """
    Store offspring algorithms and codes in the database.
    Returns list of (algorithm_id, code_id) tuples for evaluation.
    """
    logger.info(f"Storing {len(offspring_list)} offspring with tag: {output_tag}")
    stored_pairs: List[Tuple[str, str]] = []

    for offspring in offspring_list:
        code_str = offspring_codes.get(offspring.algorithm_id)
        if code_str is None:
            logger.warning(f"No code for offspring {offspring.algorithm_id[:16]}, skipping")
            continue

        code_id = get_id(code_str)

        algo_result = AlgorithmResult(
            id=offspring.algorithm_id,
            algorithm=offspring.algorithm_json,
            status=AlgorithmStatus.CodeGenerated,
            last_updated=datetime.now(),
            prompt="genetic_evolution_crossover",
            par2=NOT_INITIALIZED,
            error_rate=NOT_INITIALIZED,
            other_metrics={
                "parent_a": offspring.parent_a_id,
                "parent_b": offspring.parent_b_id,
                "evolution_method": "causal_crossover",
            },
            code_id_list=[code_id],
            target_function=offspring.target_function,
            parent_id=offspring.parent_a_id,
        )
        update_algorithm_result(algo_result)
        update_router_table(CHATGPT_DATA_GENERATION_TABLE, offspring.algorithm_id, output_tag)

        code_result = CodeResult(
            id=code_id,
            algorithm_id=offspring.algorithm_id,
            code=code_str,
            status=CodeStatus.Generated,
            par2=None,
            last_updated=datetime.now(),
            build_success=NOT_INITIALIZED,
        )
        update_code_result(code_result)

        stored_pairs.append((offspring.algorithm_id, code_id))
        logger.info(f"  Stored {offspring.algorithm_id[:16]} -> code {code_id[:16]}")

    logger.info(f"Stored {len(stored_pairs)} offspring in DB")
    return stored_pairs


# ---------------------------------------------------------------------------
# Stage 6: Evaluate with Simulation (using EvaluationPipeline)
# ---------------------------------------------------------------------------

def evaluate_offspring(
    stored_pairs: List[Tuple[str, str]],
    generation_tag: Optional[str] = None,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Evaluate offspring by building solvers and submitting SLURM jobs.

    Uses EvaluationPipeline from evaluation.py. Build happens locally first;
    if build fails the code is excluded immediately.

    Returns:
        (successful_pairs, failed_pairs) where each is a list of
        (algorithm_id, code_id) tuples.
    """
    from llmsat.pipelines.evaluation import EvaluationPipeline

    logger.info(f"Evaluating {len(stored_pairs)} offspring via EvaluationPipeline")
    pipeline = EvaluationPipeline(generation_tag=generation_tag)

    successful: List[Tuple[str, str]] = []
    failed: List[Tuple[str, str]] = []

    for i, (algorithm_id, code_id) in enumerate(stored_pairs):
        logger.info(f"[{i+1}/{len(stored_pairs)}] Evaluating code {code_id[:16]}...")
        try:
            result = pipeline.run_single_solver(code_id)
            if result is None:
                logger.warning(f"  Build/submit FAILED for {code_id[:16]}")
                failed.append((algorithm_id, code_id))
            else:
                successful.append((algorithm_id, code_id))
                logger.info(f"  Build OK, SLURM jobs submitted for {code_id[:16]}")
        except Exception as e:
            logger.error(f"  Evaluation raised exception: {e}")
            failed.append((algorithm_id, code_id))

    logger.info(
        f"Evaluation: {len(successful)} submitted OK, {len(failed)} failed"
    )
    return successful, failed


def collect_par2_scores(
    successful_pairs: List[Tuple[str, str]],
    generation_tag: Optional[str] = None,
) -> Dict[str, float]:
    """
    Collect PAR2 scores for successfully built offspring.

    NOTE: PAR2 is only available after SLURM jobs complete. This function reads
    whatever is available from the DB or local solving_times files at call time.

    Returns dict mapping algorithm_id -> PAR2.
    """
    par2_scores: Dict[str, float] = {}
    for algo_id, code_id in successful_pairs:
        # Try DB first
        try:
            code_result = get_code_result(code_id)
            if code_result is not None and code_result.par2 is not None:
                par2_scores[algo_id] = code_result.par2
                continue
        except Exception:
            pass
        # Try local solving_times file
        try:
            algo_result = get_algorithm_result(algo_id)
            parent_id = algo_result.parent_id if algo_result else None
            times_path = get_solver_solving_times_path(
                algo_id, code_id,
                generation_tag=generation_tag,
                parent_id=parent_id,
            )
            if os.path.exists(times_path):
                with open(times_path, "r") as f:
                    times = json.load(f)
                if times:
                    par2_scores[algo_id] = sum(times.values()) / len(times)
        except Exception:
            pass
    return par2_scores


# ---------------------------------------------------------------------------
# Stage 7: PAR2-based Selection (preserve top-tier for next loop iteration)
# ---------------------------------------------------------------------------

def select_by_par2(
    offspring_list: List[OffspringResult],
    population: List[Individual],
    par2_scores: Dict[str, float],
    offspring_codes: Dict[str, str],
    par2_threshold: Optional[float] = None,
    keep_top_n: Optional[int] = None,
    generation: int = 1,
) -> Tuple[List[Individual], List[Individual]]:
    """
    Select offspring purely by PAR2 score.

    Logic:
      1. Only consider offspring present in par2_scores (build + eval succeeded).
      2. If par2_threshold is set, discard offspring whose PAR2 > threshold.
         If not set, discard offspring whose PAR2 >= best parent PAR2 (no improvement).
      3. Rank remaining by PAR2 ascending (lower = better).
      4. If keep_top_n is set, keep only the top N.

    Returns:
        (selected_individuals, all_evaluated_individuals)
        - selected_individuals: those passing the PAR2 gate (top-tier)
        - all_evaluated_individuals: all offspring that had PAR2 available
    """
    pop_lookup: Dict[str, Individual] = {ind.algorithm_id: ind for ind in population}

    # Determine default threshold: best PAR2 among parents in the current population
    best_parent_par2 = min((ind.par2 for ind in population), default=float("inf"))
    effective_threshold = par2_threshold if par2_threshold is not None else best_parent_par2

    all_evaluated: List[Individual] = []
    candidates: List[Tuple[OffspringResult, float]] = []

    for offspring in offspring_list:
        par2 = par2_scores.get(offspring.algorithm_id)
        if par2 is None:
            logger.debug(
                f"No PAR2 for {offspring.algorithm_id[:16]} "
                f"(SLURM may not have completed yet)"
            )
            continue

        code_str = offspring_codes.get(offspring.algorithm_id, "")
        code_id = get_id(code_str) if code_str else ""

        ind = Individual(
            algorithm_id=offspring.algorithm_id,
            algorithm_json=offspring.algorithm_json,
            code_id=code_id,
            code=code_str,
            par2=par2,
            target_function=offspring.target_function,
            parent_id=offspring.parent_a_id,
            generation=generation,
        )
        all_evaluated.append(ind)

        if par2 < effective_threshold:
            candidates.append((offspring, par2))
            logger.info(
                f"  IMPROVED {offspring.algorithm_id[:16]}: "
                f"PAR2={par2:.2f} < threshold={effective_threshold:.2f}"
            )
        else:
            logger.info(
                f"  No improvement {offspring.algorithm_id[:16]}: "
                f"PAR2={par2:.2f} >= threshold={effective_threshold:.2f}"
            )

    # Sort by PAR2 ascending
    candidates.sort(key=lambda x: x[1])

    if keep_top_n is not None and len(candidates) > keep_top_n:
        candidates = candidates[:keep_top_n]

    selected: List[Individual] = []
    for offspring, par2 in candidates:
        code_str = offspring_codes.get(offspring.algorithm_id, "")
        code_id = get_id(code_str) if code_str else ""
        ind = Individual(
            algorithm_id=offspring.algorithm_id,
            algorithm_json=offspring.algorithm_json,
            code_id=code_id,
            code=code_str,
            par2=par2,
            target_function=offspring.target_function,
            parent_id=offspring.parent_a_id,
            generation=generation,
        )
        selected.append(ind)

    logger.info(
        f"PAR2 selection: {len(offspring_list)} offspring -> "
        f"{len(all_evaluated)} evaluated -> {len(selected)} selected "
        f"(threshold={effective_threshold:.2f}"
        + (f", keep_top_n={keep_top_n}" if keep_top_n else "")
        + ")"
    )
    return selected, all_evaluated


# ---------------------------------------------------------------------------
# Output Helpers
# ---------------------------------------------------------------------------

def save_causal_reports(reports: Dict[str, CausalReport], output_dir: str) -> str:
    """Save causal reports to JSON file."""
    path = os.path.join(output_dir, "causal_reports.json")
    data = {}
    for algo_id, report in reports.items():
        data[algo_id] = {
            "algorithm_id": report.algorithm_id,
            "strengths": report.strengths,
            "weaknesses": report.weaknesses,
            "key_mechanisms": report.key_mechanisms,
            "improvement_suggestions": report.improvement_suggestions,
        }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved causal reports to {path}")
    return path


def load_causal_reports(output_dir: str) -> Dict[str, CausalReport]:
    """Load causal reports from a previously saved JSON file."""
    path = os.path.join(output_dir, "causal_reports.json")
    if not os.path.exists(path):
        logger.error(f"Causal reports file not found: {path}")
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    reports = {}
    for algo_id, entry in data.items():
        reports[algo_id] = CausalReport(
            algorithm_id=entry["algorithm_id"],
            strengths=entry.get("strengths", []),
            weaknesses=entry.get("weaknesses", []),
            key_mechanisms=entry.get("key_mechanisms", []),
            improvement_suggestions=entry.get("improvement_suggestions", []),
        )
    logger.info(f"Loaded {len(reports)} causal reports from {path}")
    return reports


def save_rubric_scores(rubric_scores: List[RubricScore], output_dir: str, iteration: int = 0) -> str:
    """Save rubric scores to JSON file (per iteration)."""
    path = os.path.join(output_dir, f"rubric_scores_iter{iteration}.json")
    data = []
    for rs in rubric_scores:
        data.append({
            "offspring_id": rs.offspring_id,
            "parent_a_id": rs.parent_a_id,
            "parent_b_id": rs.parent_b_id,
            "scores": rs.dimension_scores,
            "weighted_total": rs.weighted_total,
            "justifications": rs.justifications,
        })
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved rubric scores to {path}")
    return path


def save_crossover_results(
    offspring_list: List[OffspringResult], output_dir: str, iteration: int = 0
) -> str:
    """Save crossover results to JSON file (per iteration)."""
    path = os.path.join(output_dir, f"crossover_results_iter{iteration}.json")
    data = []
    for off in offspring_list:
        data.append({
            "algorithm_id": off.algorithm_id,
            "parent_a_id": off.parent_a_id,
            "parent_b_id": off.parent_b_id,
            "algorithm_json": off.algorithm_json,
            "reason": off.reason,
            "parent_a_strengths_used": off.parent_a_strengths_used,
            "parent_b_strengths_used": off.parent_b_strengths_used,
            "target_function": off.target_function,
        })
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved crossover results to {path}")
    return path


def save_offspring_codes(codes: Dict[str, str], output_dir: str, iteration: int = 0) -> str:
    """Save offspring codes to JSON file (per iteration)."""
    path = os.path.join(output_dir, f"offspring_codes_iter{iteration}.json")
    with open(path, "w") as f:
        json.dump(codes, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved offspring codes to {path}")
    return path


def save_top_tier(top_tier: List[Individual], output_dir: str) -> str:
    """Save the current top-tier individuals (selected offspring) to JSON."""
    path = os.path.join(output_dir, "top_tier.json")
    data = []
    for ind in top_tier:
        data.append({
            "algorithm_id": ind.algorithm_id,
            "algorithm_json": ind.algorithm_json,
            "code_id": ind.code_id,
            "par2": ind.par2,
            "target_function": ind.target_function,
            "parent_id": ind.parent_id,
            "generation": ind.generation,
        })
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(top_tier)} top-tier individuals to {path}")
    return path


# ---------------------------------------------------------------------------
# Main Evolution Loop
# ---------------------------------------------------------------------------

def run_evolution(
    generation_tag: str,
    code_prompt_path: str,
    output_tag: Optional[str] = None,
    folder: Optional[str] = None,
    top_k: Optional[int] = None,
    max_pairs: Optional[int] = None,  # kept for CLI compat, no longer used
    model: str = "gpt-4.1",
    baseline_par2: Optional[float] = None,
    evaluate: bool = False,
    causal_only: bool = False,
    skip_causal: bool = False,
    evaluate_only: bool = False,
    rubric_min: float = 6.0,
    rubric_keep_top_n: Optional[int] = None,
    par2_threshold: Optional[float] = None,
    par2_keep_top_n: Optional[int] = None,
    rubric_weights: Optional[Dict[str, float]] = None,  # kept for compat, no longer used
    max_iterations: int = 5,
    minibatch_size: int = DEFAULT_MINIBATCH_SIZE,
    prev_output_tags: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Run the full genetic evolution pipeline with an iterative loop.

    Pipeline (per iteration):
      1. Load population (leaders only, from DB or folder)
      2. Causal analysis — once on initial population, then for newly promoted individuals
      3. LLM-proposed combinations — single LLM call per minibatch proposes top-k pairs
         (replaces exhaustive pair enumeration + per-offspring rubric scoring)
      4. Crossover — generate offspring algorithm for each proposed pair
      5. Code generation — translate offspring algorithms to C code
      6. Evaluation — build solver and submit SLURM jobs (optional)
      7. Selection by PAR2 — keep offspring that improve over best parent

    Minibatching:
      If the population exceeds minibatch_size, it is split into batches of that size.
      Each batch gets its own LLM combination-proposal call; proposals are merged and
      deduplicated, then sorted by score before crossover.

    Loop continues until:
      - PAR2 cannot be improved further (no new selected offspring), OR
      - max_iterations is reached

    Args:
        generation_tag: Source population generation tag
        code_prompt_path: Path to the coder prompt template
        output_tag: Tag for offspring generation (default: {generation_tag}_gen1)
        folder: Path to outputs folder to load population from files instead of DB.
        top_k: Combinations to request per minibatch from the LLM (default: 5)
        max_pairs: Deprecated — no longer used; pairs are controlled by top_k.
        model: OpenAI model for LLM calls
        baseline_par2: Baseline PAR2 for comparison in causal analysis
        evaluate: Whether to build and evaluate offspring via SLURM
        causal_only: Only run causal analysis, skip combination loop
        skip_causal: Skip causal analysis, load from file
        rubric_min: Minimum combination score (1-10) to accept a proposed pair (default 6.0)
        rubric_keep_top_n: Keep at most N proposed combinations (default: all passing)
        par2_threshold: PAR2 hard gate for selection. If None, uses best parent PAR2.
        par2_keep_top_n: Keep at most N offspring in PAR2 selection (default: all selected)
        rubric_weights: Deprecated — no longer used.
        max_iterations: Maximum number of evolution loop iterations (default: 5)
        minibatch_size: Max leaders per LLM combination-proposal call (default: 10)

    Returns:
        Summary dict with results from each stage and iteration.
    """
    if output_tag is None:
        output_tag = f"{generation_tag}_gen1"

    output_dir = get_generation_output_dir(output_tag)
    logger.info(f"Evolution pipeline: {generation_tag} -> {output_tag}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Max iterations: {max_iterations}")

    summary: Dict[str, Any] = {
        "generation_tag": generation_tag,
        "output_tag": output_tag,
        "timestamp": datetime.now().isoformat(),
        "iterations": [],
    }

    # ------------------------------------------------------------------
    # evaluate_only: skip all LLM stages, re-run evaluation on offspring
    # already stored in DB under output_tag (e.g. output_tag_iter1, ...)
    # ------------------------------------------------------------------
    if evaluate_only:
        logger.info(f"evaluate_only mode: fetching stored offspring under '{output_tag}*'")
        for iteration in range(max_iterations):
            iter_output_tag = f"{output_tag}_iter{iteration + 1}"
            algorithm_ids = get_ids_from_router_table(
                CHATGPT_DATA_GENERATION_TABLE, iter_output_tag
            )
            if not algorithm_ids:
                logger.info(f"No stored offspring found for tag '{iter_output_tag}', stopping")
                break
            stored_pairs: List[Tuple[str, str]] = []
            for algo_id in algorithm_ids:
                algo = get_algorithm_result(algo_id)
                if algo and algo.code_id_list:
                    for code_id in algo.code_id_list:
                        stored_pairs.append((algo_id, code_id))
            logger.info(
                f"Iteration {iteration + 1}: found {len(stored_pairs)} "
                f"(algorithm_id, code_id) pairs under '{iter_output_tag}'"
            )
            iter_summary: Dict[str, Any] = {
                "iteration": iteration + 1,
                "stored": len(stored_pairs),
            }
            if stored_pairs:
                successful_pairs, failed_pairs = evaluate_offspring(
                    stored_pairs, generation_tag=iter_output_tag
                )
                iter_summary["evaluation"] = {
                    "build_success": len(successful_pairs),
                    "build_failed": len(failed_pairs),
                }
            summary["iterations"].append(iter_summary)
        return summary

    # ------------------------------------------------------------------
    # Stage 1: Load initial population
    # ------------------------------------------------------------------
    if folder:
        logger.info(f"Loading population from folder: {folder}")
        population = load_population_from_folder(folder, generation_tag=generation_tag)
    else:
        population = load_population(generation_tag)

    if not population:
        logger.error("No individuals loaded. Check generation tag and database.")
        return {"error": "Empty population"}
    
    # Load any existing causal reports cached in the current output dir (e.g. re-run)
    causal_reports: Dict[str, CausalReport] = {}
    existing_causal_path = os.path.join(output_dir, "causal_reports.json")
    if os.path.exists(existing_causal_path):
        causal_reports = load_causal_reports(output_dir)
        logger.info(f"Loaded {len(causal_reports)} existing causal reports from current output dir")

    # ------------------------------------------------------------------
    # Stage 1b: Merge improvements from previous run (if --prev_output_tags given)
    # ------------------------------------------------------------------
    if prev_output_tags:
        # Load from ALL iterations of prev_output_tags (iter1, iter2, ...)
        # until a tag with no stored offspring is found
        latest_v = -1 # assume improvement computed based on the latest v
        for prev_output_tag in prev_output_tags:
            prev_iter_tag = f"{prev_output_tag}_iter1"
            prev_individuals = load_population(prev_iter_tag)
            # prev_individuals.sort(key=lambda x: x.par2)
            logger.info(f"  Loaded {len(prev_individuals)} evaluated offspring from '{prev_iter_tag}'")
            for prev_ind in prev_individuals:
                prev_ind.generation = int(prev_output_tag.split('v')[-1])
                latest_v = max(latest_v, prev_ind.generation)
            population.extend(prev_individuals)
            # Load causal reports for this prev_output_tag if available
            prev_causal_path = os.path.join("outputs", prev_output_tag, "causal_reports.json")
            if os.path.exists(prev_causal_path):
                prev_reports = load_causal_reports(os.path.join("outputs", prev_output_tag))
                causal_reports.update(prev_reports)
                logger.info(f"  Loaded {len(prev_reports)} causal reports from '{prev_output_tag}'")

        population.sort(key=lambda x: x.par2)
        population = population[:par2_keep_top_n]

        best_original_par2 = min(ind.par2 for ind in population)
        # improvements = [ind for ind in prev_individuals if ind.par2 < best_original_par2]

        individual_generation = [p.generation for p in population]
        generation_counter = Counter(individual_generation)
        logger_str = f"Current best PAR2 is: {best_original_par2}, updated population from all iterations (top-{par2_keep_top_n}): gathering "
        
        for ngen, cnt in generation_counter.items():
            logger_str += f"{cnt} individuals from {ngen}-th generation, "
        logger_str = logger_str[:-2] + '.'
        logger.info(logger_str)

        improvements = [p for p in population if p.generation == latest_v]

        # Early stop: if prev run produced offspring but none beat the original, stagnated
        if len(improvements) == 0:
            logger.info("No improvement over original population from previous run — stopping.")
            summary["status"] = "no_improvement"
            summary_path = os.path.join(output_dir, "evolution_summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            return summary

        summary["prev_output_tags"] = prev_output_tags
        summary["prev_improvements_added"] = len(improvements)

    summary["population_size"] = len(population)
    summary["par2_stats"] = {
        "best": min(ind.par2 for ind in population),
        "worst": max(ind.par2 for ind in population),
        "mean": sum(ind.par2 for ind in population) / len(population),
    }
    logger.info(
        f"Population PAR2: best={summary['par2_stats']['best']:.2f}, "
        f"worst={summary['par2_stats']['worst']:.2f}, "
        f"mean={summary['par2_stats']['mean']:.2f}"
    )

    # ------------------------------------------------------------------
    # Stage 2: Causal Analysis (once on the initial population)
    # ------------------------------------------------------------------
    if skip_causal:
        logger.info("Skipping causal analysis, loading from file")
        if not causal_reports:
            causal_reports = load_causal_reports(output_dir)
            logger.error("No causal reports found. Run without --skip_causal first.")
            return {"error": "No causal reports"}
        else:
            logger.info(f"Using {len(causal_reports)} causal reports (pre-loaded or from file)")
    else:
        new_individuals = [ind for ind in population if ind.algorithm_id not in causal_reports]
        if new_individuals:
            logger.info(
                f"Generating causal reports for {len(new_individuals)} new individuals "
                f"({len(population) - len(new_individuals)} already loaded)"
            )
            new_reports = generate_causal_reports(new_individuals, model=model, baseline_par2=baseline_par2)
            causal_reports.update(new_reports)
        else:
            logger.info(f"All {len(population)} individuals already have causal reports, skipping generation")
        save_causal_reports(causal_reports, output_dir)

    summary["causal_reports_count"] = len(causal_reports)

    if causal_only:
        logger.info("Causal-only mode: stopping after causal analysis")
        summary["mode"] = "causal_only"
        summary_path = os.path.join(output_dir, "evolution_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    # ------------------------------------------------------------------
    # Evolution Loop: Combination -> Rubric Filter -> Code Gen -> Eval -> PAR2 Select
    # ------------------------------------------------------------------
    # current_population includes original population + top-tier selected offspring
    current_population = list(population)
    # top_tier: best offspring accumulated across iterations (with PAR2)
    top_tier: List[Individual] = []
    best_par2_so_far = min(ind.par2 for ind in population)

    for iteration in range(max_iterations):
        logger.info(f"\n{'='*60}")
        logger.info(f"EVOLUTION ITERATION {iteration + 1}/{max_iterations}")
        logger.info(f"Current population size: {len(current_population)}")
        logger.info(f"Best PAR2 so far: {best_par2_so_far:.2f}")
        logger.info(f"{'='*60}")

        iter_output_tag = f"{output_tag}_iter{iteration + 1}"
        iter_summary: Dict[str, Any] = {
            "iteration": iteration + 1,
            "population_size": len(current_population),
            "best_par2_entering": best_par2_so_far,
        }

        # Generate causal reports for any new members of current_population
        # that don't already have one (e.g. top-tier offspring from prior iterations)
        missing_causal = [
            ind for ind in current_population
            if ind.algorithm_id not in causal_reports
        ]
        if missing_causal:
            logger.info(
                f"Generating causal reports for {len(missing_causal)} new population members "
                f"(promoted from previous iteration)"
            )
            new_causal = generate_causal_reports(missing_causal, model=model, baseline_par2=baseline_par2)
            causal_reports.update(new_causal)
            save_causal_reports(causal_reports, output_dir)

        # Stage 3: LLM-proposed combinations
        # The LLM receives all leaders + their causal analyses and proposes the top-k
        # most promising pairs. For large populations, minibatching is used.
        effective_top_k = top_k if top_k is not None else 5
        proposals_with_pairs = propose_combinations_llm(
            current_population,
            causal_reports,
            top_k=effective_top_k,
            minibatch_size=minibatch_size,
            combination_score_min=rubric_min,
            model=model,
        )

        # Apply keep_top_n cap after score filter
        if rubric_keep_top_n is not None and len(proposals_with_pairs) > rubric_keep_top_n:
            proposals_with_pairs = proposals_with_pairs[:rubric_keep_top_n]
            logger.info(f"Capped proposals to top {rubric_keep_top_n}")

        proposals = [p for p, _, _ in proposals_with_pairs]
        pairs = [(pa, pb) for _, pa, pb in proposals_with_pairs]

        save_combination_proposals(proposals, output_dir, iteration=iteration + 1)

        iter_summary["combination_proposals"] = {
            "proposals_count": len(proposals),
            "combination_score_min": rubric_min,
            "avg_score": (
                sum(p.score for p in proposals) / len(proposals) if proposals else 0.0
            ),
        }

        if not pairs:
            logger.warning("No combination proposals generated, stopping loop")
            iter_summary["status"] = "no_proposals"
            summary["iterations"].append(iter_summary)
            break

        # Stage 4: Crossover — generate offspring algorithm for each proposed pair
        offspring_list = perform_crossover(pairs, causal_reports, model=model)
        save_crossover_results(offspring_list, output_dir, iteration=iteration + 1)

        iter_summary["crossover"] = {
            "pairs_attempted": len(pairs),
            "offspring_produced": len(offspring_list),
        }

        if not offspring_list:
            logger.warning("No offspring from crossover, stopping loop")
            iter_summary["status"] = "no_offspring"
            summary["iterations"].append(iter_summary)
            break

        # Stage 5: Code generation for all offspring (LLM already filtered via proposals)
        # Pass current_population so parent C code can be included as reference
        filtered_offspring = offspring_list  # all proposed offspring proceed to code gen
        offspring_codes = generate_offspring_code(
            filtered_offspring, code_prompt_path, model=model, population=current_population
        )
        save_offspring_codes(offspring_codes, output_dir, iteration=iteration + 1)

        iter_summary["code_generation"] = {
            "codes_generated": len(offspring_codes),
        }

        # Stage 5: Store in DB
        stored_pairs = store_offspring(filtered_offspring, offspring_codes, iter_output_tag)
        iter_summary["stored"] = len(stored_pairs)

        # Stage 6: Evaluate with simulation
        if evaluate and stored_pairs:
            logger.info(f"Evaluating {len(stored_pairs)} offspring (build + SLURM)")
            successful_pairs, failed_pairs = evaluate_offspring(
                stored_pairs, generation_tag=iter_output_tag
            )

            iter_summary["evaluation"] = {
                "build_success": len(successful_pairs),
                "build_failed": len(failed_pairs),
            }

            # Collect PAR2 scores (may be partial if SLURM hasn't finished)
            par2_scores = collect_par2_scores(successful_pairs, generation_tag=iter_output_tag)
            logger.info(f"PAR2 scores available for {len(par2_scores)}/{len(successful_pairs)} offspring")

            # Stage 7: PAR2-based selection
            selected_individuals, all_evaluated = select_by_par2(
                filtered_offspring,
                current_population,
                par2_scores,
                offspring_codes,
                par2_threshold=par2_threshold,
                keep_top_n=par2_keep_top_n,
                generation=iteration + 1,
            )

            iter_summary["par2_selection"] = {
                "evaluated": len(all_evaluated),
                "selected": len(selected_individuals),
                "par2_threshold": par2_threshold,
                "selected_offspring": [
                    {
                        "algorithm_id": ind.algorithm_id,
                        "par2": ind.par2,
                        "generation": ind.generation,
                    }
                    for ind in selected_individuals
                ],
            }

            if selected_individuals:
                # Update top_tier: keep best unique individuals by algorithm_id
                existing_ids = {ind.algorithm_id for ind in top_tier}
                for ind in selected_individuals:
                    if ind.algorithm_id not in existing_ids:
                        top_tier.append(ind)
                        existing_ids.add(ind.algorithm_id)

                # Sort top_tier by PAR2 ascending and keep best
                top_tier.sort(key=lambda x: x.par2)

                new_best = top_tier[0].par2
                if new_best < best_par2_so_far:
                    logger.info(
                        f"PAR2 improved: {best_par2_so_far:.2f} -> {new_best:.2f} "
                        f"(delta={best_par2_so_far - new_best:.2f})"
                    )
                    best_par2_so_far = new_best
                    iter_summary["par2_improved"] = True
                    iter_summary["new_best_par2"] = new_best
                else:
                    logger.info(f"No PAR2 improvement this iteration (best={best_par2_so_far:.2f})")
                    iter_summary["par2_improved"] = False

                # Merge top-tier offspring into current_population for next iteration
                # so that top-tier offspring can be combined with each other
                current_ids = {ind.algorithm_id for ind in current_population}
                new_members = [ind for ind in selected_individuals if ind.algorithm_id not in current_ids]
                current_population.extend(new_members)
                logger.info(
                    f"Added {len(new_members)} new individuals to population "
                    f"(total: {len(current_population)})"
                )

                # Update causal reports for new members
                if new_members:
                    logger.info(f"Generating causal reports for {len(new_members)} new individuals")
                    new_reports = generate_causal_reports(new_members, model=model, baseline_par2=baseline_par2)
                    causal_reports.update(new_reports)

                save_top_tier(top_tier, output_dir)
                iter_summary["status"] = "improved" if iter_summary.get("par2_improved") else "no_improvement"

            else:
                logger.info("No offspring selected by PAR2 this iteration")
                iter_summary["status"] = "no_par2_selection"
                iter_summary["par2_improved"] = False

                if not all_evaluated:
                    # SLURM may not have finished; do not stop the loop prematurely
                    logger.warning(
                        "No PAR2 scores available yet (SLURM may not have completed). "
                        "Consider re-running after SLURM jobs finish."
                    )
                    summary["iterations"].append(iter_summary)
                    break

        else:
            if not evaluate:
                logger.info(
                    "Evaluation skipped. Run evaluation.py separately:\n"
                    f"  python src/llmsat/pipelines/evaluation.py "
                    f"--run_all --generation_tag {iter_output_tag}"
                )
            iter_summary["evaluation"] = "skipped"
            iter_summary["par2_selection"] = "skipped (no evaluation)"
            iter_summary["status"] = "eval_skipped"
            summary["iterations"].append(iter_summary)
            # Without evaluation we can't loop meaningfully
            break

        summary["iterations"].append(iter_summary)

        # Check stopping condition: no improvement in this iteration
        if not iter_summary.get("par2_improved", False) and evaluate:
            if all_evaluated:
                logger.info(
                    f"Stopping: PAR2 could not be improved further "
                    f"(best={best_par2_so_far:.2f})"
                )
                break

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    summary["final_best_par2"] = best_par2_so_far
    summary["top_tier_count"] = len(top_tier)
    summary["top_tier"] = [
        {
            "algorithm_id": ind.algorithm_id,
            "par2": ind.par2,
            "generation": ind.generation,
        }
        for ind in top_tier
    ]

    summary_path = os.path.join(output_dir, "evolution_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved evolution summary to {summary_path}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Genetic Evolution Pipeline for LLM-SAT solver heuristics"
    )
    parser.add_argument(
        "--generation_tag",
        type=str,
        default=None,
        help=(
            "Source population generation tag (e.g., gemini_trial1). "
            "Required when loading from DB. When --folder is used, auto-derived "
            "from the folder name if omitted."
        ),
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help=(
            "Path to outputs folder to load population from files instead of DB. "
            "e.g. 'outputs/gemini_trial1'. "
            "PAR2 scores are loaded from local solving_times JSON files or DB."
        ),
    )
    parser.add_argument(
        "--code_prompt_path",
        type=str,
        default="data/prompts/coder_prompt.txt",
        help="Path to the coder prompt template",
    )
    parser.add_argument(
        "--output_tag",
        type=str,
        default=None,
        help="Tag for offspring generation (default: {generation_tag}_gen1)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help=(
            "Number of combinations to request per minibatch from the LLM (default: 5). "
            "The LLM proposes this many pairs per batch; actual pairs used may be fewer "
            "after score filtering."
        ),
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="Deprecated — no longer used. Pairs are controlled by --top_k.",
    )
    parser.add_argument(
        "--minibatch_size",
        type=int,
        default=DEFAULT_MINIBATCH_SIZE,
        help=(
            f"Max leaders per LLM combination-proposal call (default: {DEFAULT_MINIBATCH_SIZE}). "
            "If the population exceeds this, it is split into minibatches of this size."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1",
        help="OpenAI model for LLM calls",
    )
    parser.add_argument(
        "--baseline_par2",
        type=float,
        default=None,
        help="Baseline PAR2 score for comparison in causal analysis",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=False,
        help="Build and evaluate offspring via SLURM after generation",
    )
    parser.add_argument(
        "--causal_only",
        action="store_true",
        default=False,
        help="Only run causal analysis (Stage 1), skip crossover loop",
    )
    parser.add_argument(
        "--skip_causal",
        action="store_true",
        default=False,
        help="Skip causal analysis and load from existing file",
    )
    parser.add_argument(
        "--evaluate_only",
        action="store_true",
        default=False,
        help=(
            "Skip all LLM stages (causal, crossover, rubric, codegen) and directly "
            "run evaluation on offspring already stored in DB under output_tag. "
            "Requires --output_tag (or derives it as {generation_tag}_gen1). "
            "Iterates over output_tag_iter1, output_tag_iter2, ... up to --max_iterations."
        ),
    )
    parser.add_argument(
        "--rubric_min",
        type=float,
        default=6.0,
        help=(
            "Minimum combination score (1-10) for LLM-proposed pairs to proceed to crossover "
            "(default: 6.0). Proposals below this score are discarded."
        ),
    )
    parser.add_argument(
        "--rubric_keep_top_n",
        type=int,
        default=None,
        help=(
            "Keep at most N proposed combinations after score filtering (default: all passing). "
            "Applied after --rubric_min filter, taking the highest-scored proposals."
        ),
    )
    parser.add_argument(
        "--par2_threshold",
        type=float,
        default=None,
        help=(
            "PAR2 hard gate for selection. Offspring with PAR2 >= this are discarded. "
            "Default: uses best parent PAR2 (only improvements are kept)."
        ),
    )
    parser.add_argument(
        "--par2_keep_top_n",
        type=int,
        default=None,
        help="Keep at most N offspring in PAR2 selection per iteration (default: all selected)",
    )
    parser.add_argument(
        "--prev_output_tags",
        type=str,
        nargs="+", 
        default=None,
        help=(
            "Output tag of the previous run to load evaluated offspring from. "
            "Loads all tags that have PAR2 in DB, "
            "keeps only those that improved over the original population best PAR2. "
            "Stops early if no improvements found (evolution stagnated)."
        ),
    )

    args = parser.parse_args()

    # Resolve generation_tag
    if args.generation_tag is None:
        if args.folder:
            folder_norm = args.folder.rstrip("/\\")
            parts = folder_norm.replace("\\", "/").split("/")
            for part in reversed(parts):
                if (
                    not part.startswith("batch_")
                    and part not in ("outputs", "batch_batches", "")
                ):
                    args.generation_tag = part
                    break
            if args.generation_tag is None:
                args.generation_tag = os.path.basename(folder_norm)
            logger.info(f"Auto-derived generation_tag from folder: {args.generation_tag}")
        else:
            parser.error("--generation_tag is required when not using --folder")

    summary = run_evolution(
        generation_tag=args.generation_tag,
        code_prompt_path=args.code_prompt_path,
        output_tag=args.output_tag,
        folder=args.folder,
        top_k=args.top_k,
        max_pairs=args.max_pairs,
        model=args.model,
        baseline_par2=args.baseline_par2,
        evaluate=args.evaluate,
        causal_only=args.causal_only,
        skip_causal=args.skip_causal,
        evaluate_only=args.evaluate_only,
        rubric_min=args.rubric_min,
        rubric_keep_top_n=args.rubric_keep_top_n,
        par2_threshold=args.par2_threshold,
        par2_keep_top_n=args.par2_keep_top_n,
        max_iterations=1,  # always 1 per run; SLURM is async so multi-iter must be driven manually
        minibatch_size=args.minibatch_size,
        prev_output_tags=args.prev_output_tags,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("GENETIC EVOLUTION SUMMARY")
    print("=" * 60)
    print(f"Source: {summary.get('generation_tag')}")
    print(f"Output: {summary.get('output_tag')}")
    print(f"Population: {summary.get('population_size', 0)} individuals")

    par2 = summary.get("par2_stats", {})
    if par2:
        print(f"Initial PAR2: best={par2.get('best', 'N/A'):.2f}, mean={par2.get('mean', 'N/A'):.2f}")

    print(f"Final best PAR2: {summary.get('final_best_par2', 'N/A')}")
    print(f"Top-tier offspring: {summary.get('top_tier_count', 0)}")

    for i, iter_info in enumerate(summary.get("iterations", [])):
        print(f"\n--- Iteration {i+1} ---")
        proposals_info = iter_info.get("combination_proposals", {})
        if isinstance(proposals_info, dict):
            print(f"  Combination proposals: {proposals_info.get('proposals_count', 0)} "
                  f"(score_min={proposals_info.get('combination_score_min', 'N/A')}, "
                  f"avg_score={proposals_info.get('avg_score', 0):.2f})")
        crossover = iter_info.get("crossover", {})
        if isinstance(crossover, dict):
            print(f"  Crossover: {crossover.get('pairs_attempted', 0)} pairs -> "
                  f"{crossover.get('offspring_produced', 0)} offspring")
        code_gen = iter_info.get("code_generation", {})
        if isinstance(code_gen, dict):
            print(f"  Code generation: {code_gen.get('codes_generated', 0)} codes")
        eval_info = iter_info.get("evaluation", {})
        if isinstance(eval_info, dict):
            print(f"  Evaluation: {eval_info.get('build_success', 0)} built OK, "
                  f"{eval_info.get('build_failed', 0)} failed")
        par2_sel = iter_info.get("par2_selection", {})
        if isinstance(par2_sel, dict):
            print(f"  PAR2 selection: {par2_sel.get('evaluated', 0)} evaluated -> "
                  f"{par2_sel.get('selected', 0)} selected")
            if iter_info.get("par2_improved"):
                print(f"  *** PAR2 improved! New best: {iter_info.get('new_best_par2', 'N/A'):.2f} ***")
        print(f"  Status: {iter_info.get('status', 'unknown')}")

    top_tier = summary.get("top_tier", [])
    if top_tier:
        print(f"\nTop-tier offspring (sorted by PAR2):")
        for entry in top_tier[:10]:
            print(f"  {entry['algorithm_id'][:16]}... PAR2={entry['par2']:.2f} "
                  f"(gen={entry['generation']})")

    print("=" * 60)


if __name__ == "__main__":
    main()
