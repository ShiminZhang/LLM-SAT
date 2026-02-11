"""
Genetic Evolution Pipeline for LLM-SAT.

Implements population-based evolution of SAT solver heuristics:
  1. Causal Analysis: LLM analyzes what makes each solution perform well/poorly
  2. Crossover: LLM combines strengths of parent pairs into offspring algorithms
  3. Code Generation: LLM translates ALL offspring algorithms into C code
  4. Evaluation: Build solver and evaluate on SAT benchmarks via SLURM.
     Buggy builds (compilation failures) are filtered out here.
  5. Rubric Selection: LLM scores successfully-built offspring on a
     multi-dimensional rubric (improvement, novelty, soundness, feasibility,
     complementarity). PAR2 acts as a hard threshold gate, rubric score
     ranks and filters the survivors.

Usage:
    python src/llmsat/pipelines/genetic_evolution.py \
        --generation_tag controlled_mutation \
        --code_prompt_path data/prompts/coder_prompt.txt \
        --top_k 10 \
        --model gpt-4.1 \
        --rubric_min 5.0 \
        --rubric_keep_top_n 10 \
        --evaluate
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
from llmsat.utils.chatgpt_helper import get_response_from_chatgpt
from llmsat.utils.paths import get_generation_output_dir, get_solver_solving_times_path
from llmsat.pipelines.chatgpt_data_generation import (
    parse_algorithm_response,
    parse_code_response,
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
    logger.info(f"Found {len(algorithm_ids)} algorithms in generation tag")

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


def _find_batch_dir(generation_tag: str) -> Optional[str]:
    """Find the batch_batch_* subdirectory inside outputs/{generation_tag}/."""
    gen_dir = f"outputs/{generation_tag}"
    if not os.path.isdir(gen_dir):
        return None
    for entry in os.listdir(gen_dir):
        if entry.startswith("batch_") and os.path.isdir(os.path.join(gen_dir, entry)):
            return os.path.join(gen_dir, entry)
    return None


def _extract_text_from_batch_line(line_json: dict) -> Optional[str]:
    """Extract the assistant text from a single OpenAI batch output JSON line."""
    resp = line_json.get("response") or line_json
    body = resp.get("body", resp) if isinstance(resp, dict) else resp
    # Try output_text shortcut
    if isinstance(body, dict):
        ot = body.get("output_text")
        if ot:
            return ot
    # Traverse output -> content -> output_text
    outputs = None
    if isinstance(body, dict):
        outputs = body.get("output") or body.get("outputs")
    if isinstance(outputs, list):
        for item in outputs:
            content = item.get("content") if isinstance(item, dict) else None
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        text = part.get("text")
                        if text:
                            return text
    return None


def _load_par2_for_algorithm(algorithm_id: str, code_id: str) -> Optional[float]:
    """
    Try to get PAR2 score from multiple sources:
      1. Database (get_code_result)
      2. Local solving_times JSON file
    """
    # Try DB first
    try:
        code_result = get_code_result(code_id)
        if code_result is not None and code_result.par2 is not None:
            return code_result.par2
    except Exception:
        pass

    # Try local solving_times file
    try:
        times_path = get_solver_solving_times_path(algorithm_id, code_id)
        if os.path.exists(times_path):
            with open(times_path, "r") as f:
                times = json.load(f)
            if times:
                return sum(times.values()) / len(times)
    except Exception:
        pass

    return None


def load_population_from_folder(folder: str) -> List[Individual]:
    """
    Load the population directly from an outputs folder, without requiring
    the database for algorithms/codes (PAR2 scores still come from DB or
    local solving_times files).

    The folder should be e.g. "outputs/controlled_mutation" and contain:
      - batch_batch_*/leaders_output.txt        (NL algorithms for leaders)
      - batch_batch_*/member_output_batch_*.txt  (NL algorithms for members)
      - batch_batch_*/code_output_batch_*.txt    (generated C code)
      - batch_batch_*/team_batch_id_map_*.json   (maps code batches -> algorithm IDs)

    This function:
      1. Parses leaders_output.txt + member_output files for NL algorithms
      2. Uses the latest team_batch_id_map JSON to map code batches -> algorithm IDs
      3. Parses code_output files for C code
      4. Looks up PAR2 from DB or local files
      5. Returns Individual objects for every algorithm that has code + PAR2
    """
    logger.info(f"Loading population from folder: {folder}")

    # Locate the batch subdirectory
    batch_dir = None
    if os.path.isdir(folder):
        for entry in sorted(os.listdir(folder)):
            full = os.path.join(folder, entry)
            if entry.startswith("batch_") and os.path.isdir(full):
                batch_dir = full
                break

    if batch_dir is None:
        # Maybe the folder itself is the batch directory
        if os.path.exists(os.path.join(folder, "leaders_output.txt")):
            batch_dir = folder
        else:
            logger.error(f"No batch directory found in {folder}")
            return []

    logger.info(f"Using batch directory: {batch_dir}")

    # ------------------------------------------------------------------
    # 1. Parse algorithms from leaders_output.txt + member_output files
    # ------------------------------------------------------------------
    # algorithm_id -> algorithm_json_string
    algorithms: Dict[str, str] = {}
    # algorithm_id -> parent_id (None for leaders)
    parent_map: Dict[str, Optional[str]] = {}

    # Leaders
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
                except Exception as e:
                    logger.warning(f"Failed to parse leader line: {e}")
    logger.info(f"Parsed {len(algorithms)} leaders from {leaders_path}")

    # Members
    member_files = sorted(glob.glob(os.path.join(batch_dir, "member_output_batch_*.txt")))
    # Determine which leader each member batch belongs to
    # Load the latest team_batch_id_map to get member_batch_map
    map_files = sorted(glob.glob(os.path.join(batch_dir, "team_batch_id_map_*.json")))
    member_batch_to_leader: Dict[str, str] = {}
    if map_files:
        # Use the latest map file (last by sort = latest timestamp)
        with open(map_files[-1], "r") as f:
            batch_map = json.load(f)
        member_batch_to_leader = batch_map.get("member_batch_map", {})

    num_members = 0
    for member_file in member_files:
        # Extract batch_id from filename: member_output_batch_{batch_id}.txt
        fname = os.path.basename(member_file)
        batch_id = fname.replace("member_output_", "").replace(".txt", "")
        leader_id = member_batch_to_leader.get(batch_id)

        with open(member_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    resp = json.loads(line)
                    algo_str, _ = parse_algorithm_response(resp)
                    algo_id = get_id(algo_str)
                    algorithms[algo_id] = algo_str
                    parent_map[algo_id] = leader_id
                    num_members += 1
                except Exception as e:
                    logger.warning(f"Failed to parse member line: {e}")
    logger.info(f"Parsed {num_members} members from {len(member_files)} files")

    # ------------------------------------------------------------------
    # 2. Build code_batch_id -> algorithm_id mapping from team_batch_id_map
    # ------------------------------------------------------------------
    code_batch_to_algo: Dict[str, str] = {}
    if map_files:
        with open(map_files[-1], "r") as f:
            batch_map = json.load(f)
        code_batch_to_algo = batch_map.get("code_batch_map", {})

    # ------------------------------------------------------------------
    # 3. Parse code from code_output_batch_*.txt files
    # ------------------------------------------------------------------
    # algorithm_id -> list of (code_id, code_str)
    algo_codes: Dict[str, List[Tuple[str, str]]] = {}

    code_output_files = sorted(glob.glob(os.path.join(batch_dir, "code_output_batch_*.txt")))
    for code_file in code_output_files:
        fname = os.path.basename(code_file)
        # Extract batch_id: code_output_batch_{batch_id}.txt
        batch_id = fname.replace("code_output_", "").replace(".txt", "")
        algo_id = code_batch_to_algo.get(batch_id)

        if algo_id is None:
            # Try to infer from code_batch_input file with same algo hash
            logger.debug(f"No mapping for code batch {batch_id}, skipping")
            continue

        with open(code_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    resp = json.loads(line)
                    code_str = parse_code_response(resp)
                    code_id = get_id(code_str)
                    if algo_id not in algo_codes:
                        algo_codes[algo_id] = []
                    algo_codes[algo_id].append((code_id, code_str))
                except Exception as e:
                    logger.warning(f"Failed to parse code line: {e}")

    logger.info(f"Parsed codes for {len(algo_codes)} algorithms from {len(code_output_files)} files")

    # ------------------------------------------------------------------
    # 4. Assemble Individuals: match algorithm + best code + PAR2
    # ------------------------------------------------------------------
    population: List[Individual] = []
    skipped_no_code = 0
    skipped_no_par2 = 0

    for algo_id, algo_json in algorithms.items():
        codes = algo_codes.get(algo_id, [])
        if not codes:
            skipped_no_code += 1
            continue

        # Find code with best (lowest) PAR2
        best_code_id = None
        best_code_str = None
        best_par2 = float("inf")

        for code_id, code_str in codes:
            par2 = _load_par2_for_algorithm(algo_id, code_id)
            if par2 is not None and par2 < best_par2:
                best_par2 = par2
                best_code_id = code_id
                best_code_str = code_str

        if best_code_id is None or best_par2 == float("inf"):
            skipped_no_par2 += 1
            continue

        individual = Individual(
            algorithm_id=algo_id,
            algorithm_json=algo_json,
            code_id=best_code_id,
            code=best_code_str,
            par2=best_par2,
            target_function="kissat_restarting",
            parent_id=parent_map.get(algo_id),
        )
        population.append(individual)

    logger.info(
        f"Loaded {len(population)} individuals from folder "
        f"(skipped {skipped_no_code} with no code, {skipped_no_par2} with no PAR2)"
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

### PAR2 Score: {individual.par2:.2f} (lower is better){baseline_section}

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
    # Try to extract JSON from markdown code block
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        # Try to find raw JSON object
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
        response = get_response_from_chatgpt(
            prompt=prompt,
            system_message=CAUSAL_ANALYSIS_SYSTEM_MESSAGE,
            model=model,
            temperature=0.3,  # Lower temperature for analytical consistency
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
# Stage 3: Crossover
# ---------------------------------------------------------------------------

CROSSOVER_SYSTEM_MESSAGE = (
    "You are an expert SAT solver researcher performing genetic crossover to create "
    "improved solver heuristics. You combine the strengths of parent algorithms while "
    "avoiding their weaknesses."
)


def build_crossover_prompt(
    parent_a: Individual,
    parent_b: Individual,
    causal_a: CausalReport,
    causal_b: CausalReport,
) -> str:
    """Build the prompt for crossover of two parents."""
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

The offspring should target the function: kissat_restarting

Output your answer as JSON only:
```json
{{
    "name": "Brief algorithm name (<=6 words)",
    "algorithm": "Complete algorithmic description. Step 1: ... Step 2: ...",
    "reason": "Why this combination improves on both parents.",
    "target_function": "kissat_restarting",
    "parent_a_strengths_used": ["which strengths from Parent A were incorporated"],
    "parent_b_strengths_used": ["which strengths from Parent B were incorporated"]
}}
```"""


def parse_crossover_response(
    response: str, parent_a_id: str, parent_b_id: str
) -> Optional[OffspringResult]:
    """Parse the LLM crossover response into an OffspringResult."""
    # Strip markdown code block if present
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

    # Build algorithm spec in the same format as the original pipeline
    algo_spec = {
        "name": data.get("name", "Crossover Offspring"),
        "algorithm": data.get("algorithm", ""),
        "target_function": data.get("target_function", "kissat_restarting"),
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
    # Sort by PAR2 (ascending = best first)
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
    Stage 2: Perform crossover for each pair of parents.

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
        response = get_response_from_chatgpt(
            prompt=prompt,
            system_message=CROSSOVER_SYSTEM_MESSAGE,
            model=model,
            temperature=0.7,
        )

        offspring = parse_crossover_response(
            response, parent_a.algorithm_id, parent_b.algorithm_id
        )
        if offspring is not None:
            offspring_list.append(offspring)
            logger.info(f"  Offspring: {offspring.algorithm_id[:16]}...")
        else:
            logger.warning(f"  Failed to parse offspring")

    logger.info(f"Crossover produced {len(offspring_list)} offspring")
    return offspring_list


# ---------------------------------------------------------------------------
# Stage 3b: Rubric-Based Offspring Selection
# ---------------------------------------------------------------------------

RUBRIC_SYSTEM_MESSAGE = (
    "You are an expert SAT solver researcher evaluating the quality of a genetically-evolved "
    "solver heuristic. You compare the offspring algorithm against its two parent algorithms "
    "and provide rigorous multi-dimensional scoring."
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
    # Extract JSON from markdown code block
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
            # Clamp to 1-10
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
    Evaluate each offspring using the multi-dimensional rubric via LLM.

    Compares each offspring against its two parents and their causal reports
    to produce a quality score across multiple dimensions.

    Returns list of RubricScore objects.
    """
    logger.info(f"Evaluating {len(offspring_list)} offspring with rubric scoring")

    # Build lookup for quick parent access
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
        response = get_response_from_chatgpt(
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


def select_offspring(
    offspring_list: List[OffspringResult],
    rubric_scores: List[RubricScore],
    par2_scores: Dict[str, float],
    par2_threshold: Optional[float] = None,
    rubric_min: float = 5.0,
    keep_top_n: Optional[int] = None,
) -> List[Tuple[OffspringResult, float, float]]:
    """
    Select the best offspring after evaluation, using PAR2 as a threshold gate
    and rubric score for ranking.

    This runs AFTER code generation and evaluation. Only offspring that built
    successfully (present in par2_scores) are considered.

    Selection logic:
      1. Filter out offspring not in par2_scores (build failed / not evaluated).
      2. If par2_threshold is set, discard offspring whose PAR2 exceeds it.
      3. Discard offspring whose weighted rubric score is below rubric_min.
      4. Rank remaining offspring by weighted rubric score (descending).
      5. If keep_top_n is specified, keep only the top N.

    Args:
        offspring_list: All offspring from crossover.
        rubric_scores: Rubric evaluations (one per offspring).
        par2_scores: Dict mapping offspring algorithm_id -> PAR2 from evaluation.
                     Only offspring present here are considered (build succeeded).
        par2_threshold: Maximum PAR2 to keep (lower is better). Offspring above
                        this are discarded. If None, no PAR2 filtering.
        rubric_min: Minimum weighted rubric score to keep (default 5.0).
        keep_top_n: Keep at most this many offspring (default: all passing).

    Returns:
        List of (OffspringResult, rubric_weighted_total, par2) tuples,
        ranked by rubric score descending.
    """
    # Build lookup: offspring_id -> rubric score
    rubric_lookup: Dict[str, RubricScore] = {s.offspring_id: s for s in rubric_scores}

    # (offspring, rubric_weighted, par2)
    candidates: List[Tuple[OffspringResult, float, float]] = []

    for offspring in offspring_list:
        # Must have PAR2 (i.e. build succeeded and evaluation ran)
        par2 = par2_scores.get(offspring.algorithm_id)
        if par2 is None:
            logger.debug(
                f"No PAR2 for {offspring.algorithm_id[:16]} "
                f"(build failed or not evaluated), skipping"
            )
            continue

        # PAR2 threshold gate
        if par2_threshold is not None and par2 > par2_threshold:
            logger.info(
                f"  Filtered out {offspring.algorithm_id[:16]}: "
                f"PAR2={par2:.2f} > threshold={par2_threshold:.2f}"
            )
            continue

        rscore = rubric_lookup.get(offspring.algorithm_id)
        if rscore is None:
            logger.debug(f"No rubric score for {offspring.algorithm_id[:16]}, skipping")
            continue

        # Rubric minimum gate
        if rscore.weighted_total < rubric_min:
            logger.info(
                f"  Filtered out {offspring.algorithm_id[:16]}: "
                f"rubric={rscore.weighted_total:.2f} < min={rubric_min:.2f}"
            )
            continue

        candidates.append((offspring, rscore.weighted_total, par2))

    # Sort by weighted rubric score descending
    candidates.sort(key=lambda x: x[1], reverse=True)

    if keep_top_n is not None and len(candidates) > keep_top_n:
        candidates = candidates[:keep_top_n]

    logger.info(
        f"Selection: {len(offspring_list)} offspring -> {len(candidates)} selected "
        f"(rubric_min={rubric_min}"
        + (f", par2_threshold={par2_threshold}" if par2_threshold is not None else "")
        + (f", keep_top_n={keep_top_n}" if keep_top_n is not None else "")
        + ")"
    )

    for off, rsc, p2 in candidates:
        logger.info(f"  SELECTED {off.algorithm_id[:16]}: PAR2={p2:.2f}, rubric={rsc:.2f}")

    return candidates


# ---------------------------------------------------------------------------
# Output Helpers — Rubric
# ---------------------------------------------------------------------------

def save_rubric_scores(rubric_scores: List[RubricScore], output_dir: str) -> str:
    """Save rubric scores to JSON file."""
    path = os.path.join(output_dir, "rubric_scores.json")
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


def load_rubric_scores(output_dir: str) -> List[RubricScore]:
    """Load rubric scores from a previously saved JSON file."""
    path = os.path.join(output_dir, "rubric_scores.json")
    if not os.path.exists(path):
        logger.error(f"Rubric scores file not found: {path}")
        return []
    with open(path, "r") as f:
        data = json.load(f)
    scores = []
    for entry in data:
        rs = RubricScore(
            offspring_id=entry["offspring_id"],
            parent_a_id=entry["parent_a_id"],
            parent_b_id=entry["parent_b_id"],
        )
        for dim, val in entry.get("scores", {}).items():
            if hasattr(rs, dim):
                setattr(rs, dim, val)
        rs.weighted_total = entry.get("weighted_total", 0.0)
        rs.justifications = entry.get("justifications", {})
        scores.append(rs)
    logger.info(f"Loaded {len(scores)} rubric scores from {path}")
    return scores


# ---------------------------------------------------------------------------
# Stage 4: Code Generation for Offspring
# ---------------------------------------------------------------------------

def generate_offspring_code(
    offspring_list: List[OffspringResult],
    code_prompt_template_path: str,
    model: str = "gpt-4.1",
) -> Dict[str, str]:
    """
    Stage 3: Generate C code for each offspring algorithm.

    Uses the same coder prompt template as the main pipeline.
    Returns dict mapping algorithm_id -> code string.
    """
    logger.info(f"Generating code for {len(offspring_list)} offspring")
    code_prompt_template = read_code_prompt_template(code_prompt_template_path)
    offspring_codes: Dict[str, str] = {}

    for i, offspring in enumerate(offspring_list):
        logger.info(
            f"[{i+1}/{len(offspring_list)}] Generating code for {offspring.algorithm_id[:16]}..."
        )

        # The coder prompt expects the algorithm text, not JSON
        try:
            algo_spec = json.loads(offspring.algorithm_json)
            algorithm_text = algo_spec.get("algorithm", offspring.algorithm_json)
        except json.JSONDecodeError:
            algorithm_text = offspring.algorithm_json

        prompt = generate_code_prompt(code_prompt_template, algorithm_text)
        response = get_response_from_chatgpt(
            prompt=prompt,
            system_message="You are an expert in writing C code for Kissat-based SAT Solvers.",
            model=model,
            temperature=0.5,
        )

        # Parse code from response (reuse existing parser logic)
        code = parse_code_response({"response": {"output_text": response}})
        offspring_codes[offspring.algorithm_id] = code
        logger.info(f"  Generated code ({len(code)} chars)")

    return offspring_codes


# ---------------------------------------------------------------------------
# Stage 5: Store and Evaluate
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

        # Store algorithm
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
            target_function="kissat_restarting",
            parent_id=offspring.parent_a_id,  # Track lineage
        )
        update_algorithm_result(algo_result)
        update_router_table(CHATGPT_DATA_GENERATION_TABLE, offspring.algorithm_id, output_tag)

        # Store code
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


def evaluate_offspring(
    stored_pairs: List[Tuple[str, str]],
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Evaluate offspring by building solvers and submitting SLURM jobs.

    Uses the existing EvaluationPipeline. Build happens locally first;
    if build fails the code is buggy and excluded immediately.

    Returns:
        (successful_pairs, failed_pairs) where each is a list of
        (algorithm_id, code_id) tuples.
    """
    from llmsat.pipelines.evaluation import EvaluationPipeline

    logger.info(f"Evaluating {len(stored_pairs)} offspring")
    pipeline = EvaluationPipeline()

    successful: List[Tuple[str, str]] = []
    failed: List[Tuple[str, str]] = []

    for i, (algorithm_id, code_id) in enumerate(stored_pairs):
        logger.info(f"[{i+1}/{len(stored_pairs)}] Evaluating code {code_id[:16]}...")
        try:
            pipeline.run_single_solver(code_id)
            # Check if build succeeded by reading the code result back
            code_result = get_code_result(code_id)
            if code_result is not None and code_result.status == CodeStatus.BuildFailed:
                logger.warning(f"  Build FAILED for {code_id[:16]} (buggy code)")
                failed.append((algorithm_id, code_id))
            else:
                successful.append((algorithm_id, code_id))
                logger.info(f"  Build OK, SLURM jobs submitted for {code_id[:16]}")
        except Exception as e:
            logger.error(f"  Evaluation failed: {e}")
            failed.append((algorithm_id, code_id))

    logger.info(
        f"Evaluation: {len(successful)} built successfully, "
        f"{len(failed)} failed (buggy)"
    )
    return successful, failed


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


def save_crossover_results(
    offspring_list: List[OffspringResult], output_dir: str
) -> str:
    """Save crossover results to JSON file."""
    path = os.path.join(output_dir, "crossover_results.json")
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
        })
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved crossover results to {path}")
    return path


def save_offspring_codes(codes: Dict[str, str], output_dir: str) -> str:
    """Save offspring codes to JSON file."""
    path = os.path.join(output_dir, "offspring_codes.json")
    with open(path, "w") as f:
        json.dump(codes, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved offspring codes to {path}")
    return path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_evolution(
    generation_tag: str,
    code_prompt_path: str,
    output_tag: Optional[str] = None,
    folder: Optional[str] = None,
    top_k: Optional[int] = None,
    max_pairs: Optional[int] = None,
    model: str = "gpt-4.1",
    baseline_par2: Optional[float] = None,
    evaluate: bool = False,
    causal_only: bool = False,
    skip_causal: bool = False,
    skip_rubric: bool = False,
    rubric_min: float = 5.0,
    rubric_keep_top_n: Optional[int] = None,
    par2_threshold: Optional[float] = None,
    rubric_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Run the full genetic evolution pipeline.

    Pipeline stages:
      1. Load population (from DB or folder)
      2. Causal analysis (LLM analyzes each individual)
      3. Crossover (LLM combines all pairs using causal reports)
      4. Code generation (LLM generates C code for ALL offspring)
      5. Store in DB
      6. Evaluate (build + SLURM) — filters out buggy builds
      7. Rubric selection (LLM scores buildable offspring on multi-dimensional
         rubric, then PAR2 threshold gates + rubric score ranks and filters)

    Args:
        generation_tag: Source population generation tag
        code_prompt_path: Path to the coder prompt template
        output_tag: Tag for offspring generation (default: {generation_tag}_gen1)
        folder: Path to outputs folder to load population from files instead of DB.
                 e.g. "outputs/controlled_mutation" or
                 "outputs/controlled_mutation/batch_batch_6983d461f9dc8190b4544560a1a35c01"
        top_k: Number of top individuals for crossover selection
        max_pairs: Maximum number of crossover pairs
        model: OpenAI model for LLM calls
        baseline_par2: Baseline PAR2 for comparison in causal analysis
        evaluate: Whether to build and evaluate offspring via SLURM
        causal_only: Only run causal analysis, skip crossover
        skip_causal: Skip causal analysis, load from file
        skip_rubric: Skip rubric scoring (use all offspring without selection)
        rubric_min: Minimum weighted rubric score to keep offspring (default 5.0)
        rubric_keep_top_n: Keep at most N offspring after rubric ranking (default: all passing)
        par2_threshold: PAR2 hard gate — offspring above this are discarded (used post-eval)
        rubric_weights: Custom weights for rubric dimensions (default: balanced with emphasis
                        on improvement and soundness)

    Returns:
        Summary dict with results from each stage.
    """
    if output_tag is None:
        output_tag = f"{generation_tag}_gen1"

    output_dir = get_generation_output_dir(output_tag)
    logger.info(f"Evolution pipeline: {generation_tag} -> {output_tag}")
    logger.info(f"Output directory: {output_dir}")

    summary: Dict[str, Any] = {
        "generation_tag": generation_tag,
        "output_tag": output_tag,
        "timestamp": datetime.now().isoformat(),
    }

    # Stage 1: Load population
    if folder:
        logger.info(f"Loading population from folder: {folder}")
        population = load_population_from_folder(folder)
    else:
        population = load_population(generation_tag)
    if not population:
        logger.error("No individuals loaded. Check generation tag and database.")
        return {"error": "Empty population"}

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

    # Stage 2: Causal analysis
    if skip_causal:
        logger.info("Skipping causal analysis, loading from file")
        causal_reports = load_causal_reports(output_dir)
        if not causal_reports:
            logger.error("No causal reports found to load. Run without --skip_causal first.")
            return {"error": "No causal reports"}
    else:
        causal_reports = generate_causal_reports(population, model=model, baseline_par2=baseline_par2)
        save_causal_reports(causal_reports, output_dir)

    summary["causal_reports_count"] = len(causal_reports)

    if causal_only:
        logger.info("Causal-only mode: stopping after causal analysis")
        summary["mode"] = "causal_only"
        # Save summary
        summary_path = os.path.join(output_dir, "evolution_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    # Stage 3: Crossover (all pairs)
    pairs = select_pairs(population, top_k=top_k, max_pairs=max_pairs)
    offspring_list = perform_crossover(pairs, causal_reports, model=model)
    save_crossover_results(offspring_list, output_dir)

    summary["crossover"] = {
        "pairs_attempted": len(pairs),
        "offspring_produced": len(offspring_list),
    }

    if not offspring_list:
        logger.warning("No offspring produced from crossover")
        return summary

    # Stage 4: Code generation (all offspring)
    offspring_codes = generate_offspring_code(offspring_list, code_prompt_path, model=model)
    save_offspring_codes(offspring_codes, output_dir)

    summary["code_generation"] = {
        "codes_generated": len(offspring_codes),
    }

    # Stage 5: Store in DB
    stored_pairs = store_offspring(offspring_list, offspring_codes, output_tag)
    summary["stored"] = len(stored_pairs)

    # Stage 6: Evaluate — build + SLURM (filters out buggy code)
    if evaluate and stored_pairs:
        logger.info("Starting evaluation of all offspring (build + SLURM)")
        successful_pairs, failed_pairs = evaluate_offspring(stored_pairs)

        summary["evaluation"] = {
            "build_success": len(successful_pairs),
            "build_failed": len(failed_pairs),
            "failed_ids": [cid for _, cid in failed_pairs],
        }

        # Collect PAR2 scores for successfully built offspring
        # NOTE: PAR2 is available only after SLURM jobs complete.
        # At this point, build success is known but PAR2 may not be.
        # We read whatever PAR2 is available from the DB.
        par2_scores: Dict[str, float] = {}
        for algo_id, code_id in successful_pairs:
            code_result = get_code_result(code_id)
            if code_result is not None and code_result.par2 is not None:
                par2_scores[algo_id] = code_result.par2

        # Build set of algorithms that built successfully
        successful_algo_ids = {algo_id for algo_id, _ in successful_pairs}
        # Filter offspring_list to only those that built successfully
        buildable_offspring = [
            off for off in offspring_list
            if off.algorithm_id in successful_algo_ids
        ]

        logger.info(
            f"After build: {len(buildable_offspring)} buildable offspring, "
            f"{len(par2_scores)} with PAR2 available"
        )

        # Stage 7: Rubric scoring + selection (on buildable offspring only)
        if not skip_rubric and buildable_offspring:
            logger.info("Running rubric-based evaluation on buildable offspring")
            rubric_scores = evaluate_offspring_rubric(
                buildable_offspring,
                population,
                causal_reports,
                model=model,
                weights=rubric_weights,
            )
            save_rubric_scores(rubric_scores, output_dir)

            selected = select_offspring(
                buildable_offspring,
                rubric_scores,
                par2_scores=par2_scores,
                par2_threshold=par2_threshold,
                rubric_min=rubric_min,
                keep_top_n=rubric_keep_top_n,
            )

            summary["rubric_selection"] = {
                "buildable": len(buildable_offspring),
                "scored": len(rubric_scores),
                "selected": len(selected),
                "rubric_min": rubric_min,
                "keep_top_n": rubric_keep_top_n,
                "par2_threshold": par2_threshold,
                "avg_weighted_score": (
                    sum(s.weighted_total for s in rubric_scores) / len(rubric_scores)
                    if rubric_scores else 0.0
                ),
                "dimension_averages": {
                    dim: (
                        sum(getattr(s, dim) for s in rubric_scores) / len(rubric_scores)
                        if rubric_scores else 0.0
                    )
                    for dim in RUBRIC_DIMENSIONS
                },
                "selected_offspring": [
                    {
                        "algorithm_id": off.algorithm_id,
                        "rubric_score": rsc,
                        "par2": p2,
                    }
                    for off, rsc, p2 in selected
                ],
            }
        else:
            if skip_rubric:
                logger.info("Skipping rubric scoring (--skip_rubric)")
            summary["rubric_selection"] = "skipped"

    else:
        summary["evaluation"] = "skipped"
        summary["rubric_selection"] = "skipped (no evaluation)"
        if not evaluate:
            logger.info(
                f"Evaluation skipped. To evaluate later, run:\n"
                f"  python src/llmsat/pipelines/evaluation.py "
                f"--run_all --generation_tag {output_tag}"
            )

    # Save summary
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
            "Source population generation tag (e.g., controlled_mutation). "
            "Required when loading from DB. When --folder is used, this is "
            "optional and auto-derived from the folder name if omitted."
        ),
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help=(
            "Path to outputs folder to load population from files instead of DB. "
            "e.g. 'outputs/controlled_mutation' or "
            "'outputs/controlled_mutation/batch_batch_xxx'. "
            "PAR2 scores are still loaded from the DB or local solving_times files."
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
        help="Number of top individuals to use for crossover (default: all)",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="Maximum number of crossover pairs (default: all combinations)",
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
        help="Only run causal analysis (Stage 1), skip crossover",
    )
    parser.add_argument(
        "--skip_causal",
        action="store_true",
        default=False,
        help="Skip causal analysis and load from existing file",
    )
    parser.add_argument(
        "--skip_rubric",
        action="store_true",
        default=False,
        help="Skip rubric scoring — use all offspring without LLM-based selection",
    )
    parser.add_argument(
        "--rubric_min",
        type=float,
        default=5.0,
        help=(
            "Minimum weighted rubric score to keep an offspring (default: 5.0). "
            "Offspring below this are discarded before code generation."
        ),
    )
    parser.add_argument(
        "--rubric_keep_top_n",
        type=int,
        default=None,
        help="Keep at most N offspring after rubric ranking (default: all passing)",
    )
    parser.add_argument(
        "--par2_threshold",
        type=float,
        default=None,
        help=(
            "PAR2 hard gate — offspring with PAR2 above this value are discarded. "
            "Only effective if PAR2 scores are available (e.g., after evaluation)."
        ),
    )

    args = parser.parse_args()

    # Resolve generation_tag: required for DB mode, auto-derived for folder mode
    if args.generation_tag is None:
        if args.folder:
            # Derive from folder path: "outputs/controlled_mutation/batch_..." -> "controlled_mutation"
            folder_norm = args.folder.rstrip("/\\")
            parts = folder_norm.replace("\\", "/").split("/")
            # Walk backwards to find the first part that isn't a batch_ dir
            for part in reversed(parts):
                if not part.startswith("batch_") and part != "outputs" and part:
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
        skip_rubric=args.skip_rubric,
        rubric_min=args.rubric_min,
        rubric_keep_top_n=args.rubric_keep_top_n,
        par2_threshold=args.par2_threshold,
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
        print(f"PAR2: best={par2.get('best', 'N/A'):.2f}, mean={par2.get('mean', 'N/A'):.2f}")

    print(f"Causal reports: {summary.get('causal_reports_count', 0)}")

    crossover = summary.get("crossover", {})
    if crossover:
        print(f"Crossover: {crossover.get('pairs_attempted', 0)} pairs -> {crossover.get('offspring_produced', 0)} offspring")

    print(f"Code generated: {summary.get('code_generation', {}).get('codes_generated', 0)}")
    print(f"Stored: {summary.get('stored', 0)}")

    evaluation = summary.get("evaluation", {})
    if isinstance(evaluation, dict):
        print(f"Evaluation: {evaluation.get('build_success', 0)} built OK, "
              f"{evaluation.get('build_failed', 0)} failed (buggy)")
    else:
        print(f"Evaluation: {evaluation}")

    rubric = summary.get("rubric_selection", {})
    if isinstance(rubric, dict):
        print(f"Rubric selection: {rubric.get('buildable', 0)} buildable -> "
              f"{rubric.get('scored', 0)} scored -> {rubric.get('selected', 0)} selected "
              f"(min={rubric.get('rubric_min', 'N/A')}, avg_score={rubric.get('avg_weighted_score', 0):.2f})")
        dim_avgs = rubric.get("dimension_averages", {})
        if dim_avgs:
            dims_str = ", ".join(f"{d.replace('_', ' ').title()}={v:.1f}" for d, v in dim_avgs.items())
            print(f"  Dimension averages: {dims_str}")
        selected_list = rubric.get("selected_offspring", [])
        if selected_list:
            print("  Top selected:")
            for entry in selected_list[:5]:
                print(f"    {entry['algorithm_id'][:16]}... "
                      f"PAR2={entry['par2']:.2f}, rubric={entry['rubric_score']:.2f}")
    elif isinstance(rubric, str):
        print(f"Rubric selection: {rubric}")

    print("=" * 60)


if __name__ == "__main__":
    main()
