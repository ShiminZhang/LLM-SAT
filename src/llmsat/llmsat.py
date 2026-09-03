import json
import logging
import os
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import hashlib
NOT_INITIALIZED = "NOT_INITIALIZED"
from llmsat.config import BASE_SOLVER_PATH, PYTHON_ACTIVATE_PATH, BASELINE_PAR2
PYENV_PATH = PYTHON_ACTIVATE_PATH  # backwards-compat alias
RECOVERED_ALGORITHM = "recovered_algorithm"
# Kept under the historical name for compatibility with existing imports.
# Experiment suites can pin another complete formula family without rewriting
# evaluation code; the default remains the full cryptography-ascon family.
SAT2025_BENCHMARK_PATH = os.environ.get(
    "LLMSAT_BENCHMARK_DIR",
    "data/benchmarks/formula-families/cryptography-ascon",
)
CHATGPT_DATA_GENERATION_TABLE = "chatgpt_datagen"


def normalize_par2(raw_par2_score: List[float], baseline_par2: float) -> List[float]:
    """Normalize PAR2 scores by dividing by baseline.

    Args:
        raw_par2_score: [easy, hard, sat, unsat, all] raw scores
        baseline_par2: Baseline solver's overall PAR2 on this cluster

    Returns:
        [easy, hard, sat, unsat, all] normalized scores (lower is better, 1.0 = baseline)
    """
    if baseline_par2 <= 0:
        raise ValueError("baseline_par2 must be positive")
    return [score / baseline_par2 if score is not None else None
            for score in raw_par2_score]

#DATATYPES
ALGORITHM = "algorithm"
CODE = "code"

class CodeStatus:
    Pending = "pending" # generating
    Generated = "generated"
    BuildFailed = "build_failed"
    Evaluating = "evaluating"
    Evaluated = "evaluated"

class AlgorithmStatus:
    Generated = "generated"
    CodeGenerated = "code_generated"
    Evaluating = "evaluating"
    Evaluated = "evaluated"

@dataclass
class TaskResult:
    task_id: str
    status: str
    created_at: str

class Role(Enum):
    LEADER = "leader"
    MEMBER = "member"

@dataclass
class AlgorithmResult:
    # Identity
    id: str
    function_name: str  # name of the function this algorithm modifies (e.g. "kissat_restarting")
    description: str  # "AlgorithmName: Step 1: ... Step 2: ..." (human-readable algorithm text)
    role: Role  # whether this algorithm is a team leader or member

    # Pipeline state
    status: str
    last_updated: str
    code_id_list: List[str]  # list of code ids generated for this algorithm

    # Lineage
    parent_id: Optional[List[str]] = None  # list of parent algorithm IDs (None for leaders, [leader_id] for members, [parent_a, parent_b] for combinations)
    parent_algorithm_description: Optional[List[str]] = None  # descriptions of parent algorithms, indexed same as parent_id

    # Scores — each is [easy, hard, sat, unsat, all] or None
    raw_par2_score: Optional[List[float]] = None
    normalized_par2_score: Optional[List[float]] = None

    # LLM reasoning about why this algorithm is good
    analysis: Optional[str] = None

    # For mutants only: which step of the leader algorithm this mutant was derived from
    mutation_step: Optional[str] = None

    # Metadata
    prompt: str = ""
    other_metrics: Dict[str, Any] = field(default_factory=dict)

    def save_to_json(self, directory: Union[str, Path]) -> None:
        """Save this AlgorithmResult as a JSON file named {self.id}.json in the given directory."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / f"{self.id}.json"
        data = asdict(self)
        data["role"] = self.role.value
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, default=str)

@dataclass
class CodeResult:
    id: str
    algorithm_id: str
    code: str
    status: str
    par2: Optional[float]
    # tag: str
    # coder: str
    last_updated: str
    build_success: bool = NOT_INITIALIZED

def get_id(input_str: str) -> str:

    return hashlib.sha256(input_str.encode()).hexdigest()

ALGORITHMS_CODE_MAP_PATH = "data/algorithms_code_map.json"
CODE_GENERATION_PROGRESS_PATH = "data/code_generation_progress.json"


def setup_logging(level: int = logging.INFO, format_string: Optional[str] = None) -> None:
    """
    Configure logging for the entire package.
    
    Call this once at application startup to set up logging configuration.
    
    Args:
        level: Logging level (default: INFO)
        format_string: Custom format string. If None, uses a default format.
    """
    # Try to use Rich for colored, pretty logs in terminal. Fallback to standard logging.
    try:
        from rich.logging import RichHandler  # type: ignore
        handler: logging.Handler = RichHandler(rich_tracebacks=True, markup=True)
        # With RichHandler it's recommended to keep format minimal and let handler render details
        fmt = "%(message)s" if format_string is None else format_string
        logging.basicConfig(
            level=level,
            format=fmt,
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[handler],
        )
    except Exception:
        if format_string is None:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(
            level=level,
            format=format_string,
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance. Use this instead of logging.getLogger() for consistency.
    
    Usage:
        from llmsat.llmsat import get_logger
        logger = get_logger(__name__)
    
    Args:
        name: Logger name, typically __name__ of the calling module.
    
    Returns:
        Logger instance configured with the package's logging settings.
    """
    return logging.getLogger(name)
