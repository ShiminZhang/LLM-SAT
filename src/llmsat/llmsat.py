import json
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import hashlib
NOT_INITIALIZED = "NOT_INITIALIZED"
BASE_SOLVER_PATH = "/home/meru/scratch/LLM-SAT/solvers/base"
RECOVERED_ALGORITHM = "recovered_algorithm"
SAT2025_BENCHMARK_PATH = "data/benchmarks/satcomp2025"
PYENV_PATH = "../../general/bin/activate"
CHATGPT_DATA_GENERATION_TABLE = "chatgpt_datagen"

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
            json.dump(data, f, indent=4)

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
