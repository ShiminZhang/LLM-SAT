import logging
from dataclasses import dataclass
from typing import List, Dict
import hashlib
NOT_INITIALIZED = "NOT_INITIALIZED"
BASE_SOLVER_PATH = "solvers/base"
RECOVERED_ALGORITHM = "recovered_algorithm"
SAT2025_BENCHMARK_PATH = "data/benchmarks/satcomp2025"
PYENV_PATH = "~/.venvs/llmsat312/bin/activate"
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

@dataclass
class AlgorithmResult:
    id: str
    algorithm: str
    status: str
    last_updated: str
    # tag: str
    # designer: str
    prompt: str # not implemented yet
    par2: list[float]
    error_rate: float
    code_id_list: List[str] # list of code ids that have been generated for this algorithm
    other_metrics: Dict[str, float]

@dataclass
class CodeResult:
    id: str
    algorithm_id: str
    code: str
    status: str
    par2: float
    # tag: str
    # coder: str
    last_updated: str
    build_success: bool = NOT_INITIALIZED

def get_id(input_str: str) -> str:

    return hashlib.sha256(input_str.encode()).hexdigest()

ALGORITHMS_CODE_MAP_PATH = "data/algorithms_code_map.json"
CODE_GENERATION_PROGRESS_PATH = "data/code_generation_progress.json"


def setup_logging(level: int = logging.INFO, format_string: str | None = None) -> None:
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