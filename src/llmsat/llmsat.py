import logging
from dataclasses import dataclass

NOT_INITIALIZED = -1
BASE_SOLVER_PATH = "solvers/base"
SAT2025_BENCHMARK_PATH = "benchmarks/SAT2025"
PYENV_PATH = "~/.pyenv/versions/3.10.10/bin/python"

class TaskStatus:
    Completed = "completed"
    Failed = "failed"
    InProgress = "in_progress"
    Pending = "pending"
    Cancelled = "cancelled"
    Queued = "queued"
    Running = "running"
    Stopped = "stopped"
    Suspended = "suspended"


@dataclass
class TaskResult:
    task_id: str
    status: str
    created_at: str

@dataclass
class AlgorithmResult:
    algorithm: str
    prompt: str
    par2: float
    error_rate: float
    code_id_list: List[str] # list of code ids that have been generated for this algorithm
    other_metrics: Dict[str, float]

@dataclass
class CodeResult(TaskResult):
    algorithm_id: str
    code: str
    solver_id: str
    build_success: bool

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
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S"
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