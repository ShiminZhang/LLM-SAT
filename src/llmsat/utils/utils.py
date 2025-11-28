import fcntl
import os
import tempfile
from pathlib import Path
from llmsat.llmsat import PYENV_PATH


def atomic_write(file_path: str, content: str) -> None:
    """
    Atomically write content to a file using fcntl locking and atomic rename.
    """
    file_path_obj = Path(file_path)
    file_dir = file_path_obj.parent
    file_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary file in the same directory to ensure atomic rename
    fd, temp_path = tempfile.mkstemp(dir=file_dir, prefix=file_path_obj.name + ".tmp.", suffix="")
    try:
        with os.fdopen(fd, "w") as f:
            # Acquire exclusive lock on temp file
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(content)
            f.flush()
            os.fsync(f.fileno())  # Ensure data is written to disk
            # Lock is released when file is closed
        
        # Atomic rename: this is atomic on the same filesystem
        os.rename(temp_path, file_path)
    except Exception:
        # Clean up temp file on error
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def atomic_read(file_path: str) -> str:
    """
    Atomically read content from a file using fcntl locking.
    """
    with open(file_path, "r") as f:
        # Acquire shared lock for reading
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        try:
            content = f.read()
        finally:
            # Release lock
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return content


def atomic_append(file_path: str, content: str) -> None:
    """
    Atomically append content to a file using fcntl locking.
    """
    file_path_obj = Path(file_path)
    file_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, "a") as f:
        # Acquire exclusive lock for appending
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())  # Ensure data is written to disk
        finally:
            # Release lock
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def wrap_command_to_slurm(
    command: str,
    mem: str="8G",
    time: str="00:00:5000",
    nodes: int=1,
    ntasks: int=1,
    cpus_per_task: int=1,
    job_name: str=None,
    output_file: str=None,
    error_file: str=None,
    dependencies: list[str]=None,
    dependency_type: str="afterok",
) -> str:
    dependencies_parameter = ""
    if dependencies is not None and len(dependencies) > 0:
        dependencies_parameter = f"--dependency={dependency_type}:{':'.join(dependencies)}"
    job_name_parameter = ""
    if job_name is not None:
        job_name_parameter = f"--job-name={job_name}"
    output_parameter = ""
    if output_file is not None:
        output_parameter = f"--output={output_file}"
    error_parameter = ""
    if error_file is not None:
        error_parameter = f"--error={error_file}"
    return f"sbatch \
        {job_name_parameter} \
        {output_parameter} \
        {error_parameter} \
        --mem={mem} \
        --time={time} \
        --nodes={nodes} \
        --ntasks={ntasks} \
        --cpus-per-task={cpus_per_task} \
        {dependencies_parameter} \
        --wrap='{command}'"

def get_activate_python_path() -> str:
    return f"source {PYENV_PATH}"