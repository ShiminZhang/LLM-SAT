import os
from typing import List, Optional, Union


def get_role_dir(parent_id: Union[None, str, List[str]]) -> str:
    """Return 'leaders' if parent_id is None, else 'members'."""
    return "leaders" if parent_id is None else "members"


def get_solver_dir(algorithm_id: str, code_id: str,
                   generation_tag: str = None, parent_id: str = None) -> str:
    if generation_tag is None:
        return f"solvers/algorithm_{algorithm_id}/code_{code_id}/"
    role = get_role_dir(parent_id)
    return f"solvers/{generation_tag}/{role}/algorithm_{algorithm_id}/code_{code_id}/"


def get_algorithm_dir(algorithm_id: str,
                      generation_tag: str = None, parent_id: str = None) -> str:
    if generation_tag is None:
        return f"solvers/algorithm_{algorithm_id}/"
    role = get_role_dir(parent_id)
    return f"solvers/{generation_tag}/{role}/algorithm_{algorithm_id}/"


def get_batch_output_dir(generation_tag: str, batch_id: str) -> str:
    if not os.path.exists(f"outputs/{generation_tag}/batch_{batch_id}/"):
        os.makedirs(f"outputs/{generation_tag}/batch_{batch_id}/")
    return f"outputs/{generation_tag}/batch_{batch_id}/"


def get_generation_output_dir(generation_tag: str) -> str:
    if not os.path.exists(f"outputs/{generation_tag}/"):
        os.makedirs(f"outputs/{generation_tag}/")
    return f"outputs/{generation_tag}/"


def get_solver_result_dir(algorithm_id: str, code_id: str,
                          generation_tag: str = None, parent_id: str = None) -> str:
    if generation_tag is None:
        path = f"solvers/algorithm_{algorithm_id}/result/code_{code_id}/"
    else:
        role = get_role_dir(parent_id)
        path = f"solvers/{generation_tag}/{role}/algorithm_{algorithm_id}/result/code_{code_id}/"
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_solver_solving_times_path(algorithm_id: str, code_id: str,
                                  generation_tag: str = None, parent_id: str = None) -> str:
    if generation_tag is None:
        return f"solvers/algorithm_{algorithm_id}/solving_times_{code_id}.json"
    role = get_role_dir(parent_id)
    return f"solvers/{generation_tag}/{role}/algorithm_{algorithm_id}/solving_times_{code_id}.json"


def get_solver_proof_dir(
    algorithm_id: str,
    code_id: str,
    generation_tag: str = None,
    parent_id: str = None,
    create_dir: bool = True,
) -> str:
    """Return directory for benchmark proof files of one generated solver."""
    base_result_dir = get_solver_result_dir(
        algorithm_id,
        code_id,
        generation_tag=generation_tag,
        parent_id=parent_id,
    )
    path = os.path.join(base_result_dir, "proofs")
    if create_dir and not os.path.exists(path):
        os.makedirs(path)
    return path


def get_solver_proof_path(
    algorithm_id: str,
    code_id: str,
    benchmark_file: str,
    generation_tag: str = None,
    parent_id: str = None,
    create_dir: bool = True,
) -> str:
    """Return proof file path for one (solver, benchmark CNF) run."""
    proof_dir = get_solver_proof_dir(
        algorithm_id,
        code_id,
        generation_tag=generation_tag,
        parent_id=parent_id,
        create_dir=create_dir,
    )
    return os.path.join(proof_dir, f"{benchmark_file}.proof")
