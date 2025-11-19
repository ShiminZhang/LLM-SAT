import os
def get_solver_dir(algorithm_id: str, code_id: str) -> str:
    return f"solvers/algorithm_{algorithm_id}/code_{code_id}/"

def get_algorithm_dir(algorithm_id: str) -> str:
    return f"solvers/algorithm_{algorithm_id}/"

def get_solver_result_dir(algorithm_id: str, code_id: str) -> str:
    if not os.path.exists(f"solvers/algorithm_{algorithm_id}/result/code_{code_id}/"):
        os.makedirs(f"solvers/algorithm_{algorithm_id}/result/code_{code_id}/")
    return f"solvers/algorithm_{algorithm_id}/result/code_{code_id}/"

def get_solver_solving_times_path(algorithm_id: str, code_id: str) -> str:
    return f"solvers/algorithm_{algorithm_id}/code_{code_id}/solving_times.json"