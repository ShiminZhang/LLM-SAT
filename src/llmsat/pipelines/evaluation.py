from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from llmsat.utils.aws import get_algorithm_result
from llmsat.llmsat import CodeResult, AlgorithmResult, BASE_SOLVER_PATH, get_logger
from llmsat.evaluation.coder import Coder
from ..evaluation.algorithm_evaluator import AlgorithmEvaluator
from ..evaluation.code_evaluator import CodeEvaluator

logger = get_logger(__name__)


@dataclass
class SolverEvaluationResult:
    solver_path: str
    solver_id: str
    build_success: bool
    par2_score: float
    other_metrics: Dict[str, float]

    def get_reward(self):
        # compute the final scalar / vector reward
        pass

def parse_solving_time(content: str) -> float:
    # parse the solving time from the content
    for line in content.split("\n"):
        if "solve time" in line:
            return float(line.split(" ")[-1])
    return None

@dataclass
class EvaluationPipeline:
    """Unified evaluation entry point for designer and coder models."""
    def __init__(self):
        pass

    def clean_solving_logs(self, algorithm_id: str, code_id: str) -> None:
        # clean the solving logs
        solver_dir = get_solver_dir(algorithm_id, code_id)
        for file in os.listdir(solver_dir):
            if file.endswith(".solving.log"):
                os.remove(f"{solver_dir}/{file}")
            if file.endswith(".slurm.log"):
                os.remove(f"{solver_dir}/{file}")

    def clean_solver(self, algorithm_id: str, code_id: str) -> None:
        # clean the solver
        solver_dir = get_solver_dir(algorithm_id, code_id)
        shutil.rmtree(solver_dir)

    def collect_results(self, algorithm_id: str, code_id: str) -> None:
        # collect the results from the solver
        solver_dir = get_solver_dir(algorithm_id, code_id)
        for file in os.listdir(solver_dir):
            solving_time = {}
            if file.endswith(".solving.log"):
                instance_name = file.split(".")[0]
                with open(f"{solver_dir}/{file}", "r") as f:
                    content = f.read()
                solving_time = parse_solving_time(content)
                solving_time[instance_name] = solving_time
            par2 = average(solving_time.values())

        # update the code result and algorithm result
        code_result = get_code_result(code_id)
        code_result.par2 = par2
        code_result.solving_time = str(solving_time)
        update_code_result(code_result)
        algorithm_result = get_algorithm_result(algorithm_id)
        algorithm_result.par2 = par2
        update_algorithm_result(algorithm_result)
        with open(get_solver_solving_times_path(algorithm_id, code_id), "w") as f:
            json.dump(solving_time, f)
        return par2

    def slurm_colloct_result(self, slurm_ids: List[str], code_id: str) -> None:
        activate_python_path = get_activate_python_path()
        pass

    def build_solver(self, code_result: CodeResult) -> None:
        code = code_result.code
        # copy original solver to a new folder
        new_solver_path = get_solver_dir(code_result.algorithm_id, code_result.id)
        shutil.copytree(BASE_SOLVER_PATH, new_solver_path)
        # replace the code in the new solver
        restart_file = f"{new_solver_path}/src/restart.c"
        # First read the file to find where to insert the code
        with open(restart_file, "r") as f:
            lines = f.readlines()
        
        # Find the insertion point (after "//LLMSAT start")
        insert_idx = None
        for i, line in enumerate(lines):
            if line.startswith("//LLMSAT start"):
                insert_idx = i + 1  # Insert after this line
                break
        
        if insert_idx is None:
            raise ValueError("Could not find '//LLMSAT start' marker in restart.c")
        
        # Write the modified content
        with open(restart_file, "w") as f:
            # Write lines before insertion point
            f.writelines(lines[:insert_idx])
            # Write the new code
            f.write(code)
            f.write("\n")
            # Write the remaining lines
            f.writelines(lines[insert_idx:])

        # try compile the solver
        try:
            output = os.popen(f"cd {new_solver_path} && make").read()
            if "error" in output:
                build_success = False
            else:
                build_success = True
        except Exception as e:
            build_success = False
        if build_success:
            new_solver_bin_path = f"{new_solver_path}/build/kissat"
            os.makedirs(f"solvers/algorithm_{code_result.algorithm_id}/code_{code_result.id}", exist_ok=True)
            os.copy(new_solver_bin_path, f"solvers/algorithm_{code_result.algorithm_id}/code_{code_result.id}/kissat")
            return new_solver_path
        else:
            return None
        return 

    def slurm_run_evaluate(self, solver_path: str, benchmark_path: str) -> None:
        # run the solver on the benchmark
        activate_python_path = get_activate_python_path()
        slurm_ids = []
        for benchmark_file in os.listdir(benchmark_path):
            if benchmark_file.endswith(".cnf"):
                command = f"{activate_python_path} && ./{solver_path}/kissat {benchmark_path}/{benchmark_file} > {solver_path}/{benchmark_file}.solving.log"
                slurm_log = f"{solver_path}/{benchmark_file}.slurm.log"
                slurm_cmd = wrap_command_to_slurm(command, output_file=slurm_log, job_name=f"solve_{benchmark_file}")
                
                slurm_id = os.popen(slurm_cmd).read().strip() # TODO test if this is correct, also , there might be a limit on the number of jobs that can be submitted at once
                logger.info(f"Submitted job {slurm_id} for {benchmark_file}")
                slurm_ids.append(slurm_id)
        return slurm_ids

    def run_single_solver(self, code_id: str) -> None:  # pragma: no cover - declaration only
        """Run evaluation for configured components."""
        code_result = get_code_result(code_id)
        assert code_result is not None, "Code result not found"
        solver_path = self.build_solver(code_result) # build in evaluation
        if solver_path is not None: # build successful
            print(f"Solver built successfully: {solver_path}")
            slurm_ids = slurm_run_evaluate(solver_path, SAT2025_BENCHMARK_PATH)
            self.slurm_colloct_result(slurm_ids,code_id)
        else: # build failed
            print(f"Solver build failed: {build_result.error}")
    
    def read_current_progress(self) -> None:
        # read the current progress from the progress file
        pass

    def read_algorithm(self, algorithm_id: str) -> None:
        algorithm_result = get_algorithm_result(algorithm_id)
        assert algorithm_result is not None, "Algorithm result not found"
        return algorithm_result

    def run_all_solvers(self, algorithm_id: str) -> None:  # pragma: no cover - declaration only
        """Run evaluation for all configured components."""
        algorithm = self.read_algorithm(algorithm_id)
        os.makedirs(f"solvers/algorithm_{algorithm_id}", exist_ok=True)
        code_id_list = self.generate_or_read_code(algorithm) # actually should only read here, generation should be done in a separate process

        assert code is not None, "Code generation failed"
        
        for code_id in code_id_list:
            self.run_single_solver(code_id)
    
    def generate_or_read_code(self, algorithm: AlgorithmResult) -> None: #
        # generate the code for the solver
        for code_id in algorithm_result.code_id_list:
            code_result = get_code_result(code_id)
            if code_result is not None:
                return code_result
        

def main():
    pass

if __name__ == "__main__":
    main()