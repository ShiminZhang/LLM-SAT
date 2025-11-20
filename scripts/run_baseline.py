import argparse
import os
import re
solver = "./solvers/kissat/build/kissat"
benchmark = "./data/benchmarks/satcomp2025/"

def parse_solving_time(file_path: str):
    lines = open(file_path, "r").readlines()
    for line in reversed(lines): 
        if "process-time" in line:
            match = re.search(r'(\d+\.?\d*)\s+seconds', line)
            if match:
                time = float(match.group(1))
                return time
    
    # logger.warning(f"Failed to parse solving time from")
    # print(file_path)
    return 5000

def run_baseline():
    for file in os.listdir(benchmark):
        if file.endswith(".cnf"):
            name = file.split(".")[0]
            output_file = f"./data/results/baseline/{name}.out"
            cmd = f"{solver} {benchmark}/{file} > {output_file}"
            sbatch_cmd = f"sbatch --time=00:00:5000 --mem=8G --cpus-per-task=1 --job-name=baseline_{name} --output={output_file}.log --error={output_file}.log --wrap=\"{cmd}\""
            os.system(sbatch_cmd)
        # exit()
# 
def collect_results():
    solving_times = {}
    for file in os.listdir(benchmark):
        if file.endswith(".cnf"):
            name = file.split(".")[0]
            output_file = f"./data/results/baseline/{name}.out"
            time = parse_solving_time(output_file)
            solving_times[name] = time
    average_time = sum(solving_times.values()) / len(solving_times)
    print(f"Average time: {average_time}")
    return solving_times

if __name__ == "__main__":
    # run_baseline()
    collect_results()