import sys
from pathlib import Path
import json

# Ensure the root of the project is in the Python path
src_dir = str(Path(__file__).parent.parent.parent.resolve())
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from experience_pool.manager import ExperiencePoolManager


def main():
    print("Initializing ExperiencePoolManager...")
    manager = ExperiencePoolManager()

    input_dir = "/local-scratch1/jla1045/LLM-SAT/solvers/algorithm_curated"
    baseline_par2 = 400
    baseline_code = "some dummy code"

    print(
        f"\nCalling manager.update() for algorithm pool with debug=True...\n"
        f"  - input_dir: {input_dir}\n"
        f"  - baseline_par2: {baseline_par2}\n"
        f"  - baseline_code: {baseline_code}\n"
    )

    summary = manager.update(
        pool_name="algorithm",
        input_dir=input_dir,
        baseline_par2=baseline_par2,
        baseline_code=baseline_code,
        debug=True,
    )

    print("\n========== UPDATE PIPELINE COMPLETE ==========")
    print("Summary of operation:")

    # Filter out errors list if empty for cleaner printing
    if not summary.get("errors"):
        summary.pop("errors", None)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
