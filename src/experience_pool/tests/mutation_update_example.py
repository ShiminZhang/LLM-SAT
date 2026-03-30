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
    
    # Run the update with the requested target directory and debug=True
    input_dir = "/local-scratch1/jla1045/LLM-SAT/solvers/mutation_curated"
    
    print(f"\nCalling manager.update() on {input_dir} with debug=True...\n")
    
    summary = manager.update(
        pool_name="mutation",
        input_dir=input_dir,
        top_k_good=5,
        top_k_bad=5,
        debug=True  # This parameter triggers the printing of internal states
    )
    
    print("\n========== UPDATE PIPELINE COMPLETE ==========")
    print("Summary of operation:")
    # Filter out errors list if empty for cleaner printing
    if not summary.get("errors"):
        summary.pop("errors", None)
        
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
