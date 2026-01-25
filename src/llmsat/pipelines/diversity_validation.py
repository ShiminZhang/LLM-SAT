"""
Validation script for diversity analysis judge.

Creates synthetic test cases with known expected diversity levels
to verify the judge is discriminating properly.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

from llmsat.pipelines.diversity_analysis import (
    build_leader_diversity_prompt,
    create_diversity_batch_input,
    parse_diversity_response,
    get_diversity_report_dir,
    JUDGE_MODEL,
)
from llmsat.utils.chatgpt_helper import (
    submit_batch_input as helper_submit_batch_input,
    block_until_completion as helper_block_until_completion,
    download_batch_outputs as helper_download_batch_outputs,
)
from llmsat.llmsat import setup_logging, get_logger

import logging
setup_logging(level=logging.INFO)
logger = get_logger(__name__)


# =============================================================================
# LOW DIVERSITY TEST CASE
# Same core algorithm with trivial variations (expect score ~1-20)
# =============================================================================

LOW_DIVERSITY_ALGORITHMS = [
    ("low_1", """### Exponential Backoff Restart

The algorithm restarts when conflicts exceed a threshold that grows exponentially.

1. Initialize restart_threshold = 100
2. Initialize multiplier = 1.5
3. On each conflict:
   - If conflicts >= restart_threshold:
     - Trigger restart
     - restart_threshold = restart_threshold * multiplier
4. Return whether to restart"""),

    ("low_2", """### Exponential Backoff Restart

The algorithm restarts when conflicts exceed a threshold that grows exponentially.

1. Initialize restart_threshold = 100
2. Initialize growth_factor = 1.5
3. On each conflict:
   - If conflicts >= restart_threshold:
     - Trigger restart
     - restart_threshold = restart_threshold * growth_factor
4. Return whether to restart"""),

    ("low_3", """### Exponential Backoff Restart Strategy

The algorithm restarts when conflicts exceed a threshold that grows exponentially.

1. Set initial restart_threshold = 100
2. Set multiplier = 1.5
3. On each conflict:
   - If conflicts >= restart_threshold:
     - Trigger restart
     - restart_threshold = restart_threshold * multiplier
4. Return restart decision"""),

    ("low_4", """### Exponential Backoff Restart

This algorithm restarts when conflicts exceed a threshold that grows exponentially.

1. Initialize restart_threshold to 100
2. Initialize multiplier to 1.5
3. On each conflict:
   - If conflicts >= restart_threshold:
     - Trigger restart
     - restart_threshold = restart_threshold * multiplier
4. Return whether to restart"""),

    ("low_5", """### Exponential Backoff Restart

The algorithm restarts when the conflict count exceeds a threshold that grows exponentially.

1. Initialize restart_threshold = 100
2. Initialize multiplier = 1.5
3. On each conflict:
   - If conflicts >= restart_threshold:
     - Trigger restart
     - restart_threshold = restart_threshold * multiplier
4. Return whether to restart"""),
]


# =============================================================================
# HIGH DIVERSITY TEST CASE
# Fundamentally different approaches (expect score ~80-100)
# =============================================================================

HIGH_DIVERSITY_ALGORITHMS = [
    ("high_1", """### Glucose-Style LBD Restart

Restarts based on Literal Block Distance (LBD) of learned clauses.

1. Track rolling average of LBD values for recent learned clauses
2. Maintain long-term LBD average
3. If recent_avg_lbd > long_term_avg_lbd * 0.8:
   - Problem is getting harder, trigger restart
4. Use Luby sequence to gate restart frequency
5. Return restart decision"""),

    ("high_2", """### Random Walk Restart

Uses randomization to escape local minima.

1. Generate random number r in [0, 1)
2. Compute restart_probability = base_prob * (1 + conflicts/10000)
3. If r < restart_probability:
   - Trigger restart
4. Cap probability at 0.01 to avoid excessive restarts
5. Return restart decision"""),

    ("high_3", """### Conflict Clause Quality Restart

Monitors the utility of learned clauses to decide restarts.

1. Track how often recent learned clauses participate in propagation
2. Compute clause_utility = propagations_from_recent / total_propagations
3. If clause_utility < 0.1:
   - Recent learning is not useful, trigger restart
4. Reset counters after restart
5. Return restart decision"""),

    ("high_4", """### Variable Activity Stagnation Restart

Restarts when variable selection becomes stagnant.

1. Track top-10 most active variables
2. Compare current top-10 with previous checkpoint
3. Compute overlap = |current ∩ previous| / 10
4. If overlap > 0.9 for 1000 conflicts:
   - Search is stuck in same region, trigger restart
5. Return restart decision"""),

    ("high_5", """### Phase Transition Detector Restart

Detects phase transitions in search behavior.

1. Monitor decision_rate = decisions / conflicts over sliding window
2. Compute rate_variance over recent windows
3. If rate_variance > threshold:
   - Search behavior is unstable, likely at phase transition
   - Trigger restart to re-stabilize
4. Adapt threshold based on problem size
5. Return restart decision"""),
]


# =============================================================================
# MEDIUM DIVERSITY TEST CASE (optional calibration)
# Same family but meaningfully different variations (expect score ~40-60)
# =============================================================================

MEDIUM_DIVERSITY_ALGORITHMS = [
    ("med_1", """### Exponential Backoff Restart

Restarts when conflicts exceed exponentially growing threshold.

1. Initialize threshold = 100, multiplier = 1.5
2. If conflicts >= threshold: restart, threshold *= multiplier
3. Return decision"""),

    ("med_2", """### Geometric Backoff Restart

Restarts when conflicts exceed geometrically growing threshold.

1. Initialize threshold = 100, base = 2
2. If conflicts >= threshold: restart, threshold = base^(restart_count)
3. Return decision"""),

    ("med_3", """### Luby Sequence Restart

Restarts following the Luby sequence pattern.

1. Compute luby_value for current restart index
2. threshold = luby_value * unit_size
3. If conflicts >= threshold: restart, increment index
4. Return decision"""),

    ("med_4", """### Linear Backoff Restart

Restarts when conflicts exceed linearly growing threshold.

1. Initialize threshold = 100, increment = 50
2. If conflicts >= threshold: restart, threshold += increment
3. Return decision"""),

    ("med_5", """### Arithmetic Backoff Restart

Restarts with arithmetically increasing intervals.

1. Initialize threshold = 100, step = 100
2. If conflicts >= threshold: restart, threshold += step, step += 10
3. Return decision"""),
]


def run_validation(output_dir: str = "outputs/diversity_validation/"):
    """Run validation tests for the diversity judge."""

    os.makedirs(output_dir, exist_ok=True)

    # Build prompts for all three test cases
    prompts = []

    # Low diversity
    prompts.append(("test_low_diversity", build_leader_diversity_prompt(LOW_DIVERSITY_ALGORITHMS)))

    # High diversity
    prompts.append(("test_high_diversity", build_leader_diversity_prompt(HIGH_DIVERSITY_ALGORITHMS)))

    # Medium diversity
    prompts.append(("test_medium_diversity", build_leader_diversity_prompt(MEDIUM_DIVERSITY_ALGORITHMS)))

    # Create batch input
    batch_input_path = os.path.join(output_dir, "validation_batch_input.jsonl")
    create_diversity_batch_input(prompts, batch_input_path)

    logger.info("Submitting validation batch...")
    batch_id = helper_submit_batch_input(batch_input_path)
    logger.info(f"Batch submitted: {batch_id}")

    logger.info("Waiting for completion...")
    status = helper_block_until_completion(batch_id)
    logger.info(f"Batch completed: {status}")

    if status != "completed":
        logger.error(f"Batch failed: {status}")
        return

    # Download results
    batch_output_path = os.path.join(output_dir, "validation_batch_output.jsonl")
    helper_download_batch_outputs(batch_id, Path(batch_output_path))

    # Parse results
    results = {}
    with open(batch_output_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            response = json.loads(line)
            custom_id = response.get("custom_id", "unknown")
            score, explanation = parse_diversity_response(response)
            results[custom_id] = {"score": score, "explanation": explanation}

    # Print validation report
    print("\n" + "=" * 70)
    print("DIVERSITY JUDGE VALIDATION REPORT")
    print("=" * 70)

    expected = {
        "test_low_diversity": ("~1-20", "Same algorithm with trivial rewording"),
        "test_high_diversity": ("~80-100", "Fundamentally different approaches"),
        "test_medium_diversity": ("~40-60", "Same family, meaningful variations"),
    }

    all_passed = True
    for test_id, (expected_range, description) in expected.items():
        result = results.get(test_id, {})
        score = result.get("score", "N/A")
        explanation = result.get("explanation", "N/A")

        print(f"\n--- {test_id} ---")
        print(f"Description: {description}")
        print(f"Expected: {expected_range}")
        print(f"Actual: {score}")
        print(f"Explanation: {explanation[:200]}..." if len(str(explanation)) > 200 else f"Explanation: {explanation}")

        # Check if in expected range
        if score is not None:
            if "low" in test_id and score > 30:
                print("⚠️  WARNING: Score higher than expected for low diversity!")
                all_passed = False
            elif "high" in test_id and score < 70:
                print("⚠️  WARNING: Score lower than expected for high diversity!")
                all_passed = False
            elif "medium" in test_id and (score < 30 or score > 70):
                print("⚠️  WARNING: Score outside expected range for medium diversity!")
                all_passed = False
            else:
                print("✓ Score in expected range")

    print("\n" + "=" * 70)
    if all_passed:
        print("VALIDATION PASSED: Judge is discriminating properly")
    else:
        print("VALIDATION CONCERNS: Judge may need prompt tuning")
    print("=" * 70)

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "validation_passed": all_passed,
    }
    with open(os.path.join(output_dir, "validation_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    return results


if __name__ == "__main__":
    run_validation()
