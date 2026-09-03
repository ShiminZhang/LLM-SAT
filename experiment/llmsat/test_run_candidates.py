"""Smoke tests for the LLM-SAT comparison budget planner."""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "experiment" / "llmsat" / "run_candidates.sh"


class TestBudgetPlan(unittest.TestCase):
    def run_plan(self, budget: int) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.update(
            SAT_REPO_ROOT=str(REPO_ROOT),
            LLMSAT_CANDIDATE_BUDGET=str(budget),
            LLMSAT_PLAN_ONLY="1",
        )
        return subprocess.run(
            ["bash", str(RUNNER), "decide"],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )

    def test_100_candidate_plan(self) -> None:
        result = self.run_plan(100)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.strip(),
            "budget=100 initial_candidates=30 iterations=5",
        )

    def test_500_candidate_plan(self) -> None:
        result = self.run_plan(500)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.strip(),
            "budget=500 initial_candidates=30 iterations=32",
        )

    def test_rejects_unrepresentable_budget(self) -> None:
        result = self.run_plan(101)
        self.assertEqual(result.returncode, 2)
        self.assertIn("must be divisible by 5", result.stderr)


if __name__ == "__main__":
    unittest.main()
