"""Small, dependency-free tests for evaluation records."""

import unittest

from llmsat.llmsat import (
    AlgorithmResult,
    AlgorithmStatus,
    CodeResult,
    CodeStatus,
    Role,
)


class TestEvaluationRecords(unittest.TestCase):
    def test_current_algorithm_and_code_schema(self) -> None:
        algorithm = AlgorithmResult(
            id="algorithm-1",
            function_name="kissat_restarting",
            description="Return whether the restart preconditions are met.",
            role=Role.LEADER,
            status=AlgorithmStatus.Generated,
            last_updated="2026-09-03T00:00:00",
            code_id_list=[],
        )
        code = CodeResult(
            id="code-1",
            algorithm_id=algorithm.id,
            code="return false;",
            status=CodeStatus.Generated,
            par2=None,
            last_updated="2026-09-03T00:00:00",
            build_success=False,
        )

        self.assertEqual(algorithm.function_name, "kissat_restarting")
        self.assertEqual(algorithm.role, Role.LEADER)
        self.assertEqual(code.algorithm_id, algorithm.id)


if __name__ == "__main__":
    unittest.main()
