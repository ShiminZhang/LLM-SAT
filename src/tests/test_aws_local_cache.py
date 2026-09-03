"""Regression tests for the Rorqual local result cache."""

from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from llmsat.llmsat import AlgorithmResult, AlgorithmStatus, Role
from llmsat.utils import aws


class TestLocalCache(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.cache_root = Path(self.temporary.name)
        self.patches = [
            patch.object(aws, "USE_LOCAL_CACHE", True),
            patch.object(aws, "LOCAL_CACHE_ROOT", self.cache_root),
        ]
        for active_patch in self.patches:
            active_patch.start()

    def tearDown(self) -> None:
        for active_patch in reversed(self.patches):
            active_patch.stop()
        self.temporary.cleanup()

    def test_algorithm_round_trip_and_idempotent_code_append(self) -> None:
        algorithm = AlgorithmResult(
            id="algorithm-1",
            function_name="kissat_decide_phase",
            description="test",
            role=Role.LEADER,
            status=AlgorithmStatus.Generated,
            last_updated="now",
            code_id_list=[],
        )
        aws.update_algorithm_result(algorithm)
        aws.append_code_id(algorithm.id, "code-1")
        aws.append_code_id(algorithm.id, "code-1")

        restored = aws.get_algorithm_result(algorithm.id)
        self.assertIsNotNone(restored)
        self.assertEqual(restored.role, Role.LEADER)
        self.assertEqual(restored.code_id_list, ["code-1"])

    def test_concurrent_router_updates_do_not_lose_ids(self) -> None:
        count = 64
        with ThreadPoolExecutor(max_workers=16) as executor:
            list(
                executor.map(
                    lambda index: aws.update_router_table(
                        "comparison", f"candidate-{index}", "iteration-1"
                    ),
                    range(count),
                )
            )

        ids = aws.get_ids_from_router_table("comparison", "iteration-1")
        self.assertEqual(len(ids), count)
        self.assertEqual(set(ids), {f"candidate-{index}" for index in range(count)})


if __name__ == "__main__":
    unittest.main()
