"""Tests for recording an exact OpenEvolve comparison configuration."""

import unittest

from render_config import render


class TestRenderConfig(unittest.TestCase):
    def test_resolves_model_and_reasoning_effort(self) -> None:
        template = (
            "name: ${COMPARISON_MODEL}\n"
            "effort: ${COMPARISON_REASONING_EFFORT}\n"
            "parallel: ${COMPARISON_MAX_CANDIDATE_JOBS}\n"
        )
        self.assertEqual(
            render(
                template,
                model="gpt-test",
                reasoning_effort="high",
                parallel_evaluations=8,
            ),
            "name: gpt-test\neffort: high\nparallel: 8\n",
        )

    def test_requires_both_placeholders(self) -> None:
        with self.assertRaisesRegex(ValueError, "COMPARISON_REASONING_EFFORT"):
            render(
                "name: ${COMPARISON_MODEL}\n",
                model="gpt-test",
                reasoning_effort="high",
                parallel_evaluations=4,
            )


if __name__ == "__main__":
    unittest.main()
