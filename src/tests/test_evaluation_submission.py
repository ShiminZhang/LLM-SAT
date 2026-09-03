"""Regression tests for resumable Slurm candidate evaluation."""

from __future__ import annotations

import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import llmsat.pipelines.evaluation as evaluation


def _bare_pipeline(tag: str = "test_iter0") -> evaluation.EvaluationPipeline:
    pipeline = object.__new__(evaluation.EvaluationPipeline)
    pipeline.generation_tag = tag
    pipeline.timeout = 5
    pipeline.wall_time = "00:01:00"
    return pipeline


class TestEvaluationSubmission(unittest.TestCase):
    def test_wait_for_submit_capacity(self) -> None:
        pipeline = _bare_pipeline()
        counts = iter([1000, 998])
        sleeps = []
        with patch.object(
            pipeline, "_current_slurm_task_count", side_effect=lambda: next(counts)
        ), patch.object(
            evaluation.time, "sleep", side_effect=lambda seconds: sleeps.append(seconds)
        ):
            pipeline._wait_for_submit_capacity(2)

        self.assertEqual(sleeps, [evaluation.SLURM_SUBMIT_POLL_INTERVAL])

    def test_resume_only_retries_missing_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original_cwd = Path.cwd()
            os.chdir(root)
            self.addCleanup(os.chdir, original_cwd)

            solver_path = root / "solver"
            result_dir = root / "results"
            solver_path.mkdir()
            result_dir.mkdir()
            pipeline = _bare_pipeline()
            submitted_counts = []
            next_job_id = iter([101, 102])

            def fake_submit(command: str, needed_slots: int) -> int:
                submitted_counts.append(needed_slots)
                script_path = Path(command.split()[-1])
                script = script_path.read_text()
                match = re.search(r'^TASK_LIST="([^"]+)"', script, re.MULTILINE)
                self.assertIsNotNone(match)
                for line in Path(match.group(1)).read_text().splitlines():
                    _, output_dir, cnf_list = line.split("\t")
                    for cnf_file in Path(cnf_list).read_text().splitlines():
                        Path(output_dir, f"{cnf_file}.solving.log").write_text(
                            "s SATISFIABLE\nc process-time: 0.01 seconds\n"
                        )
                return next(next_job_id)

            patches = [
                patch.object(evaluation, "SLURM_MAX_ARRAY_SIZE", 2),
                patch.object(evaluation, "SLURM_SUBMIT_LIMIT", 2),
                patch.object(
                    evaluation,
                    "get_generation_output_dir",
                    side_effect=lambda tag: str(root / "outputs" / tag),
                ),
                patch.object(pipeline, "_submit_with_capacity", side_effect=fake_submit),
                patch.object(pipeline, "_wait_for_job_completion", return_value=None),
            ]
            for active_patch in patches:
                active_patch.start()
                self.addCleanup(active_patch.stop)

            solver_tasks = [(str(solver_path), str(result_dir), "code-1")]
            cnf_files = ["a.cnf", "b.cnf", "c.cnf"]
            self.assertEqual(
                pipeline.slurm_run_evaluate_batch(
                    solver_tasks, "/benchmarks", cnf_files
                ),
                [101],
            )
            self.assertEqual(submitted_counts, [1])

            # Simulate one lost result after an interrupted evaluation and resume.
            os.remove(result_dir / "b.cnf.solving.log")
            with patch.object(pipeline, "_job_is_active", return_value=False):
                self.assertEqual(
                    pipeline.slurm_run_evaluate_batch(
                        solver_tasks, "/benchmarks", cnf_files
                    ),
                    [101, 102],
                )
            self.assertEqual(submitted_counts, [1, 1])

            state = json.loads(
                (
                    root
                    / "outputs"
                    / "test_iter0"
                    / "evaluation_submission_state.json"
                ).read_text()
            )
            self.assertEqual(
                [batch["status"] for batch in state["batches"]],
                ["completed"],
            )
            self.assertEqual(state["batches"][0]["attempts"][-1]["task_count"], 1)
            legacy_state = json.loads(
                (root / "outputs" / "test_iter0" / "submitted_job_ids.json").read_text()
            )
            self.assertEqual(legacy_state["job_ids"], [])
            self.assertEqual(legacy_state["completed_job_ids"], [101, 102])


if __name__ == "__main__":
    unittest.main()
