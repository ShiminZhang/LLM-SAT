"""Regression tests for the OpenEvolve runtime evaluator."""

import tempfile
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

# The parser tests do not need OpenEvolve's numerical dependencies. Provide the
# one result type imported by evaluator.py so the tests also run on login nodes.
evaluation_result_module = types.ModuleType("openevolve.evaluation_result")


class EvaluationResult:
    def __init__(self, metrics, artifacts):
        self.metrics = metrics
        self.artifacts = artifacts


evaluation_result_module.EvaluationResult = EvaluationResult
openevolve_module = types.ModuleType("openevolve")
openevolve_module.evaluation_result = evaluation_result_module
sys.modules["openevolve"] = openevolve_module
sys.modules["openevolve.evaluation_result"] = evaluation_result_module

import evaluator


class TestRuntimeParsing(unittest.TestCase):
    def test_supported_target_specs(self):
        decide_pattern, decide_source = evaluator._target_spec("kissat_decide_phase")
        restart_pattern, restart_source = evaluator._target_spec("kissat_restarting")

        self.assertTrue(
            decide_pattern.search("int kissat_decide_phase (kissat *solver, unsigned idx) {")
        )
        self.assertEqual(decide_source, Path("src/decide.c"))
        self.assertTrue(
            restart_pattern.search("bool kissat_restarting (kissat *solver) {")
        )
        self.assertEqual(restart_source, Path("src/restart.c"))

    def test_rejects_unknown_target(self):
        with self.assertRaisesRegex(ValueError, "Unsupported OE_TARGET_FUNCTION"):
            evaluator._target_spec("not_a_kissat_target")

    def test_purged_slurm_job_is_not_active(self):
        completed = subprocess.CompletedProcess(
            args=["squeue"],
            returncode=1,
            stdout="slurm_load_jobs error: Invalid job id specified\n",
        )
        with patch.object(evaluator, "_run", return_value=completed):
            self.assertFalse(evaluator._is_job_active(12345))

    def test_recognizes_only_transient_submission_failures(self):
        self.assertTrue(
            evaluator._retryable_submission_failure(
                "sbatch: error: AssocMaxSubmitJobLimit"
            )
        )
        self.assertTrue(
            evaluator._retryable_submission_failure(
                "Batch job submission failed: Job violates accounting/QOS policy"
            )
        )
        self.assertFalse(
            evaluator._retryable_submission_failure(
                "sbatch: error: Invalid account or account/partition combination"
            )
        )

    def test_submit_candidate_job_retries_transient_qos_limit(self):
        rejected = subprocess.CompletedProcess(
            args=["sbatch"],
            returncode=1,
            stdout="sbatch: error: AssocMaxSubmitJobLimit\n",
        )
        accepted = subprocess.CompletedProcess(
            args=["sbatch"], returncode=0, stdout="12345;cluster\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            script = directory / "candidate.sh"
            script.write_text("#!/bin/bash\n")
            with patch.object(evaluator, "_run", side_effect=[rejected, accepted]), patch.object(
                evaluator, "SUBMIT_RETRY_ATTEMPTS", 2
            ), patch.object(evaluator, "SUBMIT_RETRY_INTERVAL", 1), patch.object(
                evaluator.time, "sleep"
            ) as sleep:
                job_id = evaluator._submit_candidate_job(
                    directory, script, ["one.cnf"], attempt=1
                )

        self.assertEqual(job_id, 12345)
        sleep.assert_called_once_with(1)

    def test_submit_candidate_job_adds_configured_node_constraint(self):
        accepted = subprocess.CompletedProcess(
            args=["sbatch"], returncode=0, stdout="12345;cluster\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            script = directory / "candidate.sh"
            script.write_text("#!/bin/bash\n")
            with patch.object(
                evaluator, "_run", return_value=accepted
            ) as run, patch.object(evaluator, "SLURM_CONSTRAINT", "genoa"):
                evaluator._submit_candidate_job(
                    directory, script, ["one.cnf"], attempt=1
                )

        command = run.call_args.args[0]
        self.assertIn("--constraint=genoa", command)
        self.assertEqual(command[-1], str(script))

    def test_parses_current_and_legacy_cpu_times(self):
        self.assertEqual(evaluator._parse_cpu_time("OE_CPU_TIME=12.500000\n"), (12.5, False))
        self.assertEqual(
            evaluator._parse_cpu_time("OE_CPU_TIME=0.0000002728.530000\n"),
            (2728.53, True),
        )
        self.assertEqual(evaluator._parse_cpu_time("OE_CPU_TIME=broken\n"), (None, False))

    def test_runtime_summary_recovers_old_logs(self):
        with tempfile.TemporaryDirectory() as temporary:
            candidate_dir = Path(temporary)
            results = candidate_dir / "results"
            results.mkdir()
            (results / "normal.cnf.solving.log").write_text(
                "OE_STATUS=SOLVED\nOE_CPU_TIME=12.500000\n"
            )
            (results / "legacy.cnf.solving.log").write_text(
                "OE_STATUS=SOLVED\nOE_CPU_TIME=0.0000007.250000\n"
            )
            (results / "timeout.cnf.solving.log").write_text("OE_STATUS=TIMEOUT\n")
            (results / "error.cnf.solving.log").write_text("OE_STATUS=ERROR\n")

            par2, details = evaluator._parse_runtime(
                candidate_dir,
                ["normal.cnf", "legacy.cnf", "timeout.cnf", "error.cnf"],
            )

        self.assertAlmostEqual(par2, 1204.9375)
        self.assertEqual(details["solved"], 2)
        self.assertEqual(details["timeouts"], 1)
        self.assertEqual(details["errors"], 1)
        self.assertEqual(details["legacy_times_recovered"], 1)

    def test_candidate_script_uses_eight_core_worker_pool(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            solver_dir = directory / "solver"
            solver_dir.mkdir()
            script = evaluator._write_candidate_script(directory, solver_dir)
            text = script.read_text()
            checked = subprocess.run(["bash", "-n", str(script)], check=False)

        self.assertEqual(checked.returncode, 0)
        self.assertIn("EVAL_CORES=8", text)
        self.assertIn("TIMEOUT=1200", text)
        self.assertIn("wait -n", text)
        self.assertNotIn("SLURM_ARRAY_TASK_ID", text)


if __name__ == "__main__":
    unittest.main()
