#!/usr/bin/env python3
"""Unit tests for the Shinka result-contract adapter."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).with_name("evaluate.py")
SPEC = importlib.util.spec_from_file_location("shinka_sat_evaluate", MODULE_PATH)
assert SPEC and SPEC.loader
EVALUATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATE)


class ShinkaEvaluatorAdapterTests(unittest.TestCase):
    def test_success_writes_shinka_contract(self) -> None:
        raw = {"combined_score": 1.75, "par2": 571.425}
        artifacts = {
            "stage": "evaluation",
            "instances": 2,
            "solved": 2,
            "timeouts": 0,
            "errors": 0,
            "missing": 0,
            "candidate_hash": "abc",
            "protocol_hash": "protocol",
            "job_ids": ["123"],
            "cache_format_version": 2,
        }
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            EVALUATE, "evaluate_runtime", return_value=(raw, artifacts)
        ):
            self.assertTrue(EVALUATE.evaluate("candidate.c", tmp))
            metrics = json.loads((Path(tmp) / "metrics.json").read_text())
            correct = json.loads((Path(tmp) / "correct.json").read_text())

        self.assertEqual(metrics["combined_score"], 1.75)
        self.assertEqual(metrics["public"]["instances"], 2)
        self.assertEqual(metrics["private"]["proof_validation"], "not_run")
        self.assertEqual(correct, {"correct": True, "error": ""})

    def test_compile_failure_is_incorrect(self) -> None:
        raw = {"combined_score": 0.0, "par2": 2400.0}
        artifacts = {"stage": "compile", "build_output": "compiler failed"}
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            EVALUATE, "evaluate_runtime", return_value=(raw, artifacts)
        ):
            self.assertFalse(EVALUATE.evaluate("candidate.c", tmp))
            correct = json.loads((Path(tmp) / "correct.json").read_text())

        self.assertFalse(correct["correct"])
        self.assertIn("compile", correct["error"])

    def test_adapter_exception_uses_configured_par2_penalty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            EVALUATE, "evaluate_runtime", side_effect=RuntimeError("broken")
        ), patch.object(EVALUATE, "PAR2_PENALTY", 2400.0):
            self.assertFalse(EVALUATE.evaluate("candidate.c", tmp))
            metrics = json.loads((Path(tmp) / "metrics.json").read_text())

        self.assertEqual(metrics["public"]["par2"], 2400.0)


if __name__ == "__main__":
    unittest.main()
