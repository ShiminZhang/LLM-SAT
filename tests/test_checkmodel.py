"""Tests for tools/checkmodel: SAT model verification against DIMACS CNFs.

Run with:  python -m pytest tests/test_checkmodel.py -v

The binary is (re)built via make in a session fixture.  Behavior contract:
  exit 0 + "MODEL VERIFIED"  -> the printed model satisfies every clause
  exit 1 + "MODEL FAILED..." -> missing/incomplete/contradictory/falsifying model
  exit 2 + "ERROR..."        -> usage, I/O, or CNF parse errors
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKMODEL_DIR = REPO_ROOT / "tools" / "checkmodel"
CHECKMODEL_BIN = CHECKMODEL_DIR / "checkmodel"

# (1 v -2) and (2 v 3): satisfied by e.g. 1, -2, 3.
TINY_CNF = "c tiny example\np cnf 3 2\n1 -2 0\n2 3 0\n"


@pytest.fixture(scope="session")
def checkmodel(tmp_path_factory) -> str:
    proc = subprocess.run(
        ["make", "-C", str(CHECKMODEL_DIR)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"make failed:\n{proc.stdout}\n{proc.stderr}"
    assert CHECKMODEL_BIN.is_file(), "checkmodel binary missing after make"
    return str(CHECKMODEL_BIN)


def run(checkmodel: str, cnf: Path, log: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [checkmodel, str(cnf), str(log)],
        capture_output=True,
        text=True,
        timeout=120,
    )


def write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content)
    return path


def test_valid_model(checkmodel, tmp_path):
    cnf = write(tmp_path, "f.cnf", TINY_CNF)
    log = write(
        tmp_path,
        "f.log",
        "c kissat banner\ns SATISFIABLE\nv 1 -2 3 0\nc stats follow\n",
    )
    proc = run(checkmodel, cnf, log)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "MODEL VERIFIED" in proc.stdout


def test_valid_model_multiline_values(checkmodel, tmp_path):
    cnf = write(tmp_path, "f.cnf", TINY_CNF)
    log = write(
        tmp_path,
        "f.log",
        "s SATISFIABLE\nv 1\nv -2\nv 3\nv 0\n",
    )
    proc = run(checkmodel, cnf, log)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "MODEL VERIFIED" in proc.stdout


def test_invalid_model_reports_first_failing_clause(checkmodel, tmp_path):
    cnf = write(tmp_path, "f.cnf", TINY_CNF)
    # -3 falsifies clause 2 (2 v 3) because 2 is also false.
    log = write(tmp_path, "f.log", "s SATISFIABLE\nv 1 -2 -3 0\n")
    proc = run(checkmodel, cnf, log)
    assert proc.returncode == 1
    assert "MODEL FAILED" in proc.stdout
    assert "clause 2" in proc.stdout


def test_contradictory_model(checkmodel, tmp_path):
    cnf = write(tmp_path, "f.cnf", TINY_CNF)
    log = write(tmp_path, "f.log", "s SATISFIABLE\nv 1 -1 0\n")
    proc = run(checkmodel, cnf, log)
    assert proc.returncode == 1
    assert "MODEL FAILED" in proc.stdout
    assert "contradictory" in proc.stdout


def test_incomplete_v_lines(checkmodel, tmp_path):
    cnf = write(tmp_path, "f.cnf", TINY_CNF)
    # No terminating 0: truncated witness (e.g. solver killed mid-print).
    log = write(tmp_path, "f.log", "s SATISFIABLE\nv 1 -2 3\n")
    proc = run(checkmodel, cnf, log)
    assert proc.returncode == 1
    assert "MODEL FAILED" in proc.stdout


def test_sat_status_without_v_lines(checkmodel, tmp_path):
    cnf = write(tmp_path, "f.cnf", TINY_CNF)
    log = write(tmp_path, "f.log", "c something\ns SATISFIABLE\nc no witness\n")
    proc = run(checkmodel, cnf, log)
    assert proc.returncode == 1
    assert "MODEL FAILED" in proc.stdout


def test_unsat_log_is_not_a_model(checkmodel, tmp_path):
    cnf = write(tmp_path, "f.cnf", TINY_CNF)
    log = write(tmp_path, "f.log", "c solved\ns UNSATISFIABLE\n")
    proc = run(checkmodel, cnf, log)
    assert proc.returncode == 1
    assert "MODEL FAILED" in proc.stdout
    assert "UNSATISFIABLE" in proc.stdout


def test_log_without_status_line(checkmodel, tmp_path):
    cnf = write(tmp_path, "f.cnf", TINY_CNF)
    log = write(tmp_path, "f.log", "c only stats here\nc nothing else\n")
    proc = run(checkmodel, cnf, log)
    assert proc.returncode == 1
    assert "MODEL FAILED" in proc.stdout


def test_huge_padding_comment_lines(checkmodel, tmp_path):
    # Streaming must survive multi-MB single lines and many stat lines
    # without slurping the file (kissat logs are mostly comments).
    pad_line = "c " + "x" * (4 * 1024 * 1024) + "\n"
    stats = "".join(f"c stat line {i} with numbers 123 456\n" for i in range(50000))
    log = write(
        tmp_path,
        "f.log",
        pad_line + stats + "s SATISFIABLE\n" + pad_line + "v 1 -2 3 0\n" + stats,
    )
    cnf = write(tmp_path, "f.cnf", "c " + "y" * (1024 * 1024) + "\n" + TINY_CNF)
    proc = run(checkmodel, cnf, log)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "MODEL VERIFIED" in proc.stdout


def test_header_mismatch_tolerated(checkmodel, tmp_path):
    # Header declares wrong counts; the actual clauses are what matters.
    cnf = write(tmp_path, "f.cnf", "p cnf 10 99\n1 -2 0\n2 3 0\n")
    log = write(tmp_path, "f.log", "s SATISFIABLE\nv 1 -2 3 0\n")
    proc = run(checkmodel, cnf, log)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "MODEL VERIFIED" in proc.stdout


def test_model_with_extra_variables(checkmodel, tmp_path):
    # Assigning variables the CNF never mentions must not fail verification.
    cnf = write(tmp_path, "f.cnf", TINY_CNF)
    log = write(tmp_path, "f.log", "s SATISFIABLE\nv 1 -2 3 4 -5 6 0\n")
    proc = run(checkmodel, cnf, log)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "MODEL VERIFIED" in proc.stdout


def test_unassigned_variable_is_fine_when_clause_satisfied_otherwise(
    checkmodel, tmp_path
):
    # Clause (2 v 3 v 4): 3 is true, 4 unassigned -> satisfied.
    cnf = write(tmp_path, "f.cnf", "p cnf 4 2\n1 -2 0\n2 3 4 0\n")
    log = write(tmp_path, "f.log", "s SATISFIABLE\nv 1 -2 3 0\n")
    proc = run(checkmodel, cnf, log)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "MODEL VERIFIED" in proc.stdout


def test_clause_depending_on_unassigned_variable_fails(checkmodel, tmp_path):
    # Clause (4) depends solely on unassigned variable 4 -> not satisfied.
    cnf = write(tmp_path, "f.cnf", "p cnf 4 3\n1 -2 0\n2 3 0\n4 0\n")
    log = write(tmp_path, "f.log", "s SATISFIABLE\nv 1 -2 3 0\n")
    proc = run(checkmodel, cnf, log)
    assert proc.returncode == 1
    assert "MODEL FAILED" in proc.stdout
    assert "clause 3" in proc.stdout


def test_missing_cnf_file(checkmodel, tmp_path):
    log = write(tmp_path, "f.log", "s SATISFIABLE\nv 1 0\n")
    proc = run(checkmodel, tmp_path / "nope.cnf", log)
    assert proc.returncode == 2
    assert "ERROR" in proc.stderr


def test_missing_log_file(checkmodel, tmp_path):
    cnf = write(tmp_path, "f.cnf", TINY_CNF)
    proc = run(checkmodel, cnf, tmp_path / "nope.log")
    assert proc.returncode == 2
    assert "ERROR" in proc.stderr


def test_usage_error(checkmodel):
    proc = subprocess.run([checkmodel], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 2
    assert "usage" in proc.stderr.lower()


def test_empty_clause_never_satisfied(checkmodel, tmp_path):
    cnf = write(tmp_path, "f.cnf", "p cnf 1 2\n1 0\n0\n")
    log = write(tmp_path, "f.log", "s SATISFIABLE\nv 1 0\n")
    proc = run(checkmodel, cnf, log)
    assert proc.returncode == 1
    assert "MODEL FAILED" in proc.stdout
    assert "clause 2" in proc.stdout
