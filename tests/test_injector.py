"""Tests for FunctionInjector: verified splice, stale-registry relocation,
and the hardlink-safety (new inode) write invariant."""

import os
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llmsat.code_injection.injector import FunctionInjector
from llmsat.code_injection.registry import FunctionRegistry

FAKE_SOURCE = """\
#include "internal.h"

// a comment with a stray brace {
static const char *marker = "string with } brace";

void
helper_function (kissat *solver) {
  if (solver) {
    do_thing ();
  }
}

void target_function (kissat *solver) {
  const double x = 1.0;
  use (x);
}

int
other_function (void) {
  return 0;
}
"""

NEW_CODE = """void target_function (kissat *solver) {
  const double x = 2.0;  /* injected */
  use (x);
}
"""


@pytest.fixture
def solver_dir(tmp_path):
    src = tmp_path / "solver" / "src"
    src.mkdir(parents=True)
    (src / "fake.c").write_text(FAKE_SOURCE)
    return tmp_path / "solver"


def make_registry(tmp_path, start_line, end_line):
    reg_path = tmp_path / "function_registry.yaml"
    reg_path.write_text(yaml.dump({
        "version": "1.0",
        "functions": {
            "target_function": {
                "file": "src/fake.c",
                "start_line": start_line,
                "end_line": end_line,
                "signature": "void target_function (kissat *solver)",
            }
        },
    }))
    return FunctionRegistry(str(reg_path))


def test_replace_at_correct_registry_lines(solver_dir, tmp_path):
    # target_function spans lines 13-16 of FAKE_SOURCE
    reg = make_registry(tmp_path, 13, 16)
    inj = FunctionInjector(reg, str(solver_dir))
    inj.replace_function(str(solver_dir), "target_function", NEW_CODE)
    text = (solver_dir / "src" / "fake.c").read_text()
    assert "/* injected */" in text
    assert "helper_function" in text and "other_function" in text
    assert text.count("target_function") == 1


def test_stale_registry_relocates(solver_dir, tmp_path):
    # Wrong range pointing at helper_function's area: must relocate, not clobber
    reg = make_registry(tmp_path, 6, 11)
    inj = FunctionInjector(reg, str(solver_dir))
    inj.replace_function(str(solver_dir), "target_function", NEW_CODE)
    text = (solver_dir / "src" / "fake.c").read_text()
    assert "/* injected */" in text
    assert "helper_function" in text, "relocation must not overwrite the wrong function"
    assert "do_thing" in text


def test_missing_function_raises(solver_dir, tmp_path):
    reg_path = tmp_path / "function_registry.yaml"
    reg_path.write_text(yaml.dump({
        "version": "1.0",
        "functions": {
            "deleted_function": {
                "file": "src/fake.c", "start_line": 6, "end_line": 11,
                "signature": "void deleted_function (kissat *solver)",
            }
        },
    }))
    reg = FunctionRegistry(str(reg_path))
    inj = FunctionInjector(reg, str(solver_dir))
    with pytest.raises(ValueError, match="could not be relocated"):
        inj.replace_function(str(solver_dir), "deleted_function", NEW_CODE)
    assert (solver_dir / "src" / "fake.c").read_text() == FAKE_SOURCE, "file must be untouched"


def test_relocation_ignores_braces_in_comments_and_strings(solver_dir, tmp_path):
    reg = make_registry(tmp_path, 1, 3)  # stale range over the comment/string region
    inj = FunctionInjector(reg, str(solver_dir))
    rel = inj._relocate_function(
        (solver_dir / "src" / "fake.c").read_text().splitlines(keepends=True),
        "target_function",
    )
    assert rel is not None
    start, end = rel
    # 0-based line 12 .. exclusive 16 == 1-based 13-16
    assert (start, end) == (12, 16)


def test_write_preserves_hardlinked_original(solver_dir, tmp_path):
    """The corruption invariant: injecting into a hardlink clone must never
    modify the base solver's inode."""
    base_file = solver_dir / "src" / "fake.c"
    clone_dir = tmp_path / "clone" / "src"
    clone_dir.mkdir(parents=True)
    os.link(base_file, clone_dir / "fake.c")

    reg = make_registry(tmp_path, 13, 16)
    inj = FunctionInjector(reg, str(solver_dir))
    inj.replace_function(str(tmp_path / "clone"), "target_function", NEW_CODE)

    assert "/* injected */" in (clone_dir / "fake.c").read_text()
    assert base_file.read_text() == FAKE_SOURCE, "base solver inode was modified!"
    assert os.stat(base_file).st_nlink == 1, "clone must have a fresh inode"
