"""Tests for the canonical instance-key convention."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llmsat.utils.results import instance_key


def test_plain_cnf():
    assert instance_key("00fd8ac9-2.cnf") == "00fd8ac9-2"


def test_normalised_cnf():
    assert instance_key("00d5a43a-st_890.normalised.cnf") == "00d5a43a-st_890"


def test_solving_log_forms():
    assert instance_key("abc-name.cnf.solving.log") == "abc-name"
    assert instance_key("abc-name.normalised.cnf.solving.log") == "abc-name"


def test_dotted_instance_names_truncate_like_all_existing_artifacts():
    # Matches instance_categories.json / baseline_solving_times.json convention:
    # md5 prefix makes truncation collision-free.
    assert instance_key("081f111a-Break_triple_04_06.xml.cnf") == "081f111a-Break_triple_04_06"


def test_paths_are_handled():
    assert instance_key("/a/b/xyz-inst.cnf.solving.log") == "xyz-inst"
