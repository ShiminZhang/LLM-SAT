"""Comprehensive tests for MutationExperiencePool.update().

Run from repo root with:
  python src/experience_pool/tests/test_comprehensive_update_mutation.py
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

# Add src/ to sys.path so we can import from experience_pool
src_dir = str(Path(__file__).parent.parent.parent.resolve())
if src_dir not in sys.path:
	sys.path.insert(0, src_dir)

from experience_pool.manager import ExperiencePoolManager
from experience_pool.runtime import SharedRuntime
from experience_pool.types import OutcomeLabel


def print_header(title: str) -> None:
	print("\n" + "=" * 92)
	print(title)
	print("=" * 92)


def print_subheader(title: str) -> None:
	print("\n" + "-" * 92)
	print(title)
	print("-" * 92)


class TestRunner:
	"""Tiny assertion helper with pass/fail counters."""

	def __init__(self) -> None:
		self.passed = 0
		self.failed = 0

	def check(self, condition: bool, message: str) -> None:
		if condition:
			self.passed += 1
			print(f"[PASS] {message}")
		else:
			self.failed += 1
			print(f"[FAIL] {message}")

	def expect_exception(
		self,
		fn: Callable[[], object],
		expected_exception: type[BaseException],
		message: str,
	) -> None:
		try:
			fn()
		except expected_exception as exc:
			self.passed += 1
			print(f"[PASS] {message} -> caught {type(exc).__name__}: {exc}")
			return
		except Exception as exc:  # noqa: BLE001
			self.failed += 1
			print(f"[FAIL] {message} -> unexpected {type(exc).__name__}: {exc}")
			return

		self.failed += 1
		print(f"[FAIL] {message} -> no exception raised")


def reset_runtime_singleton() -> None:
	SharedRuntime._instance = None


def make_algo_payload(
	algo_id: str,
	*,
	role: str,
	description: str,
	analysis: str,
	raw_par2,
	parent_id,
) -> dict:
	"""Build minimal algorithm JSON payload used by update() parser."""

	return {
		"id": algo_id,
		"function_name": "kissat_restarting",
		"description": description,
		"role": role,
		"status": "evaluated",
		"last_updated": "2026-03-28T12:00:00",
		"code_id_list": [f"code_{algo_id}_1"],
		"parent_id": parent_id,
		"parent_algorithm_description": None,
		"raw_par2_score": raw_par2,
		"normalized_par2_score": [0, 0, 0, 0, 0],
		"analysis": analysis,
		"prompt": "dummy_prompt",
		"other_metrics": {},
	}


def write_algo_json(case_root: Path, split: str, folder_id: str, payload: dict, json_id: str | None = None) -> Path:
	"""Write `<case>/<split>/algorithm_<folder_id>/<json_id or folder_id>.json`."""

	algo_dir = case_root / split / f"algorithm_{folder_id}"
	algo_dir.mkdir(parents=True, exist_ok=True)
	file_id = json_id if json_id is not None else folder_id
	out_path = algo_dir / f"{file_id}.json"
	out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
	return out_path


def base_raw(score5: float):
	return [score5 + 10, score5 + 20, score5 + 5, score5 + 15, score5]


def run_case_1_happy_path(runner: TestRunner, mgr: ExperiencePoolManager, root: Path) -> None:
	print_subheader("Case 1: happy path + threshold + neutral")
	case = root / "case1_happy"

	write_algo_json(
		case,
		"leaders",
		"L1",
		make_algo_payload(
			"L1",
			role="leader",
			description="C1 leader L1 description",
			analysis="C1 leader analysis",
			raw_par2=base_raw(100.0),
			parent_id=None,
		),
	)

	write_algo_json(
		case,
		"members",
		"L1_M_good",
		make_algo_payload(
			"L1_M_good",
			role="member",
			description="C1 member good",
			analysis="C1 member good analysis",
			raw_par2=base_raw(85.0),  # +15%
			parent_id=["L1"],
		),
	)
	write_algo_json(
		case,
		"members",
		"L1_M_bad",
		make_algo_payload(
			"L1_M_bad",
			role="member",
			description="C1 member bad",
			analysis="C1 member bad analysis",
			raw_par2=base_raw(120.0),  # -20%
			parent_id=["L1"],
		),
	)
	write_algo_json(
		case,
		"members",
		"L1_M_neutral",
		make_algo_payload(
			"L1_M_neutral",
			role="member",
			description="C1 member neutral",
			analysis="C1 member neutral analysis",
			raw_par2=base_raw(95.0),  # +5% -> neutral
			parent_id=["L1"],
		),
	)

	summary = mgr.update("mutation", input_dir=case, threshold=0.10)
	print(summary)
	runner.check(summary["leaders_loaded"] == 1, "Case1 loaded one leader")
	runner.check(summary["members_loaded"] == 3, "Case1 loaded three members")
	runner.check(summary["persisted_good"] == 1, "Case1 persisted one good mutation")
	runner.check(summary["persisted_bad"] == 1, "Case1 persisted one bad mutation")
	runner.check(summary["neutral_skipped"] == 1, "Case1 skipped one neutral mutation")

	good_hits = mgr.retrieve(
		pool_name="mutation",
		query_text="C1 leader L1 description",
		top_k=5,
		outcome=OutcomeLabel.GOOD,
	)
	bad_hits = mgr.retrieve(
		pool_name="mutation",
		query_text="C1 leader L1 description",
		top_k=5,
		outcome=OutcomeLabel.BAD,
	)
	runner.check(
		len(good_hits) >= 1 and good_hits[0].payload.leader_algorithm_id == "L1" and good_hits[0].payload.member_algorithm_id == "L1_M_good",
		"Case1 GOOD retrieval preserves leader/member IDs from update ingestion",
	)
	runner.check(
		len(bad_hits) >= 1 and bad_hits[0].payload.leader_algorithm_id == "L1" and bad_hits[0].payload.member_algorithm_id == "L1_M_bad",
		"Case1 BAD retrieval preserves leader/member IDs from update ingestion",
	)


def run_case_2_threshold_boundary(runner: TestRunner, mgr: ExperiencePoolManager, root: Path) -> None:
	print_subheader("Case 2: exact threshold boundary inclusion")
	case = root / "case2_threshold"

	write_algo_json(
		case,
		"leaders",
		"L2",
		make_algo_payload(
			"L2",
			role="leader",
			description="C2 leader",
			analysis="C2 analysis",
			raw_par2=base_raw(200.0),
			parent_id=[],
		),
	)

	write_algo_json(
		case,
		"members",
		"L2_M_good_edge",
		make_algo_payload(
			"L2_M_good_edge",
			role="member",
			description="C2 good edge",
			analysis="C2 good edge analysis",
			raw_par2=base_raw(180.0),  # +10%
			parent_id=["L2"],
		),
	)
	write_algo_json(
		case,
		"members",
		"L2_M_bad_edge",
		make_algo_payload(
			"L2_M_bad_edge",
			role="member",
			description="C2 bad edge",
			analysis="C2 bad edge analysis",
			raw_par2=base_raw(220.0),  # -10%
			parent_id=["L2"],
		),
	)

	summary = mgr.update("mutation", input_dir=case, threshold=0.10)
	print(summary)
	runner.check(summary["persisted_good"] == 1, "Case2 good edge accepted")
	runner.check(summary["persisted_bad"] == 1, "Case2 bad edge accepted")
	runner.check(summary["neutral_skipped"] == 0, "Case2 no neutral edge examples")


def run_case_3_missing_dirs(runner: TestRunner, mgr: ExperiencePoolManager, root: Path) -> None:
	print_subheader("Case 3: missing leaders/members directories")
	case_no_leaders = root / "case3_no_leaders"
	(case_no_leaders / "members").mkdir(parents=True, exist_ok=True)

	runner.expect_exception(
		lambda: mgr.update("mutation", input_dir=case_no_leaders, threshold=0.10),
		ValueError,
		"Case3 raises on missing leaders directory",
	)

	case_no_members = root / "case3_no_members"
	(case_no_members / "leaders").mkdir(parents=True, exist_ok=True)

	runner.expect_exception(
		lambda: mgr.update("mutation", input_dir=case_no_members, threshold=0.10),
		ValueError,
		"Case3 raises on missing members directory",
	)


def run_case_4_filename_id_rules(runner: TestRunner, mgr: ExperiencePoolManager, root: Path) -> None:
	print_subheader("Case 4: filename convention and id mismatch")
	case = root / "case4_naming"

	# Leader folder exists but required L4_missing.json does not exist.
	lead_missing = case / "leaders" / "algorithm_L4_missing"
	lead_missing.mkdir(parents=True, exist_ok=True)
	(lead_missing / "wrong_name.json").write_text("{}", encoding="utf-8")

	# Leader with id mismatch in payload.
	write_algo_json(
		case,
		"leaders",
		"L4_id_mismatch",
		make_algo_payload(
			"L4_payload_other",
			role="leader",
			description="C4 leader mismatch",
			analysis="C4 analysis",
			raw_par2=base_raw(100.0),
			parent_id=None,
		),
	)

	# One valid leader to allow some member mapping.
	write_algo_json(
		case,
		"leaders",
		"L4_valid",
		make_algo_payload(
			"L4_valid",
			role="leader",
			description="C4 valid leader",
			analysis="C4 valid leader analysis",
			raw_par2=base_raw(100.0),
			parent_id=None,
		),
	)

	# Member folder with wrong JSON name.
	mem_wrong_name = case / "members" / "algorithm_M4_wrong_name"
	mem_wrong_name.mkdir(parents=True, exist_ok=True)
	(mem_wrong_name / "not_expected.json").write_text("{}", encoding="utf-8")

	# Member id mismatch.
	write_algo_json(
		case,
		"members",
		"M4_id_mismatch",
		make_algo_payload(
			"M4_payload_other",
			role="member",
			description="C4 member mismatch",
			analysis="C4 member mismatch analysis",
			raw_par2=base_raw(80.0),
			parent_id=["L4_valid"],
		),
	)

	# Valid member
	write_algo_json(
		case,
		"members",
		"M4_valid",
		make_algo_payload(
			"M4_valid",
			role="member",
			description="C4 valid member",
			analysis="C4 valid member analysis",
			raw_par2=base_raw(80.0),
			parent_id=["L4_valid"],
		),
	)

	buf = io.StringIO()
	with redirect_stdout(buf):
		summary = mgr.update("mutation", input_dir=case, threshold=0.10)
	out = buf.getvalue()
	print(summary)
	print("Captured stdout (truncated):", out[:250].replace("\n", " | "), "...")

	runner.check(summary["leaders_seen"] == 3, "Case4 saw three leader folders")
	runner.check(summary["leaders_loaded"] == 1, "Case4 loaded one valid leader")
	runner.check(summary["members_seen"] == 3, "Case4 saw three member folders")
	runner.check(summary["members_loaded"] == 1, "Case4 loaded one valid member")
	runner.check(summary["persisted_good"] == 1, "Case4 persisted valid member as good")
	runner.check(
		"ID mismatch" in out and "M4_id_mismatch" in out,
		"Case4 prints member ID mismatch error with source",
	)


def run_case_5_parent_cardinality(runner: TestRunner, mgr: ExperiencePoolManager, root: Path) -> None:
	print_subheader("Case 5: parent_id cardinality/type rules")
	case = root / "case5_parent_cardinality"

	# One valid leader.
	write_algo_json(
		case,
		"leaders",
		"L5_ok",
		make_algo_payload(
			"L5_ok",
			role="leader",
			description="C5 leader ok",
			analysis="C5 leader ok analysis",
			raw_par2=base_raw(100.0),
			parent_id=None,
		),
	)

	# Invalid leaders by parent cardinality/type.
	write_algo_json(
		case,
		"leaders",
		"L5_bad_type",
		make_algo_payload(
			"L5_bad_type",
			role="leader",
			description="C5 leader bad type",
			analysis="C5 analysis",
			raw_par2=base_raw(100.0),
			parent_id="X",
		),
	)
	write_algo_json(
		case,
		"leaders",
		"L5_bad_multi",
		make_algo_payload(
			"L5_bad_multi",
			role="leader",
			description="C5 leader bad multi",
			analysis="C5 analysis",
			raw_par2=base_raw(100.0),
			parent_id=["A", "B"],
		),
	)

	# Members with invalid parent definitions.
	write_algo_json(
		case,
		"members",
		"M5_type_str",
		make_algo_payload(
			"M5_type_str",
			role="member",
			description="C5 m str",
			analysis="C5 m str analysis",
			raw_par2=base_raw(80.0),
			parent_id="L5_ok",
		),
	)
	write_algo_json(
		case,
		"members",
		"M5_zero",
		make_algo_payload(
			"M5_zero",
			role="member",
			description="C5 m zero",
			analysis="C5 m zero analysis",
			raw_par2=base_raw(80.0),
			parent_id=[],
		),
	)
	write_algo_json(
		case,
		"members",
		"M5_multi",
		make_algo_payload(
			"M5_multi",
			role="member",
			description="C5 m multi",
			analysis="C5 m multi analysis",
			raw_par2=base_raw(80.0),
			parent_id=["L5_ok", "Other"],
		),
	)
	write_algo_json(
		case,
		"members",
		"M5_blank",
		make_algo_payload(
			"M5_blank",
			role="member",
			description="C5 m blank",
			analysis="C5 m blank analysis",
			raw_par2=base_raw(80.0),
			parent_id=["   "],
		),
	)

	# One valid member.
	write_algo_json(
		case,
		"members",
		"M5_ok",
		make_algo_payload(
			"M5_ok",
			role="member",
			description="C5 m ok",
			analysis="C5 m ok analysis",
			raw_par2=base_raw(80.0),
			parent_id=["L5_ok"],
		),
	)

	buf = io.StringIO()
	with redirect_stdout(buf):
		summary = mgr.update("mutation", input_dir=case, threshold=0.10)
	out = buf.getvalue()
	print(summary)

	runner.check(summary["leaders_seen"] == 3, "Case5 saw three leaders")
	runner.check(summary["leaders_loaded"] == 1, "Case5 loaded only one valid leader")
	runner.check(summary["parent_cardinality_errors"] == 4, "Case5 counted four member parent cardinality errors")
	runner.check(summary["persisted_good"] == 1, "Case5 persisted one valid good member")
	runner.check(
		"Expected list with exactly 1 parent_id" in out and "M5_type_str" in out,
		"Case5 logs type error with member id and context",
	)


def run_case_6_validation_and_mapping(runner: TestRunner, mgr: ExperiencePoolManager, root: Path) -> None:
	print_subheader("Case 6: payload validation, score extraction, mapping, leader score=0")
	case = root / "case6_validation"

	# Leaders
	write_algo_json(
		case,
		"leaders",
		"L6_valid",
		make_algo_payload(
			"L6_valid",
			role="leader",
			description="C6 valid leader",
			analysis="C6 valid analysis",
			raw_par2=base_raw(100.0),
			parent_id=[],
		),
	)
	write_algo_json(
		case,
		"leaders",
		"L6_zero",
		make_algo_payload(
			"L6_zero",
			role="leader",
			description="C6 zero leader",
			analysis="C6 zero analysis",
			raw_par2=base_raw(0.0),
			parent_id=None,
		),
	)

	# Invalid leader examples
	write_algo_json(
		case,
		"leaders",
		"L6_empty_desc",
		make_algo_payload(
			"L6_empty_desc",
			role="leader",
			description="",
			analysis="ok",
			raw_par2=base_raw(100.0),
			parent_id=None,
		),
	)
	write_algo_json(
		case,
		"leaders",
		"L6_bad_len",
		make_algo_payload(
			"L6_bad_len",
			role="leader",
			description="ok",
			analysis="ok",
			raw_par2=[1, 2, 3, 4],
			parent_id=None,
		),
	)

	# Members: validity matrix
	write_algo_json(
		case,
		"members",
		"M6_valid",
		make_algo_payload(
			"M6_valid",
			role="member",
			description="C6 valid member",
			analysis="C6 valid member analysis",
			raw_par2=base_raw(80.0),
			parent_id=["L6_valid"],
		),
	)
	write_algo_json(
		case,
		"members",
		"M6_empty_desc",
		make_algo_payload(
			"M6_empty_desc",
			role="member",
			description="",
			analysis="x",
			raw_par2=base_raw(80.0),
			parent_id=["L6_valid"],
		),
	)
	write_algo_json(
		case,
		"members",
		"M6_empty_analysis",
		make_algo_payload(
			"M6_empty_analysis",
			role="member",
			description="x",
			analysis="",
			raw_par2=base_raw(80.0),
			parent_id=["L6_valid"],
		),
	)
	write_algo_json(
		case,
		"members",
		"M6_bad_len",
		make_algo_payload(
			"M6_bad_len",
			role="member",
			description="x",
			analysis="x",
			raw_par2=[1, 2, 3, 4],
			parent_id=["L6_valid"],
		),
	)
	write_algo_json(
		case,
		"members",
		"M6_none_score",
		make_algo_payload(
			"M6_none_score",
			role="member",
			description="x",
			analysis="x",
			raw_par2=[1, 2, 3, 4, None],
			parent_id=["L6_valid"],
		),
	)
	write_algo_json(
		case,
		"members",
		"M6_bad_type_score",
		make_algo_payload(
			"M6_bad_type_score",
			role="member",
			description="x",
			analysis="x",
			raw_par2=[1, 2, 3, 4, "abc"],
			parent_id=["L6_valid"],
		),
	)
	write_algo_json(
		case,
		"members",
		"M6_missing_leader",
		make_algo_payload(
			"M6_missing_leader",
			role="member",
			description="x",
			analysis="x",
			raw_par2=base_raw(80.0),
			parent_id=["L_DOES_NOT_EXIST"],
		),
	)
	write_algo_json(
		case,
		"members",
		"M6_leader_zero",
		make_algo_payload(
			"M6_leader_zero",
			role="member",
			description="x",
			analysis="x",
			raw_par2=base_raw(50.0),
			parent_id=["L6_zero"],
		),
	)

	summary = mgr.update("mutation", input_dir=case, threshold=0.10)
	print(summary)
	runner.check(summary["leaders_loaded"] == 2, "Case6 loaded two valid leaders")
	runner.check(summary["members_loaded"] == 1, "Case6 loaded only one fully valid member")
	runner.check(summary["persisted_good"] == 1, "Case6 persisted one valid good mutation")
	runner.check(summary["missing_leader_skipped"] == 1, "Case6 counted missing leader mapping skip")
	runner.check(summary["invalid_skipped"] >= 6, "Case6 counted multiple invalid skips")


def run_case_7_dedupe(runner: TestRunner, mgr: ExperiencePoolManager, root: Path) -> None:
	print_subheader("Case 7: repeated update dedupe behavior")
	case = root / "case7_dedupe"

	write_algo_json(
		case,
		"leaders",
		"L7",
		make_algo_payload(
			"L7",
			role="leader",
			description="C7 leader",
			analysis="C7 leader analysis",
			raw_par2=base_raw(100.0),
			parent_id=None,
		),
	)
	write_algo_json(
		case,
		"members",
		"L7_M_good",
		make_algo_payload(
			"L7_M_good",
			role="member",
			description="C7 member good",
			analysis="C7 member good analysis",
			raw_par2=base_raw(80.0),
			parent_id=["L7"],
		),
	)
	write_algo_json(
		case,
		"members",
		"L7_M_bad",
		make_algo_payload(
			"L7_M_bad",
			role="member",
			description="C7 member bad",
			analysis="C7 member bad analysis",
			raw_par2=base_raw(130.0),
			parent_id=["L7"],
		),
	)

	s1 = mgr.update("mutation", input_dir=case, threshold=0.10)
	s2 = mgr.update("mutation", input_dir=case, threshold=0.10)
	print("first run:", s1)
	print("second run:", s2)

	runner.check(s1["persisted_created"] == 2, "Case7 first run creates two records")
	runner.check(s1["persisted_deduped"] == 0, "Case7 first run has no dedupes")
	runner.check(s2["persisted_created"] == 0, "Case7 second run creates no new records")
	runner.check(s2["persisted_deduped"] == 2, "Case7 second run dedupes both records")


def run_case_8_manager_update_no_pool(runner: TestRunner, mgr: ExperiencePoolManager) -> None:
	print_subheader("Case 8: manager.update no pool_name remains no-op")
	result = mgr.update(event_type="noop")
	runner.check(result is None, "Manager update without pool_name returns None")


def run_small_real_use_cases_demo(mgr: ExperiencePoolManager, root: Path) -> None:
	"""Print small practical dummy use cases for visual inspection."""

	print_subheader("Demo Use Cases: small practical examples")
	case = root / "demo_small_use_cases"

	# Leader baseline
	write_algo_json(
		case,
		"leaders",
		"L_demo",
		make_algo_payload(
			"L_demo",
			role="leader",
			description="Leader-demo: conflict/LBD mixed restart trigger.",
			analysis="Baseline leader for demo use cases.",
			raw_par2=base_raw(100.0),
			parent_id=None,
		),
	)

	# Good mutation (>10% better)
	write_algo_json(
		case,
		"members",
		"L_demo_M_good",
		make_algo_payload(
			"L_demo_M_good",
			role="member",
			description="Member-good: adaptive median-based threshold and volatility guard.",
			analysis="Cuts unnecessary restarts; reacts faster to spikes.",
			raw_par2=base_raw(88.0),
			parent_id=["L_demo"],
		),
	)

	# Bad mutation (>10% worse)
	write_algo_json(
		case,
		"members",
		"L_demo_M_bad",
		make_algo_payload(
			"L_demo_M_bad",
			role="member",
			description="Member-bad: stacked hard gates before restart.",
			analysis="Delays recovery from poor search trajectories.",
			raw_par2=base_raw(123.0),
			parent_id=["L_demo"],
		),
	)

	# Neutral mutation (<10% better, should be ignored)
	write_algo_json(
		case,
		"members",
		"L_demo_M_neutral",
		make_algo_payload(
			"L_demo_M_neutral",
			role="member",
			description="Member-neutral: minor trigger smoothing.",
			analysis="Small change with no material gain.",
			raw_par2=base_raw(96.0),
			parent_id=["L_demo"],
		),
	)

	# Invalid parent cardinality (shows detailed error with source path)
	write_algo_json(
		case,
		"members",
		"L_demo_M_invalid_parent",
		make_algo_payload(
			"L_demo_M_invalid_parent",
			role="member",
			description="Member-invalid-parent demo",
			analysis="Should raise cardinality error and skip.",
			raw_par2=base_raw(70.0),
			parent_id=["L_demo", "EXTRA_PARENT"],
		),
	)

	print("\n[Demo] Running update(pool='mutation', threshold=0.10) ...")
	demo_summary = mgr.update("mutation", input_dir=case, threshold=0.10)
	print("[Demo] Summary:", demo_summary)

	good_hits = mgr.retrieve(
		pool_name="mutation",
		query_text="Leader-demo: conflict/LBD mixed restart trigger.",
		top_k=5,
		outcome=OutcomeLabel.GOOD,
	)
	bad_hits = mgr.retrieve(
		pool_name="mutation",
		query_text="Leader-demo: conflict/LBD mixed restart trigger.",
		top_k=5,
		outcome=OutcomeLabel.BAD,
	)

	print("\n[Demo] Retrieved GOOD examples:")
	for i, hit in enumerate(good_hits, 1):
		print(f"  [{i}] score={hit.score:.4f} | id={hit.record_id[:10]}... | member={hit.payload.member_algorithm_description}")

	print("\n[Demo] Retrieved BAD examples:")
	for i, hit in enumerate(bad_hits, 1):
		print(f"  [{i}] score={hit.score:.4f} | id={hit.record_id[:10]}... | member={hit.payload.member_algorithm_description}")


def main() -> None:
	print_header("Mutation Update Comprehensive Test")

	runner = TestRunner()

	with TemporaryDirectory(prefix="exp_pool_update_mutation_") as tmp:
		tmp_root = Path(tmp)
		case_root = tmp_root / "cases"
		case_root.mkdir(parents=True, exist_ok=True)

		data_root = tmp_root / "experience_data"

		reset_runtime_singleton()
		manager = ExperiencePoolManager(data_root=data_root)

		run_case_1_happy_path(runner, manager, case_root)
		run_case_2_threshold_boundary(runner, manager, case_root)
		run_case_3_missing_dirs(runner, manager, case_root)
		run_case_4_filename_id_rules(runner, manager, case_root)
		run_case_5_parent_cardinality(runner, manager, case_root)
		run_case_6_validation_and_mapping(runner, manager, case_root)
		run_case_7_dedupe(runner, manager, case_root)
		run_case_8_manager_update_no_pool(runner, manager)
		run_small_real_use_cases_demo(manager, case_root)

	print_header("Summary")
	total = runner.passed + runner.failed
	print(f"Total checks: {total}")
	print(f"Passed: {runner.passed}")
	print(f"Failed: {runner.failed}")

	if runner.failed == 0:
		print("\nALL CHECKS PASSED ✅")
	else:
		print("\nSOME CHECKS FAILED ❌")
		raise SystemExit(1)


if __name__ == "__main__":
	main()

