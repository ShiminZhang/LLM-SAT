"""Comprehensive tests for CombinationExperiencePool.update().

Run from repo root with:
  python src/experience_pool/tests/test_comprehensive_update_combination.py
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
	print("\n" + "=" * 96)
	print(title)
	print("=" * 96)


def print_subheader(title: str) -> None:
	print("\n" + "-" * 96)
	print(title)
	print("-" * 96)


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


def base_raw(score5: float):
	return [score5 + 10, score5 + 20, score5 + 5, score5 + 15, score5]


def make_algo_payload(
	algo_id: str,
	*,
	role: str,
	description: str,
	analysis: str,
	raw_par2,
	parent_id,
) -> dict:
	"""Build minimal algorithm JSON payload used by combination update parser."""

	return {
		"id": algo_id,
		"function_name": "kissat_restarting",
		"description": description,
		"role": role,
		"status": "evaluated",
		"last_updated": "2026-03-29T10:00:00",
		"code_id_list": [f"code_{algo_id}_1"],
		"parent_id": parent_id,
		"parent_algorithm_description": None,
		"raw_par2_score": raw_par2,
		"normalized_par2_score": [0, 0, 0, 0, 0],
		"analysis": analysis,
		"prompt": "dummy_prompt",
		"other_metrics": {},
	}


def write_algo_json(root: Path, split: str, folder_id: str, payload: dict, json_id: str | None = None) -> Path:
	"""Write `<root>/<split>/algorithm_<folder_id>/<json_id or folder_id>.json`."""

	algo_dir = root / split / f"algorithm_{folder_id}"
	algo_dir.mkdir(parents=True, exist_ok=True)
	file_id = json_id if json_id is not None else folder_id
	out_path = algo_dir / f"{file_id}.json"
	out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
	return out_path


def run_case_1_happy_threshold_neutral(runner: TestRunner, mgr: ExperiencePoolManager, root: Path) -> None:
	print_subheader("Case 1: happy path + threshold + neutral")

	parent_root = root / "case1_parents"
	combined_root = root / "case1_combined"
	(parent_root / "members").mkdir(parents=True, exist_ok=True)

	# Parents for combination source.
	write_algo_json(
		parent_root,
		"leaders",
		"L1",
		make_algo_payload(
			"L1",
			role="leader",
			description="Case1 parent L1 description",
			analysis="Case1 parent L1 analysis",
			raw_par2=base_raw(100.0),
			parent_id=[],
		),
	)
	write_algo_json(
		parent_root,
		"leaders",
		"L2",
		make_algo_payload(
			"L2",
			role="leader",
			description="Case1 parent L2 description",
			analysis="Case1 parent L2 analysis",
			raw_par2=base_raw(101.0),
			parent_id=[],
		),
	)

	# Combined offspring: good / bad / neutral.
	write_algo_json(
		combined_root,
		"members",
		"L1_L2_good",
		make_algo_payload(
			"L1_L2_good",
			role="member",
			description="Case1 offspring good",
			analysis="Case1 offspring good analysis",
			raw_par2=base_raw(85.0),  # better than both; +15% vs L1
			parent_id=["L1", "L2"],
		),
	)
	write_algo_json(
		combined_root,
		"members",
		"L1_L2_bad",
		make_algo_payload(
			"L1_L2_bad",
			role="member",
			description="Case1 offspring bad",
			analysis="Case1 offspring bad analysis",
			raw_par2=base_raw(140.0),  # worse than both; >=10% degrade vs L2
			parent_id=["L1", "L2"],
		),
	)
	write_algo_json(
		combined_root,
		"members",
		"L1_L2_neutral",
		make_algo_payload(
			"L1_L2_neutral",
			role="member",
			description="Case1 offspring neutral",
			analysis="Case1 offspring neutral analysis",
			raw_par2=base_raw(95.0),  # better than both, <10% improvement vs both
			parent_id=["L1", "L2"],
		),
	)

	summary = mgr.update(
		"combination",
		combined_dir=combined_root,
		parent_source_dir=parent_root,
		threshold=0.10,
	)
	print(summary)

	runner.check(summary["parents_loaded"] == 2, "Case1 loaded two parent algorithms")
	runner.check(summary["combined_loaded"] == 3, "Case1 loaded three combined algorithms")
	runner.check(summary["persisted_good"] == 1, "Case1 persisted one GOOD combination")
	runner.check(summary["persisted_bad"] == 1, "Case1 persisted one BAD combination")
	runner.check(summary["neutral_skipped"] == 1, "Case1 skipped one neutral combination")

	good_hits = mgr.retrieve(
		pool_name="combination",
		query_text=["Case1 parent L1 description", "Case1 parent L2 description"],
		top_k=5,
		outcome=OutcomeLabel.GOOD,
	)
	runner.check(len(good_hits) >= 1, "Case1 GOOD retrieval returns at least one hit")
	if good_hits:
		runner.check(
			good_hits[0].payload.parent_alg1_description == "Case1 parent L1 description"
			and good_hits[0].payload.parent_alg2_description == "Case1 parent L2 description",
			"Case1 retrieval uses parent descriptions from parent-source JSON",
		)


def run_case_2_threshold_edges(runner: TestRunner, mgr: ExperiencePoolManager, root: Path) -> None:
	print_subheader("Case 2: exact threshold boundary inclusion")

	parent_root = root / "case2_parents"
	combined_root = root / "case2_combined"
	(parent_root / "members").mkdir(parents=True, exist_ok=True)

	write_algo_json(
		parent_root,
		"leaders",
		"A2",
		make_algo_payload(
			"A2",
			role="leader",
			description="Case2 parent A",
			analysis="Case2 parent A analysis",
			raw_par2=base_raw(100.0),
			parent_id=[],
		),
	)
	write_algo_json(
		parent_root,
		"leaders",
		"B2",
		make_algo_payload(
			"B2",
			role="leader",
			description="Case2 parent B",
			analysis="Case2 parent B analysis",
			raw_par2=base_raw(110.0),
			parent_id=[],
		),
	)

	write_algo_json(
		combined_root,
		"members",
		"A2_B2_good_edge",
		make_algo_payload(
			"A2_B2_good_edge",
			role="member",
			description="Case2 good edge",
			analysis="Case2 good edge analysis",
			raw_par2=base_raw(90.0),  # exactly +10% vs A2
			parent_id=["A2", "B2"],
		),
	)
	write_algo_json(
		combined_root,
		"members",
		"A2_B2_bad_edge",
		make_algo_payload(
			"A2_B2_bad_edge",
			role="member",
			description="Case2 bad edge",
			analysis="Case2 bad edge analysis",
			raw_par2=base_raw(121.0),  # exactly +10% vs B2 and worse than both
			parent_id=["A2", "B2"],
		),
	)

	summary = mgr.update(
		"combination",
		combined_dir=combined_root,
		parent_source_dir=parent_root,
		threshold=0.10,
	)
	print(summary)
	runner.check(summary["persisted_good"] == 1, "Case2 accepts GOOD edge at threshold")
	runner.check(summary["persisted_bad"] == 1, "Case2 accepts BAD edge at threshold")


def run_case_3_missing_dirs(runner: TestRunner, mgr: ExperiencePoolManager, root: Path) -> None:
	print_subheader("Case 3: missing required directories")

	combined_missing_members = root / "case3_combined_missing_members"
	combined_missing_members.mkdir(parents=True, exist_ok=True)

	parent_ok = root / "case3_parent_ok"
	(parent_ok / "leaders").mkdir(parents=True, exist_ok=True)
	(parent_ok / "members").mkdir(parents=True, exist_ok=True)

	runner.expect_exception(
		lambda: mgr.update(
			"combination",
			combined_dir=combined_missing_members,
			parent_source_dir=parent_ok,
			threshold=0.10,
		),
		ValueError,
		"Case3 raises on missing combined members directory",
	)

	combined_ok = root / "case3_combined_ok"
	(combined_ok / "members").mkdir(parents=True, exist_ok=True)

	parent_missing_leaders = root / "case3_parent_missing_leaders"
	(parent_missing_leaders / "members").mkdir(parents=True, exist_ok=True)
	runner.expect_exception(
		lambda: mgr.update(
			"combination",
			combined_dir=combined_ok,
			parent_source_dir=parent_missing_leaders,
			threshold=0.10,
		),
		ValueError,
		"Case3 raises on missing parent leaders directory",
	)

	parent_missing_members = root / "case3_parent_missing_members"
	(parent_missing_members / "leaders").mkdir(parents=True, exist_ok=True)
	runner.expect_exception(
		lambda: mgr.update(
			"combination",
			combined_dir=combined_ok,
			parent_source_dir=parent_missing_members,
			threshold=0.10,
		),
		ValueError,
		"Case3 raises on missing parent members directory",
	)


def run_case_4_cardinality_missing_parent_and_id_rules(
	runner: TestRunner,
	mgr: ExperiencePoolManager,
	root: Path,
) -> None:
	print_subheader("Case 4: parent_id rules, missing parent mapping, and id mismatch")

	parent_root = root / "case4_parents"
	combined_root = root / "case4_combined"
	(parent_root / "members").mkdir(parents=True, exist_ok=True)

	write_algo_json(
		parent_root,
		"leaders",
		"P1",
		make_algo_payload(
			"P1",
			role="leader",
			description="Case4 parent P1",
			analysis="Case4 parent P1 analysis",
			raw_par2=base_raw(100.0),
			parent_id=[],
		),
	)
	write_algo_json(
		parent_root,
		"leaders",
		"P2",
		make_algo_payload(
			"P2",
			role="leader",
			description="Case4 parent P2",
			analysis="Case4 parent P2 analysis",
			raw_par2=base_raw(110.0),
			parent_id=[],
		),
	)

	# invalid parent_id type
	write_algo_json(
		combined_root,
		"members",
		"C4_bad_type",
		make_algo_payload(
			"C4_bad_type",
			role="member",
			description="Case4 bad type",
			analysis="Case4 bad type analysis",
			raw_par2=base_raw(80.0),
			parent_id="P1,P2",
		),
	)

	# invalid parent cardinality
	write_algo_json(
		combined_root,
		"members",
		"C4_bad_card",
		make_algo_payload(
			"C4_bad_card",
			role="member",
			description="Case4 bad card",
			analysis="Case4 bad card analysis",
			raw_par2=base_raw(80.0),
			parent_id=["P1"],
		),
	)

	# missing parent mapping
	write_algo_json(
		combined_root,
		"members",
		"C4_missing_parent",
		make_algo_payload(
			"C4_missing_parent",
			role="member",
			description="Case4 missing parent",
			analysis="Case4 missing parent analysis",
			raw_par2=base_raw(80.0),
			parent_id=["P1", "PX"],
		),
	)

	# id mismatch payload vs folder
	write_algo_json(
		combined_root,
		"members",
		"C4_folder",
		make_algo_payload(
			"C4_payload_other",
			role="member",
			description="Case4 id mismatch",
			analysis="Case4 id mismatch analysis",
			raw_par2=base_raw(80.0),
			parent_id=["P1", "P2"],
		),
	)

	# valid
	write_algo_json(
		combined_root,
		"members",
		"C4_good",
		make_algo_payload(
			"C4_good",
			role="member",
			description="Case4 valid good",
			analysis="Case4 valid good analysis",
			raw_par2=base_raw(85.0),
			parent_id=["P1", "P2"],
		),
	)

	buf = io.StringIO()
	with redirect_stdout(buf):
		summary = mgr.update(
			"combination",
			combined_dir=combined_root,
			parent_source_dir=parent_root,
			threshold=0.10,
		)
	stdout_text = buf.getvalue()
	print(summary)

	runner.check(summary["parent_cardinality_errors"] == 2, "Case4 captured two parent cardinality/type errors")
	runner.check(summary["missing_parent_skipped"] == 1, "Case4 skipped one combined item for missing parent")
	runner.check(summary["persisted_good"] == 1, "Case4 persisted one valid GOOD record")
	runner.check(
		"ID mismatch" in stdout_text and "C4_payload_other" in stdout_text,
		"Case4 prints ID mismatch error details",
	)


def run_case_5_invalid_text_and_par2(runner: TestRunner, mgr: ExperiencePoolManager, root: Path) -> None:
	print_subheader("Case 5: invalid description/analysis/par2 and skip behavior")

	parent_root = root / "case5_parents"
	combined_root = root / "case5_combined"
	(parent_root / "members").mkdir(parents=True, exist_ok=True)

	write_algo_json(
		parent_root,
		"leaders",
		"Q1",
		make_algo_payload(
			"Q1",
			role="leader",
			description="Case5 parent Q1",
			analysis="Case5 parent Q1 analysis",
			raw_par2=base_raw(100.0),
			parent_id=[],
		),
	)
	write_algo_json(
		parent_root,
		"leaders",
		"Q2",
		make_algo_payload(
			"Q2",
			role="leader",
			description="Case5 parent Q2",
			analysis="Case5 parent Q2 analysis",
			raw_par2=base_raw(110.0),
			parent_id=[],
		),
	)

	# empty description
	write_algo_json(
		combined_root,
		"members",
		"C5_empty_desc",
		make_algo_payload(
			"C5_empty_desc",
			role="member",
			description=" ",
			analysis="C5 analysis",
			raw_par2=base_raw(85.0),
			parent_id=["Q1", "Q2"],
		),
	)

	# empty analysis
	write_algo_json(
		combined_root,
		"members",
		"C5_empty_analysis",
		make_algo_payload(
			"C5_empty_analysis",
			role="member",
			description="C5 desc",
			analysis="",
			raw_par2=base_raw(85.0),
			parent_id=["Q1", "Q2"],
		),
	)

	# invalid par2 length
	write_algo_json(
		combined_root,
		"members",
		"C5_bad_par2_len",
		make_algo_payload(
			"C5_bad_par2_len",
			role="member",
			description="C5 desc",
			analysis="C5 analysis",
			raw_par2=[1, 2, 3],
			parent_id=["Q1", "Q2"],
		),
	)

	# invalid par2 null representative
	write_algo_json(
		combined_root,
		"members",
		"C5_bad_par2_none",
		make_algo_payload(
			"C5_bad_par2_none",
			role="member",
			description="C5 desc",
			analysis="C5 analysis",
			raw_par2=[1, 2, 3, 4, None],
			parent_id=["Q1", "Q2"],
		),
	)

	summary = mgr.update(
		"combination",
		combined_dir=combined_root,
		parent_source_dir=parent_root,
		threshold=0.10,
	)
	print(summary)
	runner.check(summary["combined_seen"] == 4, "Case5 saw four combined folders")
	runner.check(summary["combined_loaded"] == 0, "Case5 loaded zero combined due to validation skips")
	runner.check(summary["invalid_skipped"] >= 4, "Case5 counted invalid combined items")


def run_case_6_dedup_on_second_update(runner: TestRunner, mgr: ExperiencePoolManager, root: Path) -> None:
	print_subheader("Case 6: dedupe behavior when rerunning same update")

	parent_root = root / "case6_parents"
	combined_root = root / "case6_combined"
	(parent_root / "members").mkdir(parents=True, exist_ok=True)

	write_algo_json(
		parent_root,
		"leaders",
		"R1",
		make_algo_payload(
			"R1",
			role="leader",
			description="Case6 parent R1",
			analysis="Case6 parent R1 analysis",
			raw_par2=base_raw(100.0),
			parent_id=[],
		),
	)
	write_algo_json(
		parent_root,
		"leaders",
		"R2",
		make_algo_payload(
			"R2",
			role="leader",
			description="Case6 parent R2",
			analysis="Case6 parent R2 analysis",
			raw_par2=base_raw(120.0),
			parent_id=[],
		),
	)

	write_algo_json(
		combined_root,
		"members",
		"R1_R2_good",
		make_algo_payload(
			"R1_R2_good",
			role="member",
			description="Case6 good",
			analysis="Case6 good analysis",
			raw_par2=base_raw(85.0),
			parent_id=["R1", "R2"],
		),
	)
	write_algo_json(
		combined_root,
		"members",
		"R1_R2_bad",
		make_algo_payload(
			"R1_R2_bad",
			role="member",
			description="Case6 bad",
			analysis="Case6 bad analysis",
			raw_par2=base_raw(150.0),
			parent_id=["R1", "R2"],
		),
	)

	s1 = mgr.update(
		"combination",
		combined_dir=combined_root,
		parent_source_dir=parent_root,
		threshold=0.10,
	)
	s2 = mgr.update(
		"combination",
		combined_dir=combined_root,
		parent_source_dir=parent_root,
		threshold=0.10,
	)

	print("first update:", s1)
	print("second update:", s2)
	runner.check(s1["persisted_created"] == 2, "Case6 first run creates two records")
	runner.check(s2["persisted_created"] == 0, "Case6 second run creates no new records")
	runner.check(s2["persisted_deduped"] == 2, "Case6 second run dedupes both records")


def main() -> None:
	print_header("Comprehensive CombinationExperiencePool.update Test")
	runner = TestRunner()

	with TemporaryDirectory(prefix="exp_pool_comb_update_") as data_root:
		with TemporaryDirectory(prefix="exp_pool_comb_cases_") as cases_root:
			print(f"Isolated data root: {data_root}")
			print(f"Case workspace root: {cases_root}")

			reset_runtime_singleton()
			mgr = ExperiencePoolManager(data_root=data_root)
			root = Path(cases_root)

			run_case_1_happy_threshold_neutral(runner, mgr, root)
			run_case_2_threshold_edges(runner, mgr, root)
			run_case_3_missing_dirs(runner, mgr, root)
			run_case_4_cardinality_missing_parent_and_id_rules(runner, mgr, root)
			run_case_5_invalid_text_and_par2(runner, mgr, root)
			run_case_6_dedup_on_second_update(runner, mgr, root)

	print_header("Summary")
	print(f"Passed: {runner.passed}")
	print(f"Failed: {runner.failed}")

	if runner.failed > 0:
		raise SystemExit(1)


if __name__ == "__main__":
	main()

