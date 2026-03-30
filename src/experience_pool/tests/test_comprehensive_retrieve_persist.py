"""Comprehensive validation script for ExperiencePoolManager.

Run from repo root with:
  python src/experience_pool/tests/test_comprehensive_retrieve_persist.py

This script focuses on:
1) Retrieval quality sanity checks (ranking/presence)
2) Outcome filtering and balanced retrieval behavior
3) Deduplication behavior in persist()
4) Edge cases and expected exceptions
5) Human-readable printed diagnostics
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src/ to sys.path so we can import from experience_pool
src_dir = str(Path(__file__).parent.parent.parent.resolve())
if src_dir not in sys.path:
	sys.path.insert(0, src_dir)

from tempfile import TemporaryDirectory
from typing import Callable, List

from experience_pool.manager import ExperiencePoolManager
from experience_pool.runtime import SharedRuntime
from experience_pool.types import (
	AlgorithmExperienceRecord,
	CombinationExperienceRecord,
	MutationExperienceRecord,
	OutcomeLabel,
)


def print_header(title: str) -> None:
	print("\n" + "=" * 90)
	print(title)
	print("=" * 90)


def print_subheader(title: str) -> None:
	print("\n" + "-" * 90)
	print(title)
	print("-" * 90)


class TestRunner:
	"""Minimal test runner with printed pass/fail diagnostics."""

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
			print(
				f"[FAIL] {message} -> caught unexpected {type(exc).__name__}: {exc}"
			)
			return

		self.failed += 1
		print(f"[FAIL] {message} -> no exception raised")


def reset_runtime_singleton() -> None:
	"""Reset singleton so this script can define isolated data roots per run."""

	SharedRuntime._instance = None


def print_hits(label: str, hits: List, max_payload_chars: int = 90) -> None:
	print_subheader(label)
	if not hits:
		print("(no hits)")
		return

	for i, hit in enumerate(hits, 1):
		payload_preview = str(hit.payload)
		if len(payload_preview) > max_payload_chars:
			payload_preview = payload_preview[:max_payload_chars] + "..."
		print(
			f"[{i}] score={hit.score:.6f}, pool={hit.pool_name}, "
			f"outcome={hit.outcome.value}, record_id={hit.record_id[:12]}..., "
			f"payload={payload_preview}"
		)


def main() -> None:
	print_header("Experience Pool Comprehensive Test")

	runner = TestRunner()

	with TemporaryDirectory(prefix="exp_pool_comprehensive_") as data_root:
		print(f"Using isolated data root: {data_root}")

		# Ensure this script run does not share singleton state with previous runs.
		reset_runtime_singleton()
		manager = ExperiencePoolManager(data_root=data_root)
		manager2 = ExperiencePoolManager(data_root=data_root)

		print_subheader("A) Runtime singleton behavior")
		runner.check(
			manager.runtime is manager2.runtime,
			"Managers in same process share one runtime instance",
		)

		print_subheader("B) Empty partition and top_k edge")
		empty_alg = manager.retrieve(
			pool_name="algorithm",
			query_text="any algorithm description",
			top_k=5,
			outcome=OutcomeLabel.BAD,
		)
		print_hits("Initial empty retrieval (algorithm/bad)", empty_alg)
		runner.check(
			len(empty_alg) == 0,
			"Empty partition retrieval returns no results",
		)

		top_k_zero = manager.retrieve(
			pool_name="algorithm",
			query_text="any algorithm description",
			top_k=0,
			outcome=OutcomeLabel.BAD,
		)
		runner.check(top_k_zero == [], "top_k=0 returns empty list")

		print_subheader("C) Persist + dedupe + schema/outcome validation")

		alg_bad_1 = AlgorithmExperienceRecord(
			algorithm_description=(
				"Restart every fixed 8 conflicts without considering LBD trend, "
				"decision-level variance, or trail volatility."
			),
			analysis=(
				"Rigid schedule causes restart churn and shallow exploration."
			),
			algorithm_id="alg_c_bad_1",
		)
		alg_bad_2 = AlgorithmExperienceRecord(
			algorithm_description=(
				"Delay all restarts until conflict count > 1e7, then restart aggressively."
			),
			analysis=(
				"Late restart delays recovery from bad branching trajectories."
			),
			algorithm_id="alg_c_bad_2",
		)

		rec1 = manager.persist("algorithm", alg_bad_1, OutcomeLabel.BAD)
		rec2 = manager.persist("algorithm", alg_bad_2, OutcomeLabel.BAD)
		rec1_dup = manager.persist("algorithm", alg_bad_1, OutcomeLabel.BAD)
		print("insert 1:", rec1)
		print("insert 2:", rec2)
		print("dup insert:", rec1_dup)

		runner.check(rec1.created, "First algorithm insert is created=True")
		runner.check(rec2.created, "Second algorithm insert is created=True")
		runner.check(
			not rec1_dup.created,
			"Duplicate algorithm insert is created=False",
		)
		runner.check(
			rec1_dup.partition_size == 2,
			"Duplicate insert does not increase partition size",
		)

		runner.expect_exception(
			lambda: manager.persist(
				"algorithm",
				MutationExperienceRecord(
					leader_algorithm_description="leader",
					member_algorithm_description="member",
					step="step_wrong_schema",
					analysis="analysis",
					leader_algorithm_id="leader_x",
					member_algorithm_id="member_x",
				),
				OutcomeLabel.BAD,
			),
			TypeError,
			"Wrong schema type in persist raises TypeError",
		)

		runner.expect_exception(
			lambda: manager.persist("algorithm", alg_bad_1, OutcomeLabel.GOOD),
			ValueError,
			"Invalid outcome for algorithm pool raises ValueError",
		)

		runner.expect_exception(
			lambda: manager.retrieve(
				pool_name="algorithm",
				query_text="query",
				top_k=3,
				outcome=OutcomeLabel.GOOD,
			),
			ValueError,
			"Invalid outcome in retrieve raises ValueError",
		)

		print_subheader("D) Algorithm retrieval quality sanity")
		alg_query = alg_bad_1.algorithm_description
		alg_hits = manager.retrieve(
			pool_name="algorithm",
			query_text=alg_query,
			top_k=5,
			outcome=OutcomeLabel.BAD,
		)
		print_hits("Algorithm retrieval for identical description", alg_hits)

		runner.check(len(alg_hits) >= 1, "Algorithm retrieval returns at least one hit")
		if alg_hits:
			runner.check(
				alg_hits[0].record_id == rec1.record_id,
				"Identical algorithm description ranks target record first",
			)
			runner.check(
				alg_hits[0].payload.algorithm_id == "alg_c_bad_1",
				"Algorithm retrieval preserves algorithm_id",
			)

		runner.check(
			len(alg_hits) == 2,
			"top_k larger than partition size is clamped to available records",
		)

		print_subheader("E) Mutation pool good/bad and balanced retrieval")
		mut_good_1 = MutationExperienceRecord(
			leader_algorithm_description=(
				"Leader: restart on fixed interval with weak noise estimate."
			),
			member_algorithm_description=(
				"Member: adaptive restart threshold from LBD median and volatility."
			),
			step="step_1",
			analysis=(
				"Reduced unnecessary restarts in stable phases and reacted faster to spikes."
			),
			leader_algorithm_id="mut_leader_1",
			member_algorithm_id="mut_member_good_1",
		)
		mut_good_2 = MutationExperienceRecord(
			leader_algorithm_description=(
				"Leader: smooth conflict-window trigger with static margin."
			),
			member_algorithm_description=(
				"Member: dynamic margin scaled by decision-level variance."
			),
			step="step_2",
			analysis=(
				"Improved agility without restart thrashing."
			),
			leader_algorithm_id="mut_leader_2",
			member_algorithm_id="mut_member_good_2",
		)
		mut_bad_1 = MutationExperienceRecord(
			leader_algorithm_description=(
				"Leader: conflict/LBD mixed trigger."
			),
			member_algorithm_description=(
				"Member: adds three hard gates before any restart."
			),
			step="step_3",
			analysis=(
				"Over-gating delayed recovery and regressed performance."
			),
			leader_algorithm_id="mut_leader_3",
			member_algorithm_id="mut_member_bad_1",
		)
		mut_bad_2 = MutationExperienceRecord(
			leader_algorithm_description=(
				"Leader: conservative restart fallback."
			),
			member_algorithm_description=(
				"Member: sparse fallback plus long cooldown chain."
			),
			step="step_4",
			analysis=(
				"Cooldowns blocked needed restarts in instability bursts."
			),
			leader_algorithm_id="mut_leader_4",
			member_algorithm_id="mut_member_bad_2",
		)

		mg1 = manager.persist("mutation", mut_good_1, OutcomeLabel.GOOD)
		manager.persist("mutation", mut_good_2, OutcomeLabel.GOOD)
		manager.persist("mutation", mut_bad_1, OutcomeLabel.BAD)
		manager.persist("mutation", mut_bad_2, OutcomeLabel.BAD)

		mut_query = mut_good_1.leader_algorithm_description
		mut_hits_bal = manager.retrieve(
			pool_name="mutation",
			query_text=mut_query,
			top_k=4,
			outcome=None,
			balanced=True,
		)
		print_hits("Mutation balanced retrieval", mut_hits_bal)

		outcomes_bal = {h.outcome for h in mut_hits_bal}
		runner.check(
			OutcomeLabel.GOOD in outcomes_bal and OutcomeLabel.BAD in outcomes_bal,
			"Balanced mutation retrieval includes both good and bad outcomes",
		)

		mut_hits_good_only = manager.retrieve(
			pool_name="mutation",
			query_text=mut_query,
			top_k=3,
			outcome=OutcomeLabel.GOOD,
		)
		print_hits("Mutation GOOD-only retrieval", mut_hits_good_only)
		runner.check(
			all(h.outcome == OutcomeLabel.GOOD for h in mut_hits_good_only),
			"Mutation GOOD-only filter returns only good records",
		)

		if mut_hits_good_only:
			runner.check(
				mut_hits_good_only[0].record_id == mg1.record_id,
				"Identical mutation leader description ranks expected good record first",
			)
			runner.check(
				mut_hits_good_only[0].payload.leader_algorithm_id == "mut_leader_1"
				and mut_hits_good_only[0].payload.member_algorithm_id == "mut_member_good_1",
				"Mutation retrieval preserves leader/member IDs",
			)

		print_subheader("F) Combination pool query format sensitivity")
		comb_target = CombinationExperienceRecord(
			parent_alg1_description=(
				"Parent A: fast-restart bias when LBD spikes quickly."
			),
			parent_alg2_description=(
				"Parent B: conservative restart when clause quality remains stable."
			),
			new_algorithm_description=(
				"Offspring: two-regime switch with spike mode and stable mode."
			),
			analysis=(
				"Improved by combining responsiveness with stability-aware patience."
			),
			parent_alg1_id="comb_parent_A_1",
			parent_alg2_id="comb_parent_B_1",
			new_algorithm_id="comb_new_1",
		)
		comb_distractor = CombinationExperienceRecord(
			parent_alg1_description=(
				"Parent A: conflict-window trigger with smoothing."
			),
			parent_alg2_description=(
				"Parent B: independent trail-depth trigger."
			),
			new_algorithm_description=(
				"Offspring: OR-combined trigger chain."
			),
			analysis=(
				"Over-triggering amplified both parent weaknesses."
			),
			parent_alg1_id="comb_parent_A_2",
			parent_alg2_id="comb_parent_B_2",
			new_algorithm_id="comb_new_2",
		)

		ct = manager.persist("combination", comb_target, OutcomeLabel.GOOD)
		manager.persist("combination", comb_distractor, OutcomeLabel.BAD)

		correct_combo_query = (
			f"Parent Algorithm 1: {comb_target.parent_alg1_description}\n"
			f"Parent Algorithm 2: {comb_target.parent_alg2_description}"
		)
		degraded_combo_query = (
			"Combine one aggressive and one conservative parent heuristic "
			"while preventing over-triggered restarts."
		)

		hits_correct = manager.retrieve(
			pool_name="combination",
			query_text=correct_combo_query,
			top_k=4,
			outcome=None,
			balanced=False,
		)
		hits_degraded = manager.retrieve(
			pool_name="combination",
			query_text=degraded_combo_query,
			top_k=4,
			outcome=None,
			balanced=False,
		)

		print_hits("Combination retrieval with exact parent-pair format", hits_correct)
		print_hits("Combination retrieval with degraded generic query", hits_degraded)

		runner.check(
			any(h.record_id == ct.record_id for h in hits_correct),
			"Correctly formatted combination query retrieves target record",
		)
		for h in hits_correct:
			if h.record_id == ct.record_id:
				runner.check(
					h.payload.parent_alg1_id == "comb_parent_A_1"
					and h.payload.parent_alg2_id == "comb_parent_B_1"
					and h.payload.new_algorithm_id == "comb_new_1",
					"Combination retrieval preserves parent/new algorithm IDs",
				)
				break

		def _rank_of(record_id: str, hits: List) -> int | None:
			for i, h in enumerate(hits, 1):
				if h.record_id == record_id:
					return i
			return None

		correct_rank = _rank_of(ct.record_id, hits_correct)
		degraded_rank = _rank_of(ct.record_id, hits_degraded)
		print(f"Target rank (correct query): {correct_rank}")
		print(f"Target rank (degraded query): {degraded_rank}")

		# Relative robustness check: exact format should not perform worse when both ranks exist.
		if correct_rank is not None and degraded_rank is not None:
			runner.check(
				correct_rank <= degraded_rank,
				"Exact parent-pair query ranks target at least as high as degraded query",
			)
		else:
			runner.check(
				correct_rank is not None,
				"Exact parent-pair query still finds target even if degraded query misses",
			)

		print_subheader("G) update() placeholder behavior")
		update_result = manager.update(event_type="no-op", payload={"k": 1})
		runner.check(update_result is None, "update() returns None (no-op by design)")

	print_header("Comprehensive Test Summary")
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
