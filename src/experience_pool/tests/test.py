"""
End-to-end demo for ExperiencePoolManager.

Run from repo root with:
  python src/experience_pool/tests/test.py
(or run interactively)
"""

import sys
from pathlib import Path

# Add src/ to sys.path so we can import from experience_pool
src_dir = str(Path(__file__).parent.parent.parent.resolve())
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from experience_pool.manager import ExperiencePoolManager
from experience_pool.types import (
    AlgorithmExperienceRecord,
    MutationExperienceRecord,
    CombinationExperienceRecord,
    OutcomeLabel,
)


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    # -----------------------------------------------------------------------------
    # 0) Create manager (runtime singleton: embedding model + FAISS loaded once)
    # -----------------------------------------------------------------------------
    print_header("0) Create manager and verify singleton runtime")
    manager = ExperiencePoolManager()
    manager2 = ExperiencePoolManager()

    

    # This should be True because SharedRuntime is singleton per process.
    print("runtime shared across managers:", manager.runtime is manager2.runtime)

    # -----------------------------------------------------------------------------
    # 1) ALGORITHM POOL (bad-only)
    #    record schema: (algorithm_description, analysis)
    # -----------------------------------------------------------------------------
    print_header("1) Algorithm pool: persist bad experiences, then retrieve")

    bad_alg_1 = AlgorithmExperienceRecord(
        algorithm_description=(
            "Restart every fixed 8 conflicts without considering LBD trend, "
            "decision-level variance, or trail volatility."
        ),
        analysis=(
            "Overly rigid schedule causes excessive restart churn and prevents "
            "deep useful propagation on hard UNSAT regions."
        ),
        algorithm_id="alg_bad_fixed8",
    )
    bad_alg_2 = AlgorithmExperienceRecord(
        algorithm_description=(
            "Delay all restarts until conflict count > 1e7, then restart aggressively."
        ),
        analysis=(
            "Extreme late restart policy can trap search in unproductive basin; "
            "recovery is too delayed for mixed SAT/UNSAT distributions."
        ),
        algorithm_id="alg_bad_late_restart",
    )

    r1 = manager.persist("algorithm", bad_alg_1, OutcomeLabel.BAD)
    r2 = manager.persist("algorithm", bad_alg_2, OutcomeLabel.BAD)
    r1_dup = manager.persist("algorithm", bad_alg_1, OutcomeLabel.BAD)  # dedupe demo

    print("insert 1:", r1) #PersistReceipt(record_id='0939163bfb3acdc0fe7e5780cfc6e6f6b76e1575ad5a754c0999c97028c19d9d', pool_name='combination', outcome=<OutcomeLabel.GOOD: 'good'>, created=True, partition_size=1)
    print("insert 2:", r2)
    print("duplicate insert (created should be False):", r1_dup)

    query_for_leader_gen = (
        "Need a restart heuristic for kissat_restarting that avoids rigid or "
        "extreme restart timing and adapts to search state."
    )
    alg_hits = manager.retrieve(
        pool_name="algorithm",
        query_text=query_for_leader_gen,
        top_k=3,
        outcome=OutcomeLabel.BAD,   # algorithm pool supports bad only
    )
    print("\nRetrieved algorithm bad examples:")
    for i, hit in enumerate(alg_hits, 1):
        print(f"[{i}] score={hit.score:.4f}, outcome={hit.outcome.value}")
        print("    desc:", hit.payload.algorithm_description[:120], "...")

    # -----------------------------------------------------------------------------
    # 2) MUTATION POOL (good + bad)
    #    record schema: (leader_description, member_description, analysis)
    # -----------------------------------------------------------------------------
    print_header("2) Mutation pool: persist good/bad, retrieve balanced and filtered")

    mut_good = MutationExperienceRecord(
        leader_algorithm_description=(
            "Leader: restart on fixed interval with weak noise estimate."
        ),
        member_algorithm_description=(
            "Member: restart threshold adapts using moving LBD median and "
            "decision-level volatility."
        ),
        step="step_1",
        analysis=(
            "Member outperformed leader by lowering unnecessary restarts in stable "
            "phases and reacting faster during instability."
        ),
        leader_algorithm_id="leader_demo_1",
        member_algorithm_id="member_demo_good_1",
    )
    mut_bad = MutationExperienceRecord(
        leader_algorithm_description=(
            "Leader: restart conditioned on conflict/LBD mixed trigger."
        ),
        member_algorithm_description=(
            "Member: adds three independent hard gates before any restart."
        ),
        step="step_2",
        analysis=(
            "Extra gating delayed critical restarts; performance regressed due to "
            "late recovery from poor branching trajectories."
        ),
        leader_algorithm_id="leader_demo_2",
        member_algorithm_id="member_demo_bad_1",
    )

    pmg = manager.persist("mutation", mut_good, OutcomeLabel.GOOD)
    pmb = manager.persist("mutation", mut_bad, OutcomeLabel.BAD)
    print("mutation good insert:", pmg)
    print("mutation bad insert:", pmb)

    mutation_query = "Mutate leader restart strategy to be adaptive to volatility."
    mut_balanced = manager.retrieve(
        pool_name="mutation",
        query_text=mutation_query,
        top_k=4,
        outcome=None,        # search both good+bad
        balanced=True,       # try to return both sides
    )
    print("\nRetrieved mutation balanced examples:")
    for i, hit in enumerate(mut_balanced, 1):
        print(f"[{i}] score={hit.score:.4f}, outcome={hit.outcome.value}")
        print("    leader:", hit.payload.leader_algorithm_description[:80], "...")

    mut_good_only = manager.retrieve(
        pool_name="mutation",
        query_text=mutation_query,
        top_k=2,
        outcome=OutcomeLabel.GOOD,
    )
    print("\nRetrieved mutation GOOD-only examples:")
    for i, hit in enumerate(mut_good_only, 1):
        print(f"[{i}] score={hit.score:.4f}, outcome={hit.outcome.value}")

    # -----------------------------------------------------------------------------
    # 3) COMBINATION POOL (good + bad)
    #    record schema: (parent1_description, parent2_description, new_description, analysis)
    # -----------------------------------------------------------------------------
    print_header("3) Combination pool: persist good/bad, retrieve by parent-pair context")

    comb_good = CombinationExperienceRecord(
        parent_alg1_description=(
            "Parent A: fast-restart bias when LBD spikes quickly."
        ),
        parent_alg2_description=(
            "Parent B: conservative restart when clause quality remains stable."
        ),
        new_algorithm_description=(
            "Offspring: two-regime switch; aggressive mode on spike, conservative mode "
            "during stable low-variance intervals."
        ),
        analysis=(
            "Offspring improved by combining spike responsiveness with stability-aware "
            "patience, reducing both thrashing and stagnation."
        ),
        parent_alg1_id="parent_A_demo",
        parent_alg2_id="parent_B_demo",
        new_algorithm_id="offspring_demo_good",
    )
    comb_bad = CombinationExperienceRecord(
        parent_alg1_description="Parent A: conflict-window trigger with smoothing.",
        parent_alg2_description="Parent B: independent trail-depth trigger.",
        new_algorithm_description=(
            "Offspring: applies both triggers independently with OR logic."
        ),
        analysis=(
            "Trigger conflict caused excessive restart frequency; simple OR merge "
            "amplified parent weaknesses rather than combining strengths."
        ),
        parent_alg1_id="parent_A_bad_demo",
        parent_alg2_id="parent_B_bad_demo",
        new_algorithm_id="offspring_demo_bad",
    )

    pcg = manager.persist("combination", comb_good, OutcomeLabel.GOOD)
    pcb = manager.persist("combination", comb_bad, OutcomeLabel.BAD)
    print("combination good insert:", pcg)
    print("combination bad insert:", pcb)

    combo_query = (
        "Given one aggressive and one conservative parent heuristic, suggest "
        "combination patterns that avoid restart over-triggering."
    )
    comb_hits = manager.retrieve(
        pool_name="combination",
        query_text=combo_query,
        top_k=4,
        outcome=None,       # both partitions
        balanced=True,
    )
    print("\nRetrieved combination examples:")
    for i, hit in enumerate(comb_hits, 1):
        print(f"[{i}] score={hit.score:.4f}, outcome={hit.outcome.value}")
        print("    offspring:", hit.payload.new_algorithm_description[:90], "...")

    # -----------------------------------------------------------------------------
    # 4) MUTATION update() from leaders/members directory
    # -----------------------------------------------------------------------------
    print_header("4) Mutation update() from solver directory")
    update_input_dir = "/local-scratch1/jla1045/LLM-SAT/solvers/mike_exp4_iter0_dummy"
    update_summary = manager.update(
        "mutation",
        input_dir=update_input_dir,
        threshold=0.10,
    )
    print("update input:", update_input_dir)
    print("update threshold:", 0.10)
    print("update summary:", update_summary)

    alg_hits = manager.retrieve(
        pool_name="mutation",
        query_text="Leader-L1: Step 1: Increase restart interval. Step 2: Boost glue-based retention.",
        top_k=20,
        outcome=OutcomeLabel.GOOD,   # algorithm pool supports bad only
    )
    print("\nRetrieved mutation GOOD examples:")
    for i, hit in enumerate(alg_hits, 1):
        print(f"[{i}] score={hit.score:.4f}, outcome={hit.outcome.value}")
        print(
            "    leader_id/member_id:",
            f"{hit.payload.leader_algorithm_id}/{hit.payload.member_algorithm_id}",
        )
        print("    desc:", hit.payload.leader_algorithm_description[:120], "...")

    # -----------------------------------------------------------------------------
    # 5) COMBINATION update() + list-query retrieve (N choose 2 pair search)
    # -----------------------------------------------------------------------------
    print_header("5) Combination update() + list-query retrieve")
    combined_input_dir = "/local-scratch1/jla1045/LLM-SAT/solvers/mike_exp4_gen1_dummy"
    parent_source_dir = "/local-scratch1/jla1045/LLM-SAT/solvers/mike_exp4_iter0_dummy"

    comb_update_summary = manager.update(
        "combination",
        combined_dir=combined_input_dir,
        parent_source_dir=parent_source_dir,
        threshold=0.10,
    )
    print("combined input:", combined_input_dir)
    print("parent source:", parent_source_dir)
    print("threshold:", 0.10)
    print("combination update summary:", comb_update_summary)

    # For combination retrieve, pass a list of potential leaders.
    # The pool will build all unique unordered pairs internally (N choose 2).
    potential_leaders = [
        "Leader-L1: Step 1: Increase restart interval. Step 2: Boost glue-based retention.",
        "Leader-L2: Step 1: Favor binary clauses in decisions. Step 2: Earlier rephasing.",
        "just some dummy value",
    ]

    comb_good_hits = manager.retrieve(
        pool_name="combination",
        query_text=potential_leaders,
        top_k=3,
        outcome=OutcomeLabel.GOOD,
    )

    comb_good_hits = manager.retrieve(
        pool_name="combination",
        query_text=potential_leaders,
        top_k=3,
        outcome=OutcomeLabel.BAD,
    )
    print("\nRetrieved combination GOOD examples (list query):")
    for i, hit in enumerate(comb_good_hits, 1):
        print(f"[{i}] score={hit.score:.4f}, outcome={hit.outcome.value}")
        print(
            "    parent1_id/parent2_id/new_id:",
            f"{hit.payload.parent_alg1_id}/{hit.payload.parent_alg2_id}/{hit.payload.new_algorithm_id}",
        )
        print("    offspring:", hit.payload.new_algorithm_description[:120], "...")

    # -----------------------------------------------------------------------------
    # 6) Unified search_experience_pool() API demo
    # -----------------------------------------------------------------------------
    print_header("6) Unified search_experience_pool() and parsing results")

    # Example A: algorithm pool (good side explicitly disabled by 0/0)
    alg_search = manager.search_experience_pool(
        pool_name="algorithm",
        query_text="Need robust restart strategy without rigid schedule.",
        retrieve_good_k=0,
        retrieve_bad_k=3,
        sample_good_k=0,
        sample_bad_k=2,
    )
    print("========\n")
    print(alg_search)
    print("========\n")

    print("\nAlgorithm search summary:")
    print("  good.supported:", alg_search.good.supported)
    print("  good.unique size:", len(alg_search.good.unique))
    print("  bad.retrieved size:", len(alg_search.bad.retrieved))
    print("  bad.sampled size:", len(alg_search.bad.sampled))
    print("  bad.unique size:", len(alg_search.bad.unique))
    print("  all_unique size:", len(alg_search.all_unique))

    # Parse each section directly
    for i, hit in enumerate(alg_search.bad.unique, 1):
        print(f"  [algorithm bad unique #{i}] id={hit.record_id[:12]} score={hit.score:.4f}")
        print("      alg_id:", hit.payload.algorithm_id)
        print("      desc:", hit.payload.algorithm_description[:90], "...")

    # Example B: combination pool with list query (N choose 2 handled internally)
    comb_search = manager.search_experience_pool(
        pool_name="combination",
        query_text=potential_leaders,
        retrieve_good_k=3,
        retrieve_bad_k=3,
        sample_good_k=2,
        sample_bad_k=2,
    )
    
    print("========\n")
    print(comb_search)
    print("========\n")


    print("\nCombination search summary:")
    print("  good.unique size:", len(comb_search.good.unique))
    print("  bad.unique size:", len(comb_search.bad.unique))
    print("  all_unique size:", len(comb_search.all_unique))

    # Parse merged deduped output (easy for external consumers)
    print("\nParsed combined all_unique results:")
    for i, hit in enumerate(comb_search.all_unique, 1):
        print(f"[{i}] outcome={hit.outcome.value} score={hit.score:.4f}")
        print(
            "    parent1_id/parent2_id/new_id:",
            f"{hit.payload.parent_alg1_id}/{hit.payload.parent_alg2_id}/{hit.payload.new_algorithm_id}",
        )
        print("    offspring:", hit.payload.new_algorithm_description[:120], "...")



if __name__ == "__main__":
    
    main()