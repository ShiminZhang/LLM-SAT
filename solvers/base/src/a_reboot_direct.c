/**
 * (taomengxia) [reboot: direct] 2025-04-16
 */

#ifdef ENABLE_REBOOT

#include "backtrack.h"
#include "collect.h"
#include "inline.h"
#include "inlineheap.h"
#include "inlinequeue.h"
#include "internal.h"
#include "print.h"
#include "search.h"

#include <string.h>

bool kissat_rebooting_direct(kissat *solver) {
    if (GET_OPTION(reboot) != 1)
        return false;

    if (CONFLICTS < (unsigned)GET_OPTION(rebootconflicts))
        return false;

    if (solver->statistics.reboots)     // already reboot
        return false;
    
    return true;
}

static void kissat_reboot_reduce_useless_redundant_clauses(kissat *solver) {
    for (all_clauses(c)) {
        if (c->garbage) {
            continue;
        }
        if (!c->redundant) {
            continue;
        }
        if (c->size <= (unsigned)GET_OPTION(rebootclausesize)) {
            continue;
        }
        kissat_mark_clause_as_garbage(solver, c);
    }

    kissat_sparse_collect(solver, false, 0);
}

static inline void kissat_reboot_reset_statistics (kissat *solver) {
    statistics *statistics = &solver->statistics;
    statistics->searches = 0;
    statistics->conflicts = 0;
}

static void kissat_reboot_scores_clear (kissat *solver, unsigneds *actives) {
    flags *all_flags = solver->flags;
    for (all_variables (idx)) {
        flags *flags = all_flags + idx;
        if (!flags->active)
            continue;

        PUSH_STACK(*actives, idx);

        flags->active = false;
        solver->active--;
        kissat_dequeue (solver, idx);
        if (kissat_heap_contains (SCORES, idx)) {
            kissat_pop_heap (solver, SCORES, idx);
        }
    }
    assert (!solver->active);
}

static void kissat_reboot_scores_rebuild (kissat *solver, unsigneds *actives) {
    const unsigned size = SIZE_STACK(*actives);
    for (unsigned idx = 0; idx < size; idx++) {
        const unsigned var_idx = PEEK_STACK(*actives, idx);
        flags *f = FLAGS (var_idx);
        assert(!f->active);

        f->active = true;
        solver->active++;
        kissat_enqueue (solver, var_idx);
        const double score = 1.0 - 1.0 / solver->active;
        kissat_update_heap (solver, &solver->scores, var_idx, score);
        if (solver->stable) {
            const unsigned lit = LIT (var_idx);
            if (!VALUE (lit))
                kissat_push_heap (solver, &solver->scores, var_idx);
        }
    }
}

static void kissat_reboot_reset_scores (kissat *solver) {
    solver->scinc = 1.0;        // init in kissat_init()

    bool stable = (GET_OPTION (stable) == 2);
    solver->stable = stable;

    // clear scores/heap and save active-flag in old_active_flags
    unsigned saved_active = solver->active;
    unsigneds actives;
    INIT_STACK (actives);
    kissat_reboot_scores_clear(solver, &actives);
    assert (SIZE_STACK(actives) == saved_active);

    // rebuild scores/heap
    kissat_reboot_scores_rebuild(solver, &actives);
    assert(saved_active == solver->active);
    RELEASE_STACK (actives);
}

static void kissat_reboot_reset_phases (kissat *solver) {
    const unsigned size = solver->size;
    memset(solver->phases.best, 0, size);
    memset(solver->phases.saved, 0, size);
    memset(solver->phases.target, 0, size);
}

int kissat_reboot_direct(kissat *solver) {
    kissat_section (solver, "reboot start");
    
    INC (reboots);
    solver->statistics.reboots_conflicts += CONFLICTS;

    // 1. backtrack to level 0
    kissat_backtrack_in_consistent_state(solver, 0);

    // 2. reduce useless redundant clauses
    kissat_reboot_reduce_useless_redundant_clauses(solver);

    // 3. reset info
    kissat_reboot_reset_statistics(solver);
    kissat_reboot_reset_scores(solver);
    kissat_reboot_reset_phases(solver);

    // 4. restart search.
    return kissat_search(solver);
}

#endif