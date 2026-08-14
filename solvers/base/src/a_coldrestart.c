/**
 * (taomengxia) [coldrestart] 2025-04-25
 */

#include "internal.h"
#include "collect.h"
#include "logging.h"
#include "trail.h"
#include "print.h"
#include "inlineheap.h"
#include "inlinequeue.h"

bool kissat_coldrestarting (kissat *solver) {
    if (!GET_OPTION (coldrestart))
        return false;
    if (!solver->statistics.clauses_redundant)
        return false;
    if (solver->level)
        return false;
    if (CONFLICTS < solver->limits.coldrestart.conflicts)
        return false;
    return true;
}

static void kissat_coldretart_mark_redundant_as_garbage (kissat *solver, reference start_ref) {
    assert (start_ref != INVALID_REF);
    assert (start_ref <= SIZE_STACK (solver->arena));
    ward *const arena = BEGIN_STACK (solver->arena);
    clause *start = (clause *) (arena + start_ref);
    const clause *const end = (clause *) END_STACK (solver->arena);
    assert (start < end);
    while (start != end && !start->redundant)
        start = kissat_next_clause(start);
    if (start == end) {
        solver->first_reducible = INVALID_REF;
        LOG ("no reducible clause candidate left");
        return;
    }

    const reference redundant = (ward *) start - arena;
    solver->first_reducible = redundant;
    unsigned lbd_threshold = GET_OPTION (coldrestartfclbd);
    for (clause *c = start; c != end; c = kissat_next_clause (c)) {
        if (!c->redundant)
            continue;
        if (c->garbage)
            continue;
        if (c->reason)
            continue;

        if (c->glue <= lbd_threshold)
            continue;
        kissat_mark_clause_as_garbage (solver, c);
    }
}

static void kissat_coldrestart_forgetting_order (kissat *solver) {
    assert (GET_OPTION (coldrestartfo));
    kissat_section (solver, "cold-restart : forgetting order");

    heap *scores = &solver->scores;
    flags *flags = solver->flags;
    unsigned *id = (unsigned *)malloc(sizeof(unsigned) * (VARS));
    for (unsigned i = 0; i < VARS; i++) {
        id[i] = i;
    }
    for (unsigned i = 0; i < VARS; i++) {
        unsigned j = (rand() % VARS);
        unsigned tmp = id[i];
        id[i] = id[j];
        id[j] = tmp;
    }

    solver->scinc = 1.0;
    for (unsigned i = 0; i < VARS; i++) {
        unsigned idx = id[i];
        if (flags[idx].active) {
            double new_score = 1.0 * i / VARS;
            kissat_update_heap (solver, scores, idx, new_score);

            kissat_move_to_front (solver, idx);
        }
    }

    free(id);
}

static void kissat_coldrestart_forgetting_phases (kissat *solver) {
    assert (GET_OPTION (coldrestartfo));
    kissat_section (solver, "cold-restart : forgetting phases");

    for (unsigned idx = 0; idx < VARS; idx++) {
        SAVED(idx) = rand () % 2 == 0 ? -1 : 1;
    }
}

static void kissat_coldrestart_forgetting_learnt_clauses (kissat *solver) {
    assert (GET_OPTION (coldrestartfc));
    kissat_section (solver, "cold-restart : forgetting learnt clauses");

    bool compact = kissat_compacting (solver);
    reference start = compact ? 0 : solver->first_reducible;
    if (start != INVALID_REF) {
        if (kissat_flush_and_mark_reason_clauses (solver, start)) {
            kissat_coldretart_mark_redundant_as_garbage (solver, start);
            kissat_sparse_collect (solver, compact, start);
        }
    }
}

void kissat_coldrestart (kissat *solver) {
    kissat_section (solver, "cold-restart");

    START (coldrestart);
    INC (coldrestarts);

    if (GET_OPTION (coldrestartfo)) {
        kissat_coldrestart_forgetting_order (solver);
    }

    if (GET_OPTION (coldrestartfp)) {
        kissat_coldrestart_forgetting_phases (solver);
    }

    if (GET_OPTION (coldrestartfc)) {
        kissat_coldrestart_forgetting_learnt_clauses( solver);
    }

    const uint64_t delta = (solver->statistics.coldrestarts * GET_OPTION (coldrestartint));
    solver->limits.coldrestart.conflicts = CONFLICTS + delta;

    STOP (coldrestart);
}