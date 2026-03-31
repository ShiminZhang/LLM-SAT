#include "decide.h"
#include "inlineframes.h"
#include "inlineheap.h"
#include "inlinequeue.h"
#include "print.h"

#include <inttypes.h>

static unsigned last_enqueued_unassigned_variable (kissat *solver) {
  assert (solver->unassigned);
  const links *const links = solver->links;
  const value *const values = solver->values;
  unsigned res = solver->queue.search.idx;
  if (values[LIT (res)]) {
    do {
      res = links[res].prev;
      assert (!DISCONNECTED (res));
    } while (values[LIT (res)]);
    kissat_update_queue (solver, links, res);
  }
#ifdef LOGGING
  const unsigned stamp = links[res].stamp;
  LOG ("last enqueued unassigned %s stamp %u", LOGVAR (res), stamp);
#endif
#ifdef CHECK_QUEUE
  for (unsigned i = links[res].next; !DISCONNECTED (i); i = links[i].next)
    assert (VALUE (LIT (i)));
#endif
  return res;
}

static unsigned largest_score_unassigned_variable (kissat *solver,heap* scores) {
  unsigned res = kissat_max_heap (scores);
  const value *const values = solver->values;
  while (values[LIT (res)]) {
    kissat_pop_max_heap (solver, scores);
    res = kissat_max_heap (scores);
  }

  // MAB
  if(solver->mab) {
    solver->mab_decisions++;
    if(!solver->mab_chosen[res]){
      solver->mab_chosen_tot++;
      solver->mab_chosen[res] = 1;
    }
  }
#if defined(LOGGING) || defined(CHECK_HEAP)
  const double score = kissat_get_heap_score (scores, res);
#endif
  LOG ("largest score unassigned %s score %g", LOGVAR (res), score);
#ifdef CHECK_HEAP
  for (all_variables (idx)) {
    if (!ACTIVE (idx))
      continue;
    if (VALUE (LIT (idx)))
      continue;
    const double idx_score = kissat_get_heap_score (scores, idx);
    assert (score >= idx_score);
  }
#endif
  return res;
}

void kissat_start_random_sequence (kissat *solver) {
  if (!GET_OPTION (randec))
    return;

  if (solver->stable && !GET_OPTION (randecstable))
    return;

  if (!solver->stable && !GET_OPTION (randecfocused))
    return;

  if (solver->randec)
    kissat_very_verbose (solver,
                         "continuing random decision sequence "
                         "at %s conflicts",
                         FORMAT_COUNT (CONFLICTS));
  else {
    INC (random_sequences);
    const uint64_t count = solver->statistics.random_sequences;
    const unsigned length = GET_OPTION (randeclength) * LOGN (count);
    kissat_very_verbose (solver,
                         "starting random decision sequence "
                         "at %s conflicts for %s conflicts",
                         FORMAT_COUNT (CONFLICTS), FORMAT_COUNT (length));
    solver->randec = length;

    UPDATE_CONFLICT_LIMIT (randec, random_sequences, LOGN, false);
  }
}

static unsigned next_random_decision (kissat *solver) {
  if (!VARS)
    return INVALID_IDX;

  if (solver->warming)
    return INVALID_IDX;

  if (!GET_OPTION (randec))
    return INVALID_IDX;

  if (solver->stable && !GET_OPTION (randecstable))
    return INVALID_IDX;

  if (!solver->stable && !GET_OPTION (randecfocused))
    return INVALID_IDX;

  if (!solver->randec) {
    assert (solver->level);
    if (solver->level > 1)
      return INVALID_IDX;

    uint64_t conflicts = CONFLICTS;
    limits *limits = &solver->limits;
    if (conflicts < limits->randec.conflicts)
      return INVALID_IDX;

    kissat_start_random_sequence (solver);
  }

  for (;;) {
    unsigned idx = kissat_next_random32 (&solver->random) % VARS;
    if (!ACTIVE (idx))
      continue;
    unsigned lit = LIT (idx);
    if (solver->values[lit])
      continue;
    return idx;
  }
}

unsigned kissat_next_decision_variable (kissat *solver) {
#ifdef LOGGING
  const char *type = 0;
#endif
  unsigned res = next_random_decision (solver);
  if (res == INVALID_IDX) {
    if (solver->stable) {
#ifdef LOGGING
      type = "maximum score";
#endif
      heap* scores = kissat_get_scores (solver);
      res = largest_score_unassigned_variable (solver,scores);
      INC (score_decisions);
    } else {
#ifdef LOGGING
      type = "dequeued";
#endif
      res = last_enqueued_unassigned_variable (solver);
      INC (queue_decisions);
    }
  } else {
#ifdef LOGGING
    type = "random";
#endif
    INC (random_decisions);
  }
  LOG ("next %s decision %s", type, LOGVAR (res));
  return res;
}

int kissat_decide_phase(kissat *solver, unsigned idx) {
    // STEP 1: Initialize phase momentum array
    // We use static variables to persist state across decisions without modifying the kissat struct.
    static double *clause_phase_momentum = 0;
    static unsigned allocated_vars = 0;
    static uint64_t last_decay_conflicts = 0;
    static uint64_t last_update_conflicts = 0;
    static unsigned last_redundant = 0;

    // Reset state if a new solver instance or restart from zero occurs
    if (CONFLICTS == 0 && (last_update_conflicts != 0 || last_redundant != 0)) {
        last_decay_conflicts = 0;
        last_update_conflicts = 0;
        last_redundant = 0;
        if (clause_phase_momentum) {
            for (unsigned i = 0; i < allocated_vars; i++) {
                clause_phase_momentum[i] = 0.0;
            }
        }
    }

    // Dynamic allocation to support variable additions
    if (!clause_phase_momentum || VARS > allocated_vars) {
        unsigned new_allocated = VARS + 1000;
        double *new_momentum = kissat_calloc(solver, new_allocated, sizeof(double));
        if (clause_phase_momentum) {
            for (unsigned i = 0; i < allocated_vars; i++) {
                new_momentum[i] = clause_phase_momentum[i];
            }
            kissat_free(solver, clause_phase_momentum, allocated_vars * sizeof(double));
        }
        clause_phase_momentum = new_momentum;
        allocated_vars = new_allocated;
    }

    // STEP 2: Update momentum from newly learned clauses
    // To maintain performance, we batch the O(N) clause scan every 100 conflicts.
    if (CONFLICTS >= last_update_conflicts + 100) {
        unsigned current_redundant = GET(clauses_redundant);
        
        // Handle clause database reduction (deletion of learned clauses)
        if (current_redundant < last_redundant) {
            last_redundant = current_redundant;
        }
        
        unsigned count = 0;
        for (all_clauses(c)) {
            if (c->redundant) {
                count++;
                // Identify clauses added since our last scan
                if (count > last_redundant) {
                    unsigned lbd = c->glue;
                    if (lbd == 0) lbd = 1; // Safeguard against division by zero
                    
                    for (all_literals_in_clause(lit, c)) {
                        unsigned v = IDX(lit);
                        if (v < allocated_vars) {
                            bool is_pos = (lit == LIT(v));
                            clause_phase_momentum[v] += (is_pos ? 1.0 : -1.0) * (10.0 / (double)lbd);
                        }
                    }
                }
            }
        }
        last_redundant = count;
        last_update_conflicts = CONFLICTS;
    }

    // STEP 3: Decay momentum globally every 10,000 conflicts
    if (CONFLICTS >= last_decay_conflicts + 10000) {
        for (unsigned i = 0; i < allocated_vars; i++) {
            clause_phase_momentum[i] *= 0.95;
        }
        last_decay_conflicts = CONFLICTS;
    }

    // STEP 4: Evaluate momentum
    double M = clause_phase_momentum[idx];
    if (M > 1.5 || M < -1.5) { // Equivalent to fabs(M) > 1.5 without requiring <math.h>
        if (solver->stable) {
            // STABLE mode: bias towards satisfying high-quality learned clauses (M > 0)
            return (M > 0.0) ? 1 : -1;
        } else {
            // FOCUSED mode: falsify high-quality clauses to rapidly trigger conflicts (M < 0)
            return (M < 0.0) ? 1 : -1;
        }
    }

    // STEP 5: Dual-tier mode-aware fallback for weak momentum (abs(M) <= 1.5)
    bool force = GET_OPTION(forcephase);
    
    value *target = 0;
    if (!force && GET_OPTION(target)) {
        target = solver->phases.target + idx;
    }

    value *saved = 0;
    if (!force && GET_OPTION(phasesaving)) {
        saved = solver->phases.saved + idx;
    }

    value res = 0;
    if (solver->stable) {
        // STABLE fallback: target phase
        if (target && *target) {
            res = *target;
            INC(target_decisions);
        }
    } else {
        // FOCUSED fallback: saved phase
        if (saved && *saved) {
            res = *saved;
            INC(saved_decisions);
        }
    }

    // Ultimate fallback if preferred mode-aware phase is uninitialized
    if (!res) {
        res = INITIAL_PHASE;
        INC(initial_decisions);
    }

    return res < 0 ? -1 : 1;
}

void kissat_decide (kissat *solver) {
  START (decide);
  assert (solver->unassigned);
  if (solver->warming)
    INC (warming_decisions);
  else {
    INC (decisions);
    if (solver->stable)
      INC (stable_decisions);
    else
      INC (focused_decisions);
  }
  solver->level++;
  assert (solver->level != INVALID_LEVEL);
  const unsigned idx = kissat_next_decision_variable (solver);
  const value value = kissat_decide_phase (solver, idx);
  unsigned lit = LIT (idx);
  if (value < 0)
    lit = NOT (lit);
  kissat_push_frame (solver, lit);
  assert (solver->level < SIZE_STACK (solver->frames));
  LOG ("decide literal %s", LOGLIT (lit));
  kissat_assign_decision (solver, lit);
  STOP (decide);
}

void kissat_internal_assume (kissat *solver, unsigned lit) {
  assert (solver->unassigned);
  assert (!VALUE (lit));
  solver->level++;
  assert (solver->level != INVALID_LEVEL);
  kissat_push_frame (solver, lit);
  assert (solver->level < SIZE_STACK (solver->frames));
  LOG ("assuming literal %s", LOGLIT (lit));
  kissat_assign_decision (solver, lit);
}
