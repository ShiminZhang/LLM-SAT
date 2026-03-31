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

int kissat_decide_phase (kissat *solver, unsigned idx) {
    // We use static variables to maintain state across calls without modifying the solver struct.
    // As per requirements, we use static int instead of static unsigned.
    static int allocated_vars = 0;
    static int *phase_focused = NULL;
    static int *phase_stable = NULL;
    static double *phase_score_0 = NULL; // Scores for phase -1 (False)
    static double *phase_score_1 = NULL; // Scores for phase 1 (True)
    static int last_restarts = 0;

    // Dynamically allocate or resize the tracking arrays if the number of variables increases
    if (allocated_vars < solver->vars + 1) {
        int new_size = solver->vars + 1000;
        size_t old_bytes_int = allocated_vars * sizeof(int);
        size_t new_bytes_int = new_size * sizeof(int);
        size_t old_bytes_double = allocated_vars * sizeof(double);
        size_t new_bytes_double = new_size * sizeof(double);

        phase_focused = kissat_realloc(solver, phase_focused, old_bytes_int, new_bytes_int);
        phase_stable = kissat_realloc(solver, phase_stable, old_bytes_int, new_bytes_int);
        phase_score_0 = kissat_realloc(solver, phase_score_0, old_bytes_double, new_bytes_double);
        phase_score_1 = kissat_realloc(solver, phase_score_1, old_bytes_double, new_bytes_double);
        
        for (int i = allocated_vars; i < new_size; i++) {
            phase_focused[i] = 0;
            phase_stable[i] = 0;
            phase_score_0[i] = 0.0;
            phase_score_1[i] = 0.0;
        }
        allocated_vars = new_size;
    }

    // Step 2 (Partial): Decay all scores by a factor of 0.95 at every restart.
    int current_restarts = solver->statistics.restarts;
    if (current_restarts > last_restarts) {
        double total_decay = 1.0;
        int diff = current_restarts - last_restarts;
        if (diff > 100) diff = 100; // Cap to prevent excessive loops
        for (int i = 0; i < diff; i++) {
            total_decay *= 0.95;
        }
        
        for (int i = 0; i < allocated_vars; i++) {
            phase_score_0[i] *= total_decay;
            phase_score_1[i] *= total_decay;
        }
        last_restarts = current_restarts;
    }

    bool force = GET_OPTION (forcephase);
    value *saved = 0;
    if (!force && GET_OPTION (phasesaving)) {
        saved = solver->phases.saved + idx;
    }

    value *target_ptr = 0;
    if (!force && GET_OPTION (target)) {
        if (solver->stable || GET_OPTION (target) > 1) {
            target_ptr = solver->phases.target + idx;
        }
    }

    // Step 1 Approximation: Synchronize mode-specific phase with Kissat's globally saved phase.
    // Since we cannot hook directly into backtracking from this function, we update the 
    // corresponding array lazily based on the current search mode.
    if (saved && *saved != 0) {
        if (solver->stable) {
            phase_stable[idx] = *saved;
        } else {
            phase_focused[idx] = *saved;
        }
    }

    // Step 3: Identify current search mode and retrieve baseline candidate phase P
    int P = 0;
    if (solver->stable) {
        P = phase_stable[idx];
    } else {
        P = phase_focused[idx];
    }

    // Fallback if P is uninitialized
    if (P == 0) {
        if (saved && *saved != 0) {
            P = *saved;
        } else {
            P = INITIAL_PHASE;
        }
    }

    // Step 4: Evaluate the historical LBD quality of the candidate phase
    double score_P = (P > 0) ? phase_score_1[idx] : phase_score_0[idx];
    double score_not_P = (P > 0) ? phase_score_0[idx] : phase_score_1[idx];

    int selected_phase = P;
    bool overridden = false;

    // If opposite phase has significantly better track record and meets confidence threshold
    if (score_not_P > 1.5 * score_P && score_not_P > 2.0) {
        selected_phase = -P;
        overridden = true;
    }

    // Step 5: If not overridden, evaluate absolute confidence of P
    if (!overridden) {
        if (score_P < 0.5) {
            // Poor recent performance / insufficient data -> fallback to globally saved target phase
            if (target_ptr && *target_ptr != 0) {
                selected_phase = *target_ptr;
            }
        }
    }

    // Final safety check to ensure phase is valid
    if (selected_phase == 0) {
        selected_phase = INITIAL_PHASE;
    }

    // Update Kissat statistics exactly like the baseline
    if (target_ptr && selected_phase == *target_ptr) {
        INC (target_decisions);
    } else if (saved && selected_phase == *saved) {
        INC (saved_decisions);
    } else {
        INC (initial_decisions);
    }

    LOG ("%s uses decision phase %d", LOGVAR (idx), (int) selected_phase);

    return selected_phase < 0 ? -1 : 1;
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
