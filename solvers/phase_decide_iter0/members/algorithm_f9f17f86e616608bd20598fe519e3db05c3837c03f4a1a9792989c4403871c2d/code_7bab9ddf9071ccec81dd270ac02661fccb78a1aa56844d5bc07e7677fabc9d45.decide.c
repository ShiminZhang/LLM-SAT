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
    bool force = GET_OPTION (forcephase);

    // Step 1 & 2: Maintain independent phase arrays and quality matrix
    // We map phase_focused to solver->phases.saved and phase_stable to solver->phases.target
    // For the quality matrix, we use static bounded arrays to strictly avoid hallucinating solver fields
    #define MAX_PHASE_VARS 2000000
    static double phase_score_0[MAX_PHASE_VARS];
    static double phase_score_1[MAX_PHASE_VARS];
    
    // Decay scores at every restart 
    static uint64_t last_restarts = 0;
    
    // Handle solver reset/re-initialization
    if (solver->statistics.restarts < last_restarts) {
        last_restarts = 0;
        unsigned limit = solver->vars < MAX_PHASE_VARS ? solver->vars : MAX_PHASE_VARS;
        for (unsigned i = 0; i < limit; i++) {
            phase_score_0[i] = 0.0;
            phase_score_1[i] = 0.0;
        }
    }
    
    // Apply time-decay based on the number of restarts elapsed
    if (last_restarts < solver->statistics.restarts) {
        unsigned diff = solver->statistics.restarts - last_restarts;
        double decay_factor = pow(0.95, diff);
        unsigned limit = solver->vars < MAX_PHASE_VARS ? solver->vars : MAX_PHASE_VARS;
        for (unsigned i = 0; i < limit; i++) {
            phase_score_0[i] *= decay_factor;
            phase_score_1[i] *= decay_factor;
        }
        last_restarts = solver->statistics.restarts;
    }

    value P_focused = 0;
    value P_stable = 0;

    // Retrieve native tracking arrays as our baseline fallback
    if (!force) {
        if (GET_OPTION (phasesaving) && solver->phases.saved) {
            P_focused = solver->phases.saved[idx];
        }
        if ((solver->stable || GET_OPTION (target) > 1) && solver->phases.target) {
            P_stable = solver->phases.target[idx];
        }
    }

    // Step 3: Identify current search mode and retrieve baseline candidate phase
    value P = 0;
    value P_alt = 0;

    if (!solver->stable) {
        P = P_focused;
        P_alt = P_stable;
    } else {
        P = P_stable;
        P_alt = P_focused;
    }

    // Baseline fallback logic for empty phase
    value switched_phase = 0;
    if (!solver->stable) {
        switch ((solver->statistics.switched >> 1) & 7) {
            case 1: switched_phase = INITIAL_PHASE; break;
            case 3: switched_phase = -INITIAL_PHASE; break;
        }
    }

    if (switched_phase) {
        P = switched_phase;
    } else {
        if (!P) P = P_alt;
        if (!P) P = INITIAL_PHASE;
    }

    if (!P_alt) P_alt = P; // Ensure P_alt is valid if missing

    value res = 0;
    int not_P = (P > 0) ? -1 : 1;
    
    double score_P = 0.0;
    double score_not_P = 0.0;
    double score_P_alt = 0.0;

    if (idx < MAX_PHASE_VARS) {
        score_P = (P > 0) ? phase_score_1[idx] : phase_score_0[idx];
        score_not_P = (not_P > 0) ? phase_score_1[idx] : phase_score_0[idx];
        score_P_alt = (P_alt > 0) ? phase_score_1[idx] : phase_score_0[idx];
    }

    // Step 4: Evaluate the historical LBD quality of the candidate phase
    if (score_not_P > 1.5 * score_P && score_not_P > 2.0) {
        res = not_P; // Override baseline and select !P
        LOG ("%s uses overridden opposite phase %d", LOGVAR (idx), (int) res);
        INC (initial_decisions);
    } else {
        // Step 5: Leverage cross-mode consensus if override conditions aren't met
        if (P != P_alt && score_P_alt > score_P) {
            res = P_alt;
            LOG ("%s uses cross-mode alt phase %d", LOGVAR (idx), (int) res);
            INC (target_decisions);
        } else {
            res = P; // Return baseline candidate phase
            if (res == P_stable && solver->stable && P_stable != 0) {
                LOG ("%s uses target decision phase %d", LOGVAR (idx), (int) res);
                INC (target_decisions);
            } else if (res == P_focused && P_focused != 0) {
                LOG ("%s uses saved decision phase %d", LOGVAR (idx), (int) res);
                INC (saved_decisions);
            } else {
                LOG ("%s uses initial decision phase %d", LOGVAR (idx), (int) res);
                INC (initial_decisions);
            }
        }
    }

    assert (res);
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
