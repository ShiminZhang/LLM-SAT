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
    static int *phase_focused = NULL;
    static int *phase_stable = NULL;
    static double *phase_score_0 = NULL; // Scores for False phase
    static double *phase_score_1 = NULL; // Scores for True phase
    static unsigned alloc_vars = 0;
    static uint64_t last_restarts = 0;

    // Step 1 & 2 Setup: Maintain static arrays since we cannot invent solver fields
    if (alloc_vars <= idx) {
        unsigned new_alloc = solver->vars;
        if (new_alloc <= idx) new_alloc = idx + 1;
        new_alloc = new_alloc + (new_alloc >> 2) + 1024; // 1.25x + 1024 margin

        int *new_focused = (int*) kissat_calloc(solver, new_alloc, sizeof(int));
        int *new_stable = (int*) kissat_calloc(solver, new_alloc, sizeof(int));
        double *new_score_0 = (double*) kissat_calloc(solver, new_alloc, sizeof(double));
        double *new_score_1 = (double*) kissat_calloc(solver, new_alloc, sizeof(double));

        if (alloc_vars > 0) {
            for (unsigned i = 0; i < alloc_vars; i++) {
                new_focused[i] = phase_focused[i];
                new_stable[i] = phase_stable[i];
                new_score_0[i] = phase_score_0[i];
                new_score_1[i] = phase_score_1[i];
            }
            kissat_free(solver, phase_focused, alloc_vars * sizeof(int));
            kissat_free(solver, phase_stable, alloc_vars * sizeof(int));
            kissat_free(solver, phase_score_0, alloc_vars * sizeof(double));
            kissat_free(solver, phase_score_1, alloc_vars * sizeof(double));
        }

        phase_focused = new_focused;
        phase_stable = new_stable;
        phase_score_0 = new_score_0;
        phase_score_1 = new_score_1;
        alloc_vars = new_alloc;
    }

    // Step 2 (Decay): Decay all scores by a factor of 0.95 at every restart
    if (last_restarts < solver->statistics.restarts) {
        uint64_t diff = solver->statistics.restarts - last_restarts;
        double decay_factor = pow(0.95, (double)diff);
        for (unsigned i = 0; i < alloc_vars; i++) {
            phase_score_0[i] *= decay_factor;
            phase_score_1[i] *= decay_factor;
        }
        last_restarts = solver->statistics.restarts;
    }

    bool is_stable = solver->stable;
    int P = 0;

    bool force = GET_OPTION (forcephase);
    value *saved = 0;
    if (!force && GET_OPTION (phasesaving)) {
        saved = solver->phases.saved + idx;
    }

    // Step 1 (Simulation) & Step 3: Identify the current search mode and retrieve baseline candidate phase
    // We update our mode-specific arrays from Kissat's global saved phase to maintain independence
    if (!is_stable) {
        if (saved && *saved) phase_focused[idx] = *saved;
        P = phase_focused[idx];
    } else {
        if (saved && *saved) phase_stable[idx] = *saved;
        P = phase_stable[idx];
    }

    // Fallback logic if the mode-specific array doesn't have a saved phase yet
    if (P == 0) {
        value *target = 0;
        if (!force && GET_OPTION (target) && (solver->stable || GET_OPTION (target) > 1))
            target = solver->phases.target + idx;
        
        value res = 0;
        if (!solver->stable) {
            switch ((solver->statistics.switched >> 1) & 7) {
                case 1: res = INITIAL_PHASE; break;
                case 3: res = -INITIAL_PHASE; break;
            }
        }

        if (!res && target && *target) {
            res = *target;
            LOG ("%s uses target decision phase %d", LOGVAR (idx), (int) res);
            INC (target_decisions);
        }
        if (!res && saved && *saved) {
            res = *saved;
            LOG ("%s uses saved decision phase %d", LOGVAR (idx), (int) res);
            INC (saved_decisions);
        }
        if (!res) {
            res = INITIAL_PHASE;
            LOG ("%s uses initial decision phase %d", LOGVAR (idx), (int) res);
            INC (initial_decisions);
        }
        P = (res < 0) ? -1 : 1;
    } else {
        LOG ("%s uses mode-specific saved phase %d", LOGVAR (idx), P);
        INC (saved_decisions);
        P = (P < 0) ? -1 : 1;
    }

    // Step 4: Evaluate the historical LBD quality of the candidate phase
    double score_P = (P > 0) ? phase_score_1[idx] : phase_score_0[idx];
    double score_not_P = (P > 0) ? phase_score_0[idx] : phase_score_1[idx];

    int final_phase = P;

    if (!is_stable) {
        // Focused mode: encouraging agile, localized exploration
        if (score_not_P > 1.2 * score_P && score_not_P > 1.0) {
            final_phase = -P;
            LOG ("%s overrides focused baseline %d to %d", LOGVAR (idx), P, final_phase);
        }
    } else {
        // Stable mode: demand higher confidence to preserve search continuity
        if (score_not_P > 2.0 * score_P && score_not_P > 3.0) {
            final_phase = -P;
            LOG ("%s overrides stable baseline %d to %d", LOGVAR (idx), P, final_phase);
        }
    }

    // Step 5: Return the decided phase
    return final_phase;
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
