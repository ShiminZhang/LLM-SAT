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
    // Ensure all state arrays are allocated and large enough for current VARS
    if (kissat_lpp_allocated < (int)solver->vars) {
        int old_vars = kissat_lpp_allocated;
        int new_vars = solver->vars + 1000;
        
        if (old_vars == 0) {
            kissat_lpp_false = kissat_calloc(solver, new_vars, sizeof(float));
            kissat_lpp_true = kissat_calloc(solver, new_vars, sizeof(float));
            kissat_last_seen_phase = kissat_calloc(solver, new_vars, sizeof(int));
            kissat_flip_count = kissat_calloc(solver, new_vars, sizeof(int));
            kissat_flip_window_start = kissat_calloc(solver, new_vars, sizeof(uint64_t));
        } else {
            kissat_lpp_false = kissat_realloc(solver, kissat_lpp_false, old_vars * sizeof(float), new_vars * sizeof(float));
            kissat_lpp_true = kissat_realloc(solver, kissat_lpp_true, old_vars * sizeof(float), new_vars * sizeof(float));
            kissat_last_seen_phase = kissat_realloc(solver, kissat_last_seen_phase, old_vars * sizeof(int), new_vars * sizeof(int));
            kissat_flip_count = kissat_realloc(solver, kissat_flip_count, old_vars * sizeof(int), new_vars * sizeof(int));
            kissat_flip_window_start = kissat_realloc(solver, kissat_flip_window_start, old_vars * sizeof(uint64_t), new_vars * sizeof(uint64_t));
            
            // Initialize newly allocated memory
            for (int i = old_vars; i < new_vars; i++) {
                kissat_lpp_false[i] = 0.0f;
                kissat_lpp_true[i] = 0.0f;
                kissat_last_seen_phase[i] = 0;
                kissat_flip_count[i] = 0;
                kissat_flip_window_start[i] = 0;
            }
        }
        kissat_lpp_allocated = new_vars;
    }

    // Step 2: Calculate the polarity bias delta for the target variable
    float delta = kissat_lpp_true[idx] - kissat_lpp_false[idx];

    // Step 3: Evaluate Delta against learned-confidence threshold of 2.5
    if (fabsf(delta) > 2.5f) {
        return delta > 0.0f ? 1 : -1;
    }

    // Step 4: Fall back to retrieving the standard cached phase (P_saved)
    value p_saved = 0;
    if (GET_OPTION(phasesaving)) {
        p_saved = solver->phases.saved[idx];
    }
    
    if (!p_saved) {
        p_saved = INITIAL_PHASE;
        INC(initial_decisions);
    } else {
        INC(saved_decisions);
    }

    int current_saved = p_saved;
    uint64_t current_conflicts = CONFLICTS;

    // Track variable's recent volatility
    if (kissat_last_seen_phase[idx] == 0) {
        kissat_last_seen_phase[idx] = current_saved;
        kissat_flip_window_start[idx] = current_conflicts;
    } else if (kissat_last_seen_phase[idx] != current_saved) {
        if (current_conflicts - kissat_flip_window_start[idx] > 100) {
            kissat_flip_count[idx] = 1;
            kissat_flip_window_start[idx] = current_conflicts;
        } else {
            kissat_flip_count[idx]++;
        }
        kissat_last_seen_phase[idx] = current_saved;
    } else {
        if (current_conflicts - kissat_flip_window_start[idx] > 100) {
            kissat_flip_count[idx] = 0;
            kissat_flip_window_start[idx] = current_conflicts;
        }
    }

    // Step 5: Mitigate local minimum traps (5% probability to invert if flipped > 3 times in 100 conflicts)
    if (kissat_flip_count[idx] > 3) {
        if (kissat_pick_double(&solver->random) < 0.05) {
            current_saved = -current_saved;
        }
    }

    return current_saved < 0 ? -1 : 1;
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
