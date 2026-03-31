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
  // Step 1: Initialize two floating-point arrays for all variables
  if (!phase_initialized || alloc_vars < (int)solver->vars) {
    if (phase_score_0) kissat_free (solver, phase_score_0, alloc_vars * sizeof(double));
    if (phase_score_1) kissat_free (solver, phase_score_1, alloc_vars * sizeof(double));
    alloc_vars = solver->vars;
    phase_score_0 = kissat_calloc (solver, alloc_vars, sizeof(double));
    phase_score_1 = kissat_calloc (solver, alloc_vars, sizeof(double));
    phase_initialized = 1;
  }

  // Step 3: During solver restarts, apply a multiplicative decay factor of 0.95
  uint64_t current_restarts = GET (restarts);
  if (current_restarts > last_restarts) {
    double decay = pow (0.95, (double)(current_restarts - last_restarts));
    for (int i = 0; i < alloc_vars; i++) {
      phase_score_0[i] *= decay;
      phase_score_1[i] *= decay;
    }
    last_restarts = current_restarts;
  }

  // Step 2: During conflict analysis, whenever a learned clause C is generated...
  // Since we cannot hook directly into analyze.c, we approximate this step by 
  // scanning for newly generated learned clauses (using c->searched == 0 as a marker)
  uint64_t current_conflicts = CONFLICTS;
  if (current_conflicts > last_conflicts) {
    for (all_clauses (c)) {
      if (c->redundant && !c->garbage && c->searched == 0) {
        double inc = 1.0 / (c->glue > 0 ? c->glue : 1);
        for (all_literals_in_clause (lit, c)) {
          unsigned v = IDX (lit);
          if (v < (unsigned)alloc_vars) {
            if (lit == LIT (v)) {
              phase_score_1[v] += inc;
            } else {
              phase_score_0[v] += inc;
            }
          }
        }
        c->searched = 1; // Mark as processed to prevent double-counting
      }
    }
    last_conflicts = current_conflicts;
  }

  // Step 4: Compute the confidence margin
  double delta = fabs (phase_score_1[idx] - phase_score_0[idx]);

  // --- Determine Kissat's Standard Saved Phase ---
  bool force = GET_OPTION (forcephase);
  value *target = 0;
  if (force)
    target = 0;
  else if (!GET_OPTION (target))
    target = 0;
  else if (solver->stable || GET_OPTION (target) > 1)
    target = solver->phases.target + idx;
  else
    target = 0;

  value *saved = 0;
  if (force)
    saved = 0;
  else if (GET_OPTION (phasesaving))
    saved = solver->phases.saved + idx;
  else
    saved = 0;

  value res = 0;
  if (!solver->stable) {
    switch ((solver->statistics.switched >> 1) & 7) {
    case 1:
      res = INITIAL_PHASE;
      break;
    case 3:
      res = -INITIAL_PHASE;
      break;
    }
  }

  if (!res && target && (res = *target)) {
    LOG ("%s uses target decision phase %d", LOGVAR (idx), (int) res);
    INC (target_decisions);
  } else if (!res && saved && (res = *saved)) {
    LOG ("%s uses saved decision phase %d", LOGVAR (idx), (int) res);
    INC (saved_decisions);
  } else if (!res) {
    res = INITIAL_PHASE;
    LOG ("%s uses initial decision phase %d", LOGVAR (idx), (int) res);
    INC (initial_decisions);
  }
  assert (res);

  int standard_phase = res < 0 ? -1 : 1;
  // -----------------------------------------------

  // Step 5: If delta < 0.05, momentum is inconclusive; return standard phase
  if (delta < 0.05) {
    return standard_phase;
  }

  // Step 6: Determine momentum-preferred phase
  int preferred_phase = (phase_score_1[idx] > phase_score_0[idx]) ? 1 : -1;

  // Step 7: Apply hysteresis threshold
  if (preferred_phase != standard_phase) {
    if (delta >= 0.15) {
      return preferred_phase;
    } else {
      return standard_phase;
    }
  }

  return preferred_phase;
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
