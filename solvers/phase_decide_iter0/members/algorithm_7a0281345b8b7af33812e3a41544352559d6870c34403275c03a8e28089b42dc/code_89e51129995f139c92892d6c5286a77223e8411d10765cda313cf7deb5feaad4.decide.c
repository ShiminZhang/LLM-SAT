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

  value *target_ptr;
  if (force)
    target_ptr = 0;
  else if (!GET_OPTION (target))
    target_ptr = 0;
  else if (solver->stable || GET_OPTION (target) > 1)
    target_ptr = solver->phases.target + idx;
  else
    target_ptr = 0;

  value *saved_ptr;
  if (force)
    saved_ptr = 0;
  else if (GET_OPTION (phasesaving))
    saved_ptr = solver->phases.saved + idx;
  else
    saved_ptr = 0;

  // Step 1 mapping: phase_focused corresponds to saved, phase_stable to target
  value phase_focused = saved_ptr ? *saved_ptr : 0;
  value phase_stable = target_ptr ? *target_ptr : 0;

  // Step 2 mapping: dynamic allocation of phase quality matrix
  // Using static int and dynamic allocation to avoid hallucinating solver fields
  static int allocated_vars = 0;
  static double *phase_score_pos = NULL;
  static double *phase_score_neg = NULL;

  if ((int)VARS >= allocated_vars) {
    int new_allocated = (int)VARS + 1024;
    if (phase_score_pos) {
      phase_score_pos = kissat_realloc (solver, phase_score_pos,
                                        allocated_vars * sizeof (double),
                                        new_allocated * sizeof (double));
      phase_score_neg = kissat_realloc (solver, phase_score_neg,
                                        allocated_vars * sizeof (double),
                                        new_allocated * sizeof (double));
    } else {
      phase_score_pos = kissat_calloc (solver, new_allocated, sizeof (double));
      phase_score_neg = kissat_calloc (solver, new_allocated, sizeof (double));
    }
    allocated_vars = new_allocated;
  }

  // Step 3: Identify search mode and retrieve initial candidate phase
  value P = 0;
  value P_alt = 0;

  if (!solver->stable) {
    P = phase_focused;
    P_alt = phase_stable;
  } else {
    P = phase_stable;
    P_alt = phase_focused;
  }

  // Fallback if phases are unassigned (0)
  value fallback = INITIAL_PHASE;
  if (!solver->stable) {
    switch ((solver->statistics.switched >> 1) & 7) {
    case 1:
      fallback = INITIAL_PHASE;
      break;
    case 3:
      fallback = -INITIAL_PHASE;
      break;
    }
  }

  if (!P) P = fallback;
  if (!P_alt) P_alt = fallback;

  // Retrieve scores for P and P_alt
  double score_P = (P > 0) ? phase_score_pos[idx] : phase_score_neg[idx];
  double score_P_alt = (P_alt > 0) ? phase_score_pos[idx] : phase_score_neg[idx];

  // Prevent adoption of stale phases on mode switches
  if (score_P < 0.01) {
    if (score_P_alt > score_P) {
      P = P_alt;
      score_P = score_P_alt; // Update baseline score
    }
  }

  // Step 4: Evaluate historical LBD quality of candidate phase
  value not_P = -P; // Opposite phase
  double score_not_P = (not_P > 0) ? phase_score_pos[idx] : phase_score_neg[idx];

  if (score_not_P > 1.5 * score_P && score_not_P > 2.0) {
    P = not_P;
  }

  LOG ("%s uses cross-mode decision phase %d", LOGVAR (idx), (int) P);
  assert (P);

  // Step 5: Return baseline candidate phase P
  return P < 0 ? -1 : 1;
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
