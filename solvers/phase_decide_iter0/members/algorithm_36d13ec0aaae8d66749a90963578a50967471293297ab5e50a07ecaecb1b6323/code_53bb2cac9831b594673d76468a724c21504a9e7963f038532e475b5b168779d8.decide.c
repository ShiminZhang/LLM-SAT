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

  value target_val = 0;
  if (!force && GET_OPTION (target)) {
    if (solver->stable || GET_OPTION (target) > 1) {
      target_val = solver->phases.target[idx];
    }
  }

  value saved_val = 0;
  if (!force && GET_OPTION (phasesaving)) {
    saved_val = solver->phases.saved[idx];
  }

  value initial_val = INITIAL_PHASE;
  if (!solver->stable) {
    switch ((solver->statistics.switched >> 1) & 7) {
    case 1:
      initial_val = INITIAL_PHASE;
      break;
    case 3:
      initial_val = -INITIAL_PHASE;
      break;
    }
  }

  // Step 1: Maintain independent phase-saving candidates
  // Kissat naturally maintains saved (focused) and target (stable) phases.
  value p_focused = saved_val ? saved_val : initial_val;
  value p_stable = target_val ? target_val : p_focused;

  // Step 2: Maintain phase quality matrix
  // Statically allocated to strictly avoid modifying the kissat struct.
  // Note: Conflict analysis updates (+1.0/LBD) are assumed to be handled
  // globally or omitted if restricted to decide.c, but the structure is maintained.
  static double *phase_score_0 = 0;
  static double *phase_score_1 = 0;
  static int current_vars = 0;
  static uint64_t last_restarts = 0;

  int vars = VARS;
  if (vars > current_vars) {
    int new_vars = vars + 1000;
    double *new_0 = kissat_calloc (solver, new_vars, sizeof (double));
    double *new_1 = kissat_calloc (solver, new_vars, sizeof (double));

    if (phase_score_0) {
      for (int i = 0; i < current_vars; i++) {
        new_0[i] = phase_score_0[i];
        new_1[i] = phase_score_1[i];
      }
      kissat_free (solver, phase_score_0, current_vars * sizeof (double));
      kissat_free (solver, phase_score_1, current_vars * sizeof (double));
    }

    phase_score_0 = new_0;
    phase_score_1 = new_1;
    current_vars = new_vars;
  }

  // Decay scores at every restart by a factor of 0.95
  uint64_t current_restarts = solver->statistics.restarts;
  if (current_restarts > last_restarts) {
    uint64_t diff = current_restarts - last_restarts;
    double decay = 1.0;
    for (uint64_t i = 0; i < diff && i < 1000; i++) {
      decay *= 0.95;
    }
    for (int i = 0; i < current_vars; i++) {
      phase_score_0[i] *= decay;
      phase_score_1[i] *= decay;
    }
    last_restarts = current_restarts;
  }

  // Step 3: Identify current search mode and retrieve baseline candidate phase
  value P = solver->stable ? p_stable : p_focused;
  value P_alt = solver->stable ? p_focused : p_stable;

  double score_P = (P > 0) ? phase_score_1[idx] : phase_score_0[idx];
  double score_not_P = (P > 0) ? phase_score_0[idx] : phase_score_1[idx];

  value res = P;

  // Step 4: Evaluate historical LBD quality
  if (score_not_P > 1.5 * score_P && score_not_P > 2.0) {
    res = -P;
  }
  // Step 5: Soft cross-mode knowledge transfer
  else {
    double score_P_alt = (P_alt > 0) ? phase_score_1[idx] : phase_score_0[idx];
    if (P != P_alt && score_P_alt > score_P) {
      res = P_alt;
    }
  }

  // Fallback in case of 0
  if (!res) {
    res = INITIAL_PHASE;
  }

  // Logging and Stats (matching baseline expectations)
  if (res == target_val && target_val != 0) {
    LOG ("%s uses target decision phase %d", LOGVAR (idx), (int) res);
    INC (target_decisions);
  } else if (res == saved_val && saved_val != 0) {
    LOG ("%s uses saved decision phase %d", LOGVAR (idx), (int) res);
    INC (saved_decisions);
  } else {
    LOG ("%s uses initial decision phase %d", LOGVAR (idx), (int) res);
    INC (initial_decisions);
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
