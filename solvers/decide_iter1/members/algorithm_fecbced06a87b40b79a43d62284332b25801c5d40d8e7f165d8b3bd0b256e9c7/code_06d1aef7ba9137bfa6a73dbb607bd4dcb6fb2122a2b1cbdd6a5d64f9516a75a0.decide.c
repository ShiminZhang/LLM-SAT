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
  // Step 1 (Initialization): Ensure the LPP table is allocated and large enough
  if (!kissat_lpp_false || kissat_lpp_capacity <= idx) {
    unsigned new_capacity = solver->vars;
    if (new_capacity <= idx)
      new_capacity = idx + 1;
    new_capacity += 1000; // Add margin for dynamically added variables

    float *new_false = kissat_calloc (solver, new_capacity, sizeof (float));
    float *new_true  = kissat_calloc (solver, new_capacity, sizeof (float));

    if (kissat_lpp_false) {
      for (unsigned i = 0; i < kissat_lpp_capacity; i++) {
        new_false[i] = kissat_lpp_false[i];
        new_true[i]  = kissat_lpp_true[i];
      }
      kissat_free (solver, kissat_lpp_false, kissat_lpp_capacity * sizeof (float));
      kissat_free (solver, kissat_lpp_true, kissat_lpp_capacity * sizeof (float));
    }

    kissat_lpp_false = new_false;
    kissat_lpp_true = new_true;
    kissat_lpp_capacity = new_capacity;
  }

  // Step 1 (Decay): Periodically decay all LPP scores by multiplying by 0.95 every 1000 conflicts
  static uint64_t last_decay_conflicts = 0;
  if (CONFLICTS < last_decay_conflicts) {
    last_decay_conflicts = CONFLICTS; // Handle solver restarts/re-initializations
  }
  while (CONFLICTS >= last_decay_conflicts + 1000) {
    last_decay_conflicts += 1000;
    for (unsigned i = 0; i < kissat_lpp_capacity; i++) {
      kissat_lpp_false[i] *= 0.95f;
      kissat_lpp_true[i] *= 0.95f;
    }
  }

  // Step 2: Calculate the polarity bias delta for the target variable
  float lpp_false = kissat_lpp_false[idx];
  float lpp_true  = kissat_lpp_true[idx];
  float delta = lpp_true - lpp_false;

  // Step 3: Evaluate Delta against learned-confidence threshold of 2.5
  if (fabs (delta) > 2.5f) {
    // Override phase saving and return True if Delta > 0, or False if Delta < 0
    return (delta > 0) ? 1 : -1;
  }

  // Step 4 & 5 Setup: Evaluate total conflict participation
  float total_lpp = lpp_false + lpp_true;

  bool force = GET_OPTION (forcephase);

  value *target;
  if (force)
    target = 0;
  else if (!GET_OPTION (target))
    target = 0;
  else if (solver->stable || GET_OPTION (target) > 1)
    target = solver->phases.target + idx;
  else
    target = 0;

  value *saved;
  if (force)
    saved = 0;
  else if (GET_OPTION (phasesaving))
    saved = solver->phases.saved + idx;
  else
    saved = 0;

  value res = 0;

  // Step 5: Mitigate local minimum traps in the fallback state
  if (total_lpp > 10.0f) {
    // Highly contested but lacks a clear polarity winner.
    // Deterministically return the solver's globally maintained 'target phase'
    if (target && *target) {
      res = *target;
      INC (target_decisions);
      return res < 0 ? -1 : 1;
    }
  }

  // Otherwise, fallback to retrieving the standard cached phase (P_saved) exactly.
  if (saved && *saved) {
    res = *saved;
    INC (saved_decisions);
    return res < 0 ? -1 : 1;
  }

  // Standard Kissat fallback if neither target nor saved phase was available
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

  if (!res) {
    res = INITIAL_PHASE;
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
