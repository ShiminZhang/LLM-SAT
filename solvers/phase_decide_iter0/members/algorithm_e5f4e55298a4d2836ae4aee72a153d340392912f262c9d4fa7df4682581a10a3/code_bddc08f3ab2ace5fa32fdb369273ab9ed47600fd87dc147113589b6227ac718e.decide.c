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
  // Step 1: Extend variable state to include phase quality array Q[2] and track conflicts.
  // We use static arrays within the function to avoid modifying the kissat struct,
  // strictly adhering to the requirement not to invent solver fields.
  static float *Q_0 = NULL;
  static float *Q_1 = NULL;
  static uint64_t *last_var_conflicts = NULL;
  static int allocated_vars = 0;

  if ((int)VARS >= allocated_vars) {
    int new_vars = (int)VARS + 10000;
    if (!Q_0) {
      Q_0 = (float *) kissat_calloc (solver, new_vars, sizeof (float));
      Q_1 = (float *) kissat_calloc (solver, new_vars, sizeof (float));
      last_var_conflicts = (uint64_t *) kissat_calloc (solver, new_vars, sizeof (uint64_t));
    } else {
      Q_0 = (float *) kissat_realloc (solver, Q_0, allocated_vars * sizeof (float), new_vars * sizeof (float));
      Q_1 = (float *) kissat_realloc (solver, Q_1, allocated_vars * sizeof (float), new_vars * sizeof (float));
      last_var_conflicts = (uint64_t *) kissat_realloc (solver, last_var_conflicts, allocated_vars * sizeof (uint64_t), new_vars * sizeof (uint64_t));
      for (int i = allocated_vars; i < new_vars; i++) {
        Q_0[i] = 0.0f;
        Q_1[i] = 0.0f;
        last_var_conflicts[i] = 0;
      }
    }
    allocated_vars = new_vars;
  }

  // Step 2 Approximation: Update phase quality using Exponential Moving Average.
  // Since we cannot hook into conflict analysis directly without modifying other files,
  // we lazily process recent redundant clauses for the current variable if conflicts have progressed.
  if (last_var_conflicts[idx] < CONFLICTS) {
    float q0 = Q_0[idx];
    float q1 = Q_1[idx];
    int clauses_checked = 0;
    for (all_clauses(c)) {
      if (c->redundant && !c->garbage) {
        for (all_literals_in_clause(lit, c)) {
          if (IDX(lit) == idx) {
            int p = (lit == LIT(idx)) ? 0 : 1;
            float lbd_val = (c->glue == 0) ? 1.0f : (float)c->glue;
            if (p == 0) q0 = 0.95f * q0 + 0.05f * (1.0f / lbd_val);
            else q1 = 0.95f * q1 + 0.05f * (1.0f / lbd_val);
            clauses_checked++;
            break;
          }
        }
      }
      // Limit to 20 recent clauses to prevent severe performance degradation
      if (clauses_checked >= 20) break;
    }
    Q_0[idx] = q0;
    Q_1[idx] = q1;
    last_var_conflicts[idx] = CONFLICTS;
  }

  // Step 3: Compute the standard phase S that Kissat would normally select.
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
  }

  if (!res && saved && (res = *saved)) {
    LOG ("%s uses saved decision phase %d", LOGVAR (idx), (int) res);
    INC (saved_decisions);
  }

  if (!res) {
    res = INITIAL_PHASE;
    LOG ("%s uses initial decision phase %d", LOGVAR (idx), (int) res);
    INC (initial_decisions);
  }
  assert (res);

  int S = res < 0 ? 0 : 1;
  int comp_S = 1 - S;

  // Step 4: Check if the competing phase 1-S demonstrates overwhelmingly superior historical LBD performance.
  if (CONFLICTS > 5000) {
    float q_comp = comp_S == 0 ? Q_0[idx] : Q_1[idx];
    float q_std = S == 0 ? Q_0[idx] : Q_1[idx];

    if (q_comp > 0.1f && (q_comp / (q_std + 0.001f)) > 1.5f) {
      res = comp_S == 0 ? -1 : 1;
      LOG ("%s overrides standard phase to %d due to LBD bandit", LOGVAR (idx), (int) res);
    }
  }

  // Step 5: If the threshold condition is not met, return the standard phase S.
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
