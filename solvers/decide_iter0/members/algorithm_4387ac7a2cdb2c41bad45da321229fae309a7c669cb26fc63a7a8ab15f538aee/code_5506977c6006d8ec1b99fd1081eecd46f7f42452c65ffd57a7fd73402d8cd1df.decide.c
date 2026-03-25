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

  // --- Target-Guided Contention Phasing Implementation ---
  
  static double *LPP_0 = 0;
  static double *LPP_1 = 0;
  static unsigned allocated_vars = 0;
  static uint64_t last_decay_conflicts = 0;
  static uint64_t last_learned = 0;

  // Step 1: Access (and maintain) the globally maintained LPP table.
  // Dynamically allocate/reallocate based on VARS to remain safe.
  if (VARS > allocated_vars) {
    unsigned new_size = VARS + 1000; // Pad to reduce reallocations
    if (LPP_0) {
      LPP_0 = kissat_realloc (solver, LPP_0, allocated_vars * sizeof(double), new_size * sizeof(double));
      LPP_1 = kissat_realloc (solver, LPP_1, allocated_vars * sizeof(double), new_size * sizeof(double));
      for (unsigned i = allocated_vars; i < new_size; i++) {
        LPP_0[i] = 0.0;
        LPP_1[i] = 0.0;
      }
    } else {
      LPP_0 = kissat_calloc (solver, new_size, sizeof(double));
      LPP_1 = kissat_calloc (solver, new_size, sizeof(double));
    }
    allocated_vars = new_size;
  }

  // Periodically decay all LPP scores by multiplying by 0.95 every 1000 conflicts
  if (CONFLICTS >= last_decay_conflicts + 1000) {
    for (unsigned i = 0; i < allocated_vars; i++) {
      LPP_0[i] *= 0.95;
      LPP_1[i] *= 0.95;
    }
    last_decay_conflicts = CONFLICTS;
  }

  // Process newly learned clauses. We scan the clause arena periodically 
  // and use the 'searched' field to tag clauses we have already evaluated.
  if (solver->statistics.clauses_learned > last_learned) {
    for (all_clauses (c)) {
      if (c->redundant && !c->garbage && c->searched == 0) {
        double weight = c->glue > 0 ? 1.0 / c->glue : 1.0;
        for (all_literals_in_clause (lit, c)) {
          unsigned v = IDX (lit);
          if (v < allocated_vars) {
            unsigned sign = lit & 1;
            if (sign == 0) // Positive literal implies True phase preference
              LPP_1[v] += weight;
            else           // Negative literal implies False phase preference
              LPP_0[v] += weight;
          }
        }
        c->searched = 1; // Mark as processed for the LPP table
      }
    }
    last_learned = solver->statistics.clauses_learned;
  }

  // Evaluate the phase preference if not overridden by unstable mode defaults
  if (!res) {
    // Step 2: Calculate Delta
    double delta = LPP_1[idx] - LPP_0[idx];
    
    // Step 3: Evaluate Delta against threshold
    if (delta > 2.5 || delta < -2.5) {
      res = (delta > 0) ? 1 : -1;
      LOG ("%s uses LPP delta decision phase %d", LOGVAR (idx), (int) res);
    } else {
      // Step 4 & 5: Weak preference, check contention score
      double contention = LPP_0[idx] + LPP_1[idx];
      
      if (contention > 5.0) {
        if (target && *target) {
          res = *target;
          LOG ("%s uses target decision phase %d due to high contention", LOGVAR (idx), (int) res);
          INC (target_decisions);
        } else if (saved && *saved) {
          res = *saved;
          LOG ("%s uses saved decision phase %d", LOGVAR (idx), (int) res);
          INC (saved_decisions);
        }
      } else {
        if (saved && *saved) {
          res = *saved;
          LOG ("%s uses saved decision phase %d", LOGVAR (idx), (int) res);
          INC (saved_decisions);
        }
      }
    }
  }

  // Fallback to initial phase if nothing else matched
  if (!res) {
    res = INITIAL_PHASE;
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
