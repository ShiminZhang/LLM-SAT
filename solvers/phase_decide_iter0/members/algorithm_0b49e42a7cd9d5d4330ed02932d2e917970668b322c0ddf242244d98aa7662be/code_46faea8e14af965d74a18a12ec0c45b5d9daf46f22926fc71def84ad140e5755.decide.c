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
    // Step 1: Extend variable state (handled via static arrays to avoid struct changes)
    if (solver->vars >= Q_size) {
        unsigned new_size = solver->vars + 1024;
        float *new_Q0 = kissat_calloc(solver, new_size, sizeof(float));
        float *new_Q1 = kissat_calloc(solver, new_size, sizeof(float));
        if (Q_0) {
            for (unsigned i = 0; i < Q_size; i++) {
                new_Q0[i] = Q_0[i];
                new_Q1[i] = Q_1[i];
            }
            kissat_free(solver, Q_0, Q_size * sizeof(float));
            kissat_free(solver, Q_1, Q_size * sizeof(float));
        }
        Q_0 = new_Q0;
        Q_1 = new_Q1;
        Q_size = new_size;
    }

    // Step 3: Compute the standard phase S that Kissat would normally select
    bool force = GET_OPTION(forcephase);

    value *target;
    if (force) target = 0;
    else if (!GET_OPTION(target)) target = 0;
    else if (solver->stable || GET_OPTION(target) > 1) target = solver->phases.target + idx;
    else target = 0;

    value *saved;
    if (force) saved = 0;
    else if (GET_OPTION(phasesaving)) saved = solver->phases.saved + idx;
    else saved = 0;

    value res = 0;
    int type = 0; // Track which heuristic determined the standard phase

    if (!solver->stable) {
        switch ((solver->statistics.switched >> 1) & 7) {
            case 1: res = INITIAL_PHASE; break;
            case 3: res = -INITIAL_PHASE; break;
        }
    }

    if (!res && target && (res = *target)) type = 1;
    else if (!res && saved && (res = *saved)) type = 2;
    else if (!res) { res = INITIAL_PHASE; type = 3; }

    int S = (res > 0) ? 1 : 0;
    int S_comp = 1 - S;
    int final_S = S;
    bool overridden = false;

    // Step 4: Check if the competing phase 1-S has a significantly better historical conflict quality
    if (CONFLICTS > 5000) {
        float q_comp = (S_comp == 0) ? Q_0[idx] : Q_1[idx];
        float q_std  = (S == 0) ? Q_0[idx] : Q_1[idx];
        
        if (q_comp - q_std > 0.10f) {
            final_S = S_comp;
            overridden = true;
        }
    }

    // Step 5: Epsilon-greedy exploration strategy
    if (!overridden && !solver->stable) {
        // with a small probability (2%) during focused search mode, return competing phase
        unsigned r = kissat_next_random32(&solver->random) % 100;
        if (r < 2) {
            final_S = S_comp;
            overridden = true;
        }
    }

    // Logging and statistics updates
    if (!overridden) {
        if (type == 1) {
            LOG("%s uses target decision phase %d", LOGVAR(idx), (int)res);
            INC(target_decisions);
        } else if (type == 2) {
            LOG("%s uses saved decision phase %d", LOGVAR(idx), (int)res);
            INC(saved_decisions);
        } else if (type == 3) {
            LOG("%s uses initial decision phase %d", LOGVAR(idx), (int)res);
            INC(initial_decisions);
        }
    } else {
        LOG("%s uses overridden bandit decision phase %d", LOGVAR(idx), final_S ? 1 : -1);
    }

    return final_S ? 1 : -1;
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
