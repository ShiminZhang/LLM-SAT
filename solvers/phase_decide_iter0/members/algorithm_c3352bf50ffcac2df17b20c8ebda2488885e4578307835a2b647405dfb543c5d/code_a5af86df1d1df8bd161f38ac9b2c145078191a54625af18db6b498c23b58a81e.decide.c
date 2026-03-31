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

  if (!res) {
    value p_saved = saved ? *saved : 0;
    value p_target = target ? *target : 0;

    static int *phase_toggle_count = NULL;
    static int *last_saved_phase = NULL;
    static int alloc_size = 0;
    static double total_toggles = 0.0;

    if (alloc_size < (int)solver->vars) {
      int new_size = solver->vars + 1000;
      if (phase_toggle_count == NULL) {
        phase_toggle_count = (int *) kissat_calloc (solver, new_size, sizeof (int));
        last_saved_phase = (int *) kissat_calloc (solver, new_size, sizeof (int));
      } else {
        phase_toggle_count = (int *) kissat_realloc (solver, phase_toggle_count, alloc_size * sizeof (int), new_size * sizeof (int));
        last_saved_phase = (int *) kissat_realloc (solver, last_saved_phase, alloc_size * sizeof (int), new_size * sizeof (int));
        for (int i = alloc_size; i < new_size; i++) {
          phase_toggle_count[i] = 0;
          last_saved_phase[i] = 0;
        }
      }
      alloc_size = new_size;
    }

    if (p_saved != 0) {
      if (last_saved_phase[idx] != 0 && last_saved_phase[idx] != p_saved) {
        phase_toggle_count[idx]++;
        total_toggles += 1.0;
      }
      last_saved_phase[idx] = p_saved;
    }

    static double avg_score = 0.0;
    static int decide_calls = 0;

    if (decide_calls++ % 1024 == 0) {
      double sum = 0.0;
      int count = 0;
      heap *scores = kissat_get_scores (solver);
      for (all_variables (i)) {
        if (ACTIVE (i)) {
          sum += kissat_get_heap_score (scores, i);
          count++;
        }
      }
      avg_score = count > 0 ? sum / count : 0.0;
    }

    if (p_target == 0 || p_saved == p_target) {
      res = p_saved;
      if (res) {
        LOG ("%s uses saved decision phase %d (target uninit or match)", LOGVAR (idx), (int) res);
        INC (saved_decisions);
      }
    } else {
      heap *scores = kissat_get_scores (solver);
      double score_v = kissat_get_heap_score (scores, idx);

      if (score_v > 1.5 * avg_score) {
        res = p_saved;
        if (res) {
          LOG ("%s uses saved decision phase %d (conflict driver)", LOGVAR (idx), (int) res);
          INC (saved_decisions);
        }
      } else {
        double global_avg_toggle = (solver->vars > 0) ? total_toggles / solver->vars : 0.0;
        if (phase_toggle_count[idx] < global_avg_toggle) {
          res = p_target;
          if (res) {
            LOG ("%s uses target decision phase %d (stable background)", LOGVAR (idx), (int) res);
            INC (target_decisions);
          }
        } else {
          res = p_saved;
          if (res) {
            LOG ("%s uses saved decision phase %d (volatile background)", LOGVAR (idx), (int) res);
            INC (saved_decisions);
          }
        }
      }
    }
  }

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
