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

    // Step 2: Fetch P_saved and P_target
    value P_saved = 0;
    bool used_saved_array = false;
    
    if (!solver->stable) {
        switch ((solver->statistics.switched >> 1) & 7) {
            case 1: P_saved = INITIAL_PHASE; break;
            case 3: P_saved = -INITIAL_PHASE; break;
        }
    }
    if (!P_saved && saved && *saved) {
        P_saved = *saved;
        used_saved_array = true;
    }
    if (!P_saved) {
        P_saved = INITIAL_PHASE;
    }

    value P_target = (target && *target) ? *target : 0;

    // Evaluate target age (conflicts since global target assignment was last improved)
    static int min_unassigned = -1;
    static int64_t best_trail_conflicts = -1;
    
    int current_unassigned = (int)solver->unassigned;
    int64_t current_conflicts = (int64_t)CONFLICTS;
    
    // Reset trackers on new solver instance
    if (current_conflicts == 0) {
        min_unassigned = current_unassigned;
        best_trail_conflicts = 0;
    } else if (min_unassigned == -1 || current_unassigned < min_unassigned) {
        min_unassigned = current_unassigned;
        best_trail_conflicts = current_conflicts;
    }
    
    int64_t target_age = current_conflicts - best_trail_conflicts;
    if (target_age < 0) target_age = 0;

    value final_phase = 0;

    // Step 3: If P_target is uninitialized or P_saved == P_target, immediately return P_saved
    if (P_target == 0 || P_saved == P_target) {
        final_phase = P_saved;
    } else {
        // Step 1: Maintain running average of all active variable scores
        static double avg_score = 0.0;
        static int64_t last_update_decisions = -1;
        int64_t current_decisions = (int64_t)DECISIONS;
        
        if (current_decisions == 0) {
            last_update_decisions = -1;
        }
        
        // Update average score periodically to avoid O(V) overhead on every decision
        if (last_update_decisions == -1 || current_decisions - last_update_decisions > 1000) {
            double total_score = 0.0;
            int active_vars = 0;
            heap *scores = kissat_get_scores(solver);
            for (all_variables(i)) {
                if (ACTIVE(i)) {
                    total_score += kissat_get_heap_score(scores, i);
                    active_vars++;
                }
            }
            avg_score = active_vars > 0 ? total_score / active_vars : 0.0;
            last_update_decisions = current_decisions;
        }

        // Step 4: Evaluate current heuristic score of v
        heap *scores = kissat_get_scores(solver);
        double score_v = kissat_get_heap_score(scores, idx);

        if (score_v > 1.5 * avg_score) {
            // Classify as 'conflict driver', prioritize local conflict resolution
            final_phase = P_saved;
        } else {
            // Step 5: Classify as 'background variable'
            if (target_age > 5000) {
                // Stagnation threshold exceeded, drift away from stale attractor
                final_phase = P_saved;
            } else {
                // Aggressively anchor to best known global assignment
                final_phase = P_target;
            }
        }
    }

    // Update statistics based on the chosen phase
    if (final_phase == P_target && P_target != 0 && final_phase != (used_saved_array ? *saved : 0)) {
        INC(target_decisions);
    } else if (used_saved_array && final_phase == *saved) {
        INC(saved_decisions);
    } else {
        INC(initial_decisions);
    }

    return final_phase < 0 ? -1 : 1;
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
