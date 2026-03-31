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
    static int current_vars = 0;
    static double *phase_score_pos = NULL;
    static double *phase_score_neg = NULL;
    static uint64_t last_restarts = 0;

    // Step 1 & 2 Setup: Initialize and manage static memory for phase scores
    if (phase_score_pos == NULL) {
        phase_score_pos = (double*) kissat_calloc(solver, solver->vars, sizeof(double));
        phase_score_neg = (double*) kissat_calloc(solver, solver->vars, sizeof(double));
        current_vars = (int) solver->vars;
        last_restarts = solver->statistics.restarts;
    } else if (current_vars < (int) solver->vars) {
        unsigned old_size = (unsigned) current_vars * sizeof(double);
        unsigned new_size = solver->vars * sizeof(double);
        phase_score_pos = (double*) kissat_realloc(solver, phase_score_pos, old_size, new_size);
        phase_score_neg = (double*) kissat_realloc(solver, phase_score_neg, old_size, new_size);
        for (int i = current_vars; i < (int) solver->vars; i++) {
            phase_score_pos[i] = 0.0;
            phase_score_neg[i] = 0.0;
        }
        current_vars = (int) solver->vars;
    }

    // Decay all scores by a factor of 0.95 at every restart
    if (last_restarts < solver->statistics.restarts) {
        uint64_t diff = solver->statistics.restarts - last_restarts;
        double mult = 1.0;
        for (uint64_t d = 0; d < diff; d++) {
            mult *= 0.95;
        }
        for (int i = 0; i < current_vars; i++) {
            phase_score_pos[i] *= mult;
            phase_score_neg[i] *= mult;
        }
        last_restarts = solver->statistics.restarts;
    }

    // Step 3: Identify the current search mode and retrieve the baseline candidate phase
    // We utilize Kissat's innate phase saving tracking (saved = focused, target = stable)
    value phase_focused = solver->phases.saved[idx];
    value phase_stable = solver->phases.target[idx];
    
    if (!phase_focused) phase_focused = INITIAL_PHASE;
    if (!phase_stable) phase_stable = INITIAL_PHASE;

    value P = solver->stable ? phase_stable : phase_focused;

    // Step 4: Evaluate the historical LBD quality of the candidate phase
    double score_P = (P > 0) ? phase_score_pos[idx] : phase_score_neg[idx];
    double score_not_P = (P > 0) ? phase_score_neg[idx] : phase_score_pos[idx];

    value res = P;

    if (score_not_P > 1.5 * score_P && score_not_P > 2.0) {
        // Override the baseline and select !P
        res = -P;
    } else {
        // Step 5: Evaluate the consensus between the two modes
        if (phase_focused == phase_stable) {
            res = P;
        } else {
            // Modes disagree, return the phase that has the strictly higher phase_score[v]
            double score_focused = (phase_focused > 0) ? phase_score_pos[idx] : phase_score_neg[idx];
            double score_stable = (phase_stable > 0) ? phase_score_pos[idx] : phase_score_neg[idx];
            
            if (score_focused > score_stable) {
                res = phase_focused;
            } else if (score_stable > score_focused) {
                res = phase_stable;
            } else {
                // Exact tie broken by returning the baseline candidate phase P
                res = P;
            }
        }
    }

    // Fallback sanity check (should not be triggered given INITIAL_PHASE defaults)
    if (!res) res = INITIAL_PHASE;

    LOG ("%s uses decision phase %d", LOGVAR (idx), (int) res);
    
    // Log decisions statistics reliably
    if (res == solver->phases.target[idx] && solver->phases.target[idx] != 0) {
        INC (target_decisions);
    } else if (res == solver->phases.saved[idx] && solver->phases.saved[idx] != 0) {
        INC (saved_decisions);
    } else {
        INC (initial_decisions);
    }

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
