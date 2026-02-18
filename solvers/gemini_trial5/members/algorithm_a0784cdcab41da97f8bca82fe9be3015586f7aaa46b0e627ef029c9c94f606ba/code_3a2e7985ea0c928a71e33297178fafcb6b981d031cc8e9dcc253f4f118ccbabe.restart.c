#include "restart.h"
#include "backtrack.h"
#include "bump.h"
#include "decide.h"
#include "internal.h"
#include "kimits.h"
#include "logging.h"
#include "print.h"
#include "reluctant.h"
#include "report.h"

#include <inttypes.h>
#include <math.h>

bool kissat_restarting (kissat *solver) {
  assert (solver->unassigned);
  if (!GET_OPTION (restart))
    return false;
  if (!solver->level)
    return false;
  if (CONFLICTS < solver->limits.restart.conflicts)
    return false;
  if (solver->stable)
    return kissat_reluctant_triggered (&solver->reluctant);
  const double fast = AVERAGE (fast_glue);
  const double slow = AVERAGE (slow_glue);
  const double margin = (100.0 + GET_OPTION (restartmargin)) / 100.0;
  const double limit = margin * slow;
  kissat_extremely_verbose (solver,
                            "restart glue limit %g = "
                            "%.02f * %g (slow glue) %c %g (fast glue)",
                            limit, margin, slow,
                            (limit > fast    ? '>'
                             : limit == fast ? '='
                                             : '<'),
                            fast);
  return (limit <= fast);
}

void kissat_update_focused_restart_limit (kissat *solver) {
  assert (!solver->stable);
  limits *limits = &solver->limits;
  uint64_t restarts = solver->statistics.restarts;
  uint64_t delta = GET_OPTION (restartint);
  if (restarts)
    delta += kissat_logn (restarts) - 1;
  limits->restart.conflicts = CONFLICTS + delta;
  kissat_extremely_verbose (solver,
                            "focused restart limit at %" PRIu64
                            " after %" PRIu64 " conflicts ",
                            limits->restart.conflicts, delta);
}

static unsigned reuse_stable_trail (kissat *solver) {
  const heap *const scores = kissat_get_scores(solver);
  const unsigned next_idx = kissat_next_decision_variable (solver);
  const double limit = kissat_get_heap_score (scores, next_idx);
  unsigned level = solver->level, res = 0;
  while (res < level) {
    frame *f = &FRAME (res + 1);
    const unsigned idx = IDX (f->decision);
    const double score = kissat_get_heap_score (scores, idx);
    if (score <= limit)
      break;
    res++;
  }
  return res;
}

static unsigned reuse_focused_trail (kissat *solver) {
  const links *const links = solver->links;
  const unsigned next_idx = kissat_next_decision_variable (solver);
  const unsigned limit = links[next_idx].stamp;
  LOG ("next decision variable stamp %u", limit);
  unsigned level = solver->level, res = 0;
  while (res < level) {
    frame *f = &FRAME (res + 1);
    const unsigned idx = IDX (f->decision);
    const unsigned score = links[idx].stamp;
    if (score <= limit)
      break;
    res++;
  }
  return res;
}

static unsigned reuse_trail (kissat *solver) {
  assert (solver->level);
  assert (!EMPTY_STACK (solver->trail));

  if (!GET_OPTION (restartreusetrail))
    return 0;

  unsigned res;

  if (solver->stable)
    res = reuse_stable_trail (solver);
  else
    res = reuse_focused_trail (solver);

  LOG ("matching trail level %u", res);

  if (res) {
    INC (restarts_reused_trails);
    ADD (restarts_reused_levels, res);
    LOG ("restart reuses trail at decision level %u", res);
  } else
    LOG ("restarts does not reuse the trail");

  return res;
}

void restart_mab(kissat *solver) {
    // Capture metrics from the just-completed phase
    unsigned conflicts = solver->mab_conflicts;
    double decisions = solver->mab_decisions;

    // Reset MAB tracking variables for the next phase
    // This is standard housekeeping in Kissat to prepare for the new interval
    solver->mab_decisions = 0;
    solver->mab_conflicts = 0;
    solver->mab_chosen_tot = 0;
    for (all_variables(idx)) {
        solver->mab_chosen[idx] = 0;
    }

    // Static state variables to persist R_prev and consecutive_count across calls
    // We use statics because we cannot modify the solver struct
    static double last_reward[2] = {0.0, 0.0};
    static int consecutive_run = 0;

    // Initialization check: if no selections have been made yet, reset statics
    // (This handles cases where the solver might be restarted in a test harness)
    if (solver->mab_select[0] == 0 && solver->mab_select[1] == 0) {
        last_reward[0] = 0.0;
        last_reward[1] = 0.0;
        consecutive_run = 0;
    }

    unsigned current = solver->heuristic;
    unsigned other = 1 - current; // Assumes 2 heuristics (0 and 1)

    // Step 1: Compute the raw reward R
    // R = log2(conflicts_generated + 1) / (elapsed_ticks + 1)
    double r_raw = log2((double)conflicts + 1.0) / (decisions + 1.0);

    // Step 2: Calculate the performance trend T
    // T = R - R_prev
    double r_prev = last_reward[current];
    double trend = r_raw - r_prev;
    
    // Update R_prev to R for the next time this mode is used
    last_reward[current] = r_raw;

    // Step 3: Update the Q-value (score) for the current mode
    // Q_new = 0.85 * Q_old + 0.15 * (R + 2.0 * T)
    double q_old = solver->mab_reward[current];
    double q_new = 0.85 * q_old + 0.15 * (r_raw + 2.0 * trend);
    solver->mab_reward[current] = q_new;

    // Step 4: Calculate the Selection Score S for both modes
    // For the mode just finished: S = Q_new * (0.95 ^ consecutive_count)
    double fatigue = pow(0.95, (double)consecutive_run);
    double s_current = q_new * fatigue;

    // For the inactive mode: S = Q_current
    double s_other = solver->mab_reward[other];

    // Step 5: Identify the candidate mode and perform stability check
    bool switch_mode = false;

    // Identify candidate with highest S
    if (s_other > s_current) {
        // Candidate is the inactive mode (other)
        // Stability check: prevent switch unless:
        // 1. consecutive_count >= 2 OR
        // 2. S_candidate > 1.15 * S_current
        if (consecutive_run >= 2 || s_other > 1.15 * s_current) {
            switch_mode = true;
        }
    }

    if (switch_mode) {
        solver->heuristic = other;
        consecutive_run = 0; // Reset count on switch
    } else {
        // Stay in current mode
        // solver->heuristic remains 'current'
        consecutive_run++; // Increment count
    }

    // Update selection count for the chosen heuristic
    solver->mab_select[solver->heuristic]++;
}

void kissat_restart (kissat *solver) {
  START (restart);
  INC (restarts);
  ADD (restarts_levels, solver->level);
  if (solver->stable)
    INC (stable_restarts);
  else
    INC (focused_restarts);

  unsigned old_heuristic = solver->heuristic;
  if (solver->stable && solver->mab) 
      restart_mab(solver);
  unsigned new_heuristic = solver->heuristic;

  unsigned level = old_heuristic==new_heuristic?reuse_trail (solver):0;

  kissat_extremely_verbose (solver,
                            "restarting after %" PRIu64 " conflicts"
                            " (limit %" PRIu64 ")",
                            CONFLICTS, solver->limits.restart.conflicts);
  LOG ("restarting to level %u", level);
  if (solver->stable && solver->mab) solver->heuristic = old_heuristic;
  kissat_backtrack_in_consistent_state (solver, level);
  if (solver->stable && solver->mab) solver->heuristic = new_heuristic;
  if (!solver->stable)
    kissat_update_focused_restart_limit (solver);
  
  if (solver->stable && solver->mab && old_heuristic!=new_heuristic) kissat_update_scores(solver);

  REPORT (1, 'R');
  STOP (restart);
}
