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
    // Step 1: Calculate the raw performance metric 'P' for the just-concluded restart interval
    // P = 10000 / (average_LBD * sqrt(average_trail_level) + 1)
    // We use AVERAGE(fast_glue) for LBD and AVERAGE(level) for trail level.
    const double avg_lbd = AVERAGE(fast_glue);
    const double avg_level = AVERAGE(level);
    
    // Ensure non-negative value for sqrt (averages should be positive, but safety first)
    const double safe_level = (avg_level < 0.0) ? 0.0 : avg_level;
    const double p = 10000.0 / (avg_lbd * sqrt(safe_level) + 1.0);

    // Step 2: Update the exponential moving average Score 'S' for the active mode
    // S_new = 0.8 * S_old + 0.2 * P
    // We utilize solver->mab_reward to store the scores for each heuristic (mode).
    const unsigned h = solver->heuristic;
    
    // Safety check: ensure heuristic index is within bounds (typically 0 or 1)
    if (h >= 2) return; 

    const double s_old = solver->mab_reward[h];
    const double s_new = 0.8 * s_old + 0.2 * p;
    solver->mab_reward[h] = s_new;

    // Step 3: Calculate Momentum 'M'
    // d = S_new - S_old
    // M = 0.5 * M + 0.5 * d
    // We use a static variable for momentum as we cannot modify the solver struct.
    static double momentum = 0.0;
    const double d = s_new - s_old;
    momentum = 0.5 * momentum + 0.5 * d;
    const double delta = momentum;

    // Step 4: Execute Mode Selection based on trajectory
    // Only proceed if there are multiple heuristics to switch between
    if (solver->mab_heuristics > 1) {
        const double epsilon = 1e-7; // Threshold for approximately 0

        if (delta > epsilon) {
            // Delta > 0 (Performance Accelerating): deterministic keep
            // No change to solver->heuristic
        } else if (delta < -epsilon) {
            // Delta < 0 (Performance Decelerating): switch with probability
            // prob = min(1.0, abs(Delta) * 5.0)
            double prob = fabs(delta) * 5.0;
            if (prob > 1.0) prob = 1.0;

            if (kissat_pick_double(&solver->random) < prob) {
                // Switch mode (toggle between 0 and 1)
                solver->heuristic = 1 - h;
            }
        } else {
            // Delta approx 0: revert to comparison of S_stable vs S_focused
            // We interpret heuristic 0 as Stable-like strategy and 1 as Focused-like (or VSIDS vs CHB)
            if (solver->mab_reward[0] >= solver->mab_reward[1]) {
                solver->heuristic = 0;
            } else {
                solver->heuristic = 1;
            }
        }
    }

    // Bookkeeping for MAB statistics
    solver->mab_select[solver->heuristic]++;
    
    // Reset interval counters for the next phase
    solver->mab_decisions = 0;
    solver->mab_conflicts = 0;
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
