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
    // Step 1: Calculate the raw performance metric 'P' for the just-concluded restart interval.
    // P = (200 * sqrt(average_trail_level)) / (average_LBD + 1).
    // The AVERAGE macro retrieves the Exponential Moving Average for the current mode.
    const double avg_trail = AVERAGE(level);
    const double avg_lbd = AVERAGE(fast_glue);

    // Calculate P using standard math functions (sqrt from math.h).
    // Adding 1.0 to avg_lbd prevents division by zero.
    const double P = (200.0 * sqrt(avg_trail)) / (avg_lbd + 1.0);

    // Step 2: Update the exponential moving average Score 'S' for the active mode.
    // solver->mab_reward stores the scores. solver->heuristic is the active index (0 or 1).
    unsigned h = solver->heuristic;

    // Ensure heuristic index is within expected bounds (typically 0 or 1).
    if (h > 1) h = 0; 

    const double S_old = solver->mab_reward[h];
    // S_new = 0.8 * S_old + 0.2 * P
    const double S_new = 0.8 * S_old + 0.2 * P;

    solver->mab_reward[h] = S_new;

    // Step 3: Calculate the discrete derivative 'Delta' of the score.
    const double Delta = S_new - S_old;

    // Step 4: Execute Mode Selection based on trajectory.
    const double epsilon = 1e-6; // Define threshold for "approximately 0"

    if (Delta > epsilon) {
        // Delta > 0 (Performance Accelerating): deterministic keep of current mode.
        // No change to solver->heuristic.
    } 
    else if (Delta < -epsilon) {
        // Delta < 0 (Performance Decelerating): switch mode with probability.
        // prob = min(1.0, abs(Delta) * 5.0).
        double prob = fabs(Delta) * 5.0;
        if (prob > 1.0) prob = 1.0;

        // Generate random double in [0, 1)
        const double rand_val = kissat_pick_double(&solver->random);
        
        // Switch if random value is less than probability
        if (rand_val < prob) {
            solver->heuristic = 1 - h;
        }
    } 
    else {
        // Delta approx 0: revert to comparison of S_stable vs S_focused.
        // We compare the accumulated scores of the two heuristics (0 and 1).
        const double s0 = solver->mab_reward[0];
        const double s1 = solver->mab_reward[1];

        if (s1 > s0) {
            solver->heuristic = 1;
        } else {
            solver->heuristic = 0;
        }
    }

    // Update selection statistics for the active heuristic
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
