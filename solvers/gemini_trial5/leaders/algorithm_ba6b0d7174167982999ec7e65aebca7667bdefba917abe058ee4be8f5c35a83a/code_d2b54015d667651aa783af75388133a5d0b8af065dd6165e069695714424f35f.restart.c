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
    // Persistent state variables for Stable (1) and Focused (0) modes
    static double fast_ema[2] = {0, 0};
    static double slow_ema[2] = {0, 0};
    static uint64_t last_glue2_count[2] = {0, 0};
    static uint64_t last_conflict_count[2] = {0, 0};
    static bool initialized[2] = {false, false};

    // Determine current mode index
    int current_mode = solver->stable ? 1 : 0;
    int candidate_mode = 1 - current_mode;

    // Step 1: Define Hybrid Cost Metric K
    // K = average_LBD (using fast glue average as proxy)
    double K = AVERAGE(fast_glue);

    // Calculate global glue clause ratio (LBD <= 2) in the current phase
    // We use the cumulative usage statistics for LBD 1 and 2
    uint64_t current_glue2 = solver->statistics.used[current_mode].glue[1] + 
                             solver->statistics.used[current_mode].glue[2];
    uint64_t current_conflicts = CONFLICTS;

    uint64_t delta_glue2 = current_glue2 - last_glue2_count[current_mode];
    uint64_t delta_conflicts = current_conflicts - last_conflict_count[current_mode];

    // Update history for next iteration
    last_glue2_count[current_mode] = current_glue2;
    last_conflict_count[current_mode] = current_conflicts;

    double ratio = 0.0;
    if (delta_conflicts > 0) {
        ratio = (double)delta_glue2 / (double)delta_conflicts;
    }

    // Apply reward: if ratio > 0.1, subtract 4.0 from K
    if (ratio > 0.1) {
        K -= 4.0;
    }

    // Step 2: Update EMAs for the active mode
    // alpha_fast = 0.05, alpha_slow = 0.005
    if (!initialized[current_mode]) {
        fast_ema[current_mode] = K;
        slow_ema[current_mode] = K;
        initialized[current_mode] = true;
    } else {
        fast_ema[current_mode] += 0.05 * (K - fast_ema[current_mode]);
        slow_ema[current_mode] += 0.005 * (K - slow_ema[current_mode]);
    }

    // Step 3 & 4: Calculate Performance Velocity V and Forecasted Cost F
    double F[2];
    for (int m = 0; m < 2; m++) {
        if (!initialized[m]) {
            // If the mode hasn't been initialized, assume current K to avoid bias
            F[m] = K;
        } else {
            double V = fast_ema[m] - slow_ema[m];
            F[m] = fast_ema[m] + (2.5 * V);
        }
    }

    double F_current = F[current_mode];
    double F_candidate = F[candidate_mode];

    // Step 5: Select Mode with Hysteresis
    // Only switch if F_candidate < 0.9 * F_current
    if (F_candidate < 0.9 * F_current) {
        solver->stable = !solver->stable;
        
        // Reset delta counters for the new mode to avoid using stale history
        int new_mode = solver->stable ? 1 : 0;
        last_glue2_count[new_mode] = solver->statistics.used[new_mode].glue[1] + 
                                     solver->statistics.used[new_mode].glue[2];
        last_conflict_count[new_mode] = CONFLICTS;

        kissat_extremely_verbose(solver, 
            "predictive mab switch to %s (F_curr=%.2f, F_cand=%.2f)", 
            solver->stable ? "stable" : "focused", F_current, F_candidate);
    }
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
