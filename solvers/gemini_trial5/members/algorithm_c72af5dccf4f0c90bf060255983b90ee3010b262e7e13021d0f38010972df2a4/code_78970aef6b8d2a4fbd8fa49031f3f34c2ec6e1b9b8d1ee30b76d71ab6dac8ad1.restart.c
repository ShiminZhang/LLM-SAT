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
    // Persistent state variables for EMAs and Forecasts
    // Index 0 = Focused, 1 = Stable
    static double fast_ema[2] = {0}; 
    static double slow_ema[2] = {0};
    static double forecast[2] = {0};
    static bool initialized = false;

    // Initialize state on first execution with current global average
    if (!initialized) {
        double current_metric = AVERAGE(fast_glue);
        fast_ema[0] = fast_ema[1] = current_metric;
        slow_ema[0] = slow_ema[1] = current_metric;
        forecast[0] = forecast[1] = current_metric;
        initialized = true;
    }

    // Determine current mode index (Stable=1, Focused=0)
    int mode = solver->stable ? 1 : 0;

    // Step 1: Define hybrid cost metric K
    double K = AVERAGE(fast_glue);

    // Calculate global glue clause ratio (LBD <= 2) in current phase/mode
    // Using MAX_GLUE_USED = 127 as per API reference
    uint64_t count_le2 = 0;
    uint64_t total = 0;
    
    // Sum LBD 1 and 2 for the ratio numerator (LBD 1 is possible for promoted clauses)
    count_le2 = solver->statistics.used[mode].glue[1] + 
                solver->statistics.used[mode].glue[2];

    // Sum all LBDs for the denominator
    for (int i = 0; i <= 127; i++) {
        total += solver->statistics.used[mode].glue[i];
    }

    // Apply reward (negative cost) for structural learning phases
    if (total > 0) {
        double ratio = (double)count_le2 / total;
        if (ratio > 0.1) {
            K -= 4.0;
        }
    }

    // Step 2: Update active mode's EMAs
    // fast alpha = 0.05, slow alpha = 0.005
    fast_ema[mode] = 0.05 * K + 0.95 * fast_ema[mode];
    slow_ema[mode] = 0.005 * K + 0.995 * slow_ema[mode];

    // Step 3: Calculate Asymmetric Performance Velocity V
    double fast = fast_ema[mode];
    double slow = slow_ema[mode];
    double V = 0.0;

    if (slow > 1e-9) { // Prevent division by zero
        V = (fast - slow) * (fast / slow);
    }

    // Step 4: Compute Forecasted Cost F
    // F = fast_ema + (2.5 * V)
    double F = fast + (2.5 * V);
    forecast[mode] = F;

    // Step 5: Mode Selection (Stable vs Focused)
    int candidate_mode = 1 - mode; // The other mode
    double F_current = forecast[mode];
    double F_candidate = forecast[candidate_mode];

    // Apply hysteresis buffer: only switch if candidate forecast is < 0.9 * current
    if (F_candidate < 0.9 * F_current) {
        // Toggle search mode
        solver->stable = !solver->stable;
        
        // Update heuristic to match the new mode
        // Stable (true) -> VSIDS (0)
        // Focused (false) -> CHB (1)
        solver->heuristic = solver->stable ? 0 : 1;
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
