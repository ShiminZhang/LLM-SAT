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
    // Second-Order Predictive MAB: State Variables
    // 0 = Focused, 1 = Stable
    static double fast_ema[2];
    static double slow_ema[2];
    static double prev_V[2];
    static uint64_t prev_glue_1_2[2];
    static uint64_t last_global_learned[2];
    static bool initialized = false;

    // Initialization on first run
    if (!initialized) {
        double init_k = AVERAGE(fast_glue);
        for (int i = 0; i < 2; i++) {
            fast_ema[i] = init_k;
            slow_ema[i] = init_k;
            prev_V[i] = 0.0;
            // Initialize counters to current values to start with zero delta
            prev_glue_1_2[i] = solver->statistics.used[i].glue[1] + 
                               solver->statistics.used[i].glue[2];
            last_global_learned[i] = solver->statistics.clauses_learned;
        }
        initialized = true;
    }

    // Identify active mode (0=Focused, 1=Stable)
    int mode = solver->stable ? 1 : 0;

    // Step 1: Metric Calculation (K)
    double K = AVERAGE(fast_glue);

    // Calculate Global Glue Clause Ratio (LBD <= 2) in current phase
    // We use stats deltas since the last update of this mode
    uint64_t current_g1 = solver->statistics.used[mode].glue[1];
    uint64_t current_g2 = solver->statistics.used[mode].glue[2];
    uint64_t current_sum_1_2 = current_g1 + current_g2;
    uint64_t global_learned = solver->statistics.clauses_learned;

    uint64_t delta_glue = current_sum_1_2 - prev_glue_1_2[mode];
    uint64_t delta_learned = global_learned - last_global_learned[mode];

    // Update history for next call
    prev_glue_1_2[mode] = current_sum_1_2;
    last_global_learned[mode] = global_learned;

    double ratio = 0.0;
    if (delta_learned > 0) {
        ratio = (double)delta_glue / (double)delta_learned;
    }

    // Apply "negative cost" (reward) for structural learning phases
    if (ratio > 0.1) {
        K -= 4.0;
    }

    // Step 2: Update EMAs for active mode
    // Fast alpha = 0.05, Slow alpha = 0.005
    fast_ema[mode] += 0.05 * (K - fast_ema[mode]);
    slow_ema[mode] += 0.005 * (K - slow_ema[mode]);

    // Step 3: Calculate Performance Velocity V
    double V_curr = fast_ema[mode] - slow_ema[mode];

    // Step 4: Calculate Acceleration A and Forecast F
    double A = V_curr - prev_V[mode];
    prev_V[mode] = V_curr; // Update state variable

    double F[2];
    
    // Active mode forecast: F = fast + 2V + 1A
    F[mode] = fast_ema[mode] + (2.0 * V_curr) + (1.0 * A);

    // Inactive mode forecast
    // For the inactive mode, stats haven't changed, so V is constant and A is effectively 0
    int other = 1 - mode;
    double V_other = fast_ema[other] - slow_ema[other];
    F[other] = fast_ema[other] + (2.0 * V_other);

    // Step 5: Select Mode
    // Select the mode with the lower Forecasted Cost F
    int candidate = (F[0] < F[1]) ? 0 : 1; // 0=Focused, 1=Stable

    // Apply Hysteresis: only switch if candidate is significantly better
    if (candidate != mode) {
        if (F[candidate] < 0.9 * F[mode]) {
            // Perform Mode Switch
            if (candidate == 1) {
                solver->stable = true;
                solver->heuristic = 0; // VSIDS
            } else {
                solver->stable = false;
                solver->heuristic = 1; // CHB
            }
        }
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
