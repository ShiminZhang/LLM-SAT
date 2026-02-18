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
    // Step 1: Compute Phase Metrics
    // L_phase: Average LBD of the completed phase. 
    // We use the Fast EMA of glue (LBD) as the proxy for the recent phase average.
    double l_phase = AVERAGE(fast_glue);
    if (l_phase < 1.0) l_phase = 1.0; // Safety floor

    // D: Average decision level of the phase.
    double d_avg = AVERAGE(level);
    
    // N: Total number of variables.
    double n_vars = (double)solver->vars;
    
    // d_ratio = D / N (Depth ratio)
    double d_ratio = (n_vars > 0) ? (d_avg / n_vars) : 0.0;

    // Step 2: Global LBD and Base Reward
    // L_global: Running global average LBD. 
    // We use the Slow EMA of glue (LBD) which tracks the long-term global average.
    double l_global = AVERAGE(slow_glue);
    if (l_global < 1.0) l_global = 1.0;

    // R = L_global / L_phase
    // This provides a relative performance metric centered around 1.0.
    double r = l_global / l_phase;

    // Step 3: Asymmetric Incentives
    // If the completed phase was STABLE (Heuristic 0 = VSIDS), we incentivize deep search.
    // We assume Heuristic 0 corresponds to the 'Stable' strategy.
    if (solver->heuristic == 0) {
        r = r * (1.0 + (1.5 * d_ratio));
    }

    // Step 4: Fatigue Penalty
    // Track consecutive selections of the same mode.
    static unsigned last_mode = 999;
    static unsigned consecutive_modes = 0;

    if (solver->heuristic == last_mode) {
        consecutive_modes++;
    } else {
        consecutive_modes = 1;
        last_mode = solver->heuristic;
    }

    if (consecutive_modes > 4) {
        r *= 0.6;
    }

    // Step 5: Update MAB Statistics
    // We map the continuous reward R (centered at 1.0) to a [0,1] probability 
    // to be compatible with Beta distribution updates (Bernoulli trials).
    // We scale by 0.5 (mapping [0,2] -> [0,1]) and clamp.
    double r_norm = r * 0.5;
    if (r_norm > 0.99) r_norm = 0.99;
    if (r_norm < 0.01) r_norm = 0.01;

    // Update cumulative reward (acting as Alpha in Beta distribution)
    // Note: mab_select (count) was already incremented when this phase started.
    solver->mab_reward[solver->heuristic] += r_norm;

    // Housekeeping: Reset phase counters for the next phase
    solver->mab_decisions = 0;
    solver->mab_conflicts = 0;
    solver->mab_chosen_tot = 0;
    for (all_variables(idx)) {
        solver->mab_chosen[idx] = 0;
    }

    // Step 6: Select Next Mode using Thompson Sampling on Beta Distributions
    unsigned best_arm = 0;
    double max_sample = -1e100;

    for (unsigned i = 0; i < solver->mab_heuristics; i++) {
        // Interpret statistics as Beta(alpha, beta) parameters
        // Alpha = Cumulative Reward + 1 (Prior)
        // Beta = (Total Counts - Cumulative Reward) + 1 (Prior)
        double alpha = solver->mab_reward[i] + 1.0;
        double count = (double)solver->mab_select[i];
        double beta = (count - solver->mab_reward[i]) + 1.0;

        // Safety check to ensure beta is valid
        if (beta < 1.0) beta = 1.0;

        // Normal Approximation of Beta(alpha, beta) for efficient sampling
        // Mean = alpha / (alpha + beta)
        // Variance = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))
        double sum_ab = alpha + beta;
        double mean = alpha / sum_ab;
        double var = (alpha * beta) / ((sum_ab * sum_ab) * (sum_ab + 1.0));
        double std_dev = sqrt(var);

        // Box-Muller Transform for Standard Normal Sample
        double u1 = kissat_pick_double(&solver->random);
        double u2 = kissat_pick_double(&solver->random);
        
        // Avoid log(0)
        if (u1 < 1e-9) u1 = 1e-9;

        double z = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265359 * u2);
        double sample = mean + z * std_dev;

        if (sample > max_sample) {
            max_sample = sample;
            best_arm = i;
        }
    }

    // Set the heuristic for the next phase
    solver->heuristic = best_arm;
    
    // Increment selection count for the new phase
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
