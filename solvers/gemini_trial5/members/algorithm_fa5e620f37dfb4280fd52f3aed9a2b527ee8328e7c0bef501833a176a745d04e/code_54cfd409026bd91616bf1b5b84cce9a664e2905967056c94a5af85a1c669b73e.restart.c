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
    // L: Average LBD of the phase. Using fast_glue EMA as the proxy for the phase average.
    double L = AVERAGE(fast_glue);
    if (L < 1.0) L = 1.0; // Safety floor

    // D: Average Decision Level. Using level EMA.
    double D = AVERAGE(level);
    
    // N: Total number of variables
    double N = (double)solver->vars;
    
    // d_ratio: Normalized depth ratio
    double d_ratio = (N > 0.0) ? (D / N) : 0.0;

    // Step 2: Base Reward Signal
    double R = 1.0 / L;

    // Step 3: Apply Asymmetric Incentives
    // "If the completed phase was STABLE..."
    // We map Heuristic 0 to the "Stable" strategy and Heuristic 1 to "Focused".
    if (solver->heuristic == 0) {
        R *= (1.0 + (1.5 * d_ratio));
    }

    // Step 4: Apply Progressive Fatigue
    // We track the consecutive mode streak using static variables since we cannot add fields to the solver.
    static unsigned last_heuristic = 0;
    static unsigned streak = 0;
    static bool initialized = false;

    if (!initialized) {
        last_heuristic = solver->heuristic;
        streak = 1;
        initialized = true;
    } else {
        if (solver->heuristic == last_heuristic) {
            streak++;
        } else {
            streak = 1;
            last_heuristic = solver->heuristic;
        }
    }

    // Calculate penalty P = 0.85 ^ max(0, consecutive_modes - 3)
    double penalty_exponent = (streak > 3) ? (double)(streak - 3) : 0.0;
    double P = pow(0.85, penalty_exponent);
    
    // Modify R -> R * P
    R *= P;

    // Clamp R to [0, 1] to ensure statistical stability for Beta distribution logic
    // (Ensures cumulative reward <= cumulative count)
    if (R > 1.0) R = 1.0;
    if (R < 0.0) R = 0.0;

    // Step 5: Update MAB Statistics for the current arm
    solver->mab_reward[solver->heuristic] += R;
    solver->mab_select[solver->heuristic] += 1;

    // Reset phase-specific counters
    solver->mab_decisions = 0;
    solver->mab_conflicts = 0;

    // Step 6: Select Next Mode using Thompson Sampling
    // We model the arm rewards using Beta distributions and select the arm with the highest sample.
    unsigned best_arm = 0;
    double max_sample = -1.0;
    const double PI = 3.14159265358979323846;

    for (unsigned i = 0; i < solver->mab_heuristics; i++) {
        double count = (double)solver->mab_select[i];
        double reward = solver->mab_reward[i];
        
        // Safety check to ensure valid Beta parameters
        if (reward > count) reward = count;

        // Beta(alpha, beta) with Prior(1, 1)
        double alpha = reward + 1.0;
        double beta = (count - reward) + 1.0;

        // Sample from Beta(alpha, beta) using Normal Approximation
        // Mean = alpha / (alpha + beta)
        // Variance = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))
        double sum = alpha + beta;
        double mean = alpha / sum;
        double variance = (alpha * beta) / (pow(sum, 2) * (sum + 1.0));
        double std_dev = sqrt(variance);

        // Generate Standard Normal sample using Box-Muller Transform
        double u1 = kissat_pick_double(&solver->random);
        double u2 = kissat_pick_double(&solver->random);
        
        // Handle potential log(0)
        if (u1 < 1e-9) u1 = 1e-9;

        double z = sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
        
        // Transform to Beta sample
        double sample = mean + z * std_dev;

        if (sample > max_sample) {
            max_sample = sample;
            best_arm = i;
        }
    }

    // Set the heuristic for the next phase
    solver->heuristic = best_arm;
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
