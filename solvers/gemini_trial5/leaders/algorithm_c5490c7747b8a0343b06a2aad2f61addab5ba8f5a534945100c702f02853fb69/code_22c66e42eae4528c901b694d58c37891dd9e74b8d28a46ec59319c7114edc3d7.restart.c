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
    // Step 1: Compute Metrics
    // L: Average LBD of the phase. Using fast_glue EMA as proxy for phase average.
    double L = AVERAGE(fast_glue);
    if (L < 1.0) L = 1.0;

    // D: Average decision level.
    double D = AVERAGE(level);

    // N: Number of variables
    unsigned N = solver->vars;

    // d_ratio: Depth ratio (D / N)
    double d_ratio = (N > 0) ? (D / (double)N) : 0.0;

    // Step 2: Base Reward R = 1 / L
    double R = 1.0 / L;

    // Step 3: Apply Asymmetric Incentives
    // Arm 0 is treated as the "Stable" strategy (VSIDS), Arm 1 as "Focused" (CHB).
    // If the completed phase was STABLE (heuristic 0), boost reward based on depth.
    if (solver->heuristic == 0) {
        R = R * (1.0 + (1.5 * d_ratio));
    }

    // Step 4: Apply Fatigue Penalty
    // Track consecutive usage of the same mode.
    // Note: Using static variables as we cannot modify the solver structure.
    static unsigned consecutive_modes = 0;
    static unsigned last_heuristic = 999; // Sentinel value

    if (solver->heuristic == last_heuristic) {
        consecutive_modes++;
    } else {
        consecutive_modes = 1;
        last_heuristic = solver->heuristic;
    }

    // If current mode chosen > 4 times in a row, decay reward.
    if (consecutive_modes > 4) {
        R = R * 0.6;
    }

    // Clamp R to [0, 1] for Beta distribution compatibility (Bernoulli trial logic)
    if (R > 1.0) R = 1.0;
    if (R < 0.0) R = 0.0;

    // Step 5: Update MAB Statistics
    // solver->mab_reward accumulates successes (proxy for alpha)
    // solver->mab_select accumulates total trials
    solver->mab_reward[solver->heuristic] += R;
    solver->mab_select[solver->heuristic] += 1;

    // Reset phase-specific tracking variables
    solver->mab_decisions = 0;
    solver->mab_conflicts = 0;
    solver->mab_chosen_tot = 0;
    for (all_variables(idx)) {
        solver->mab_chosen[idx] = 0;
    }

    // Step 6: Select next mode using Thompson Sampling
    // We approximate the Beta(alpha, beta) distribution with a Normal distribution
    // for efficiency and standard library compliance.
    int best_arm = 0;
    double max_sample = -1000.0;

    for (unsigned i = 0; i < solver->mab_heuristics; i++) {
        // Calculate Beta parameters
        // Prior is Beta(1,1), so add 1 to counts
        double successes = solver->mab_reward[i];
        double trials = (double)solver->mab_select[i];
        
        double alpha = successes + 1.0;
        double beta = (trials - successes) + 1.0;
        
        // Normal Approximation of Beta
        double sum = alpha + beta;
        double mean = alpha / sum;
        // Variance = (alpha * beta) / ((alpha+beta)^2 * (alpha+beta+1))
        double var = (alpha * beta) / (pow(sum, 2) * (sum + 1.0));
        double std_dev = sqrt(var);

        // Box-Muller Transform for Normal Sampling
        double u1 = kissat_pick_double(&solver->random);
        double u2 = kissat_pick_double(&solver->random);
        
        // Avoid log(0)
        if (u1 < 1e-10) u1 = 1e-10;
        
        // Generate sample
        double z = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
        double sample = mean + std_dev * z;

        if (sample > max_sample) {
            max_sample = sample;
            best_arm = i;
        }
    }

    // Update the solver's heuristic for the next phase
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
