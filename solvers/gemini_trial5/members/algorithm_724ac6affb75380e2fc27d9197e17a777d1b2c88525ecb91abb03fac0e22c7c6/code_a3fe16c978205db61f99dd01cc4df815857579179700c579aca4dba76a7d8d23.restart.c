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
    // -------------------------------------------------------------------------
    // 1. Update Rewards (Standard MAB Maintenance)
    // -------------------------------------------------------------------------

    // Calculate score for the phase just finished based on log ratio of decisions/conflicts
    double decisions = solver->mab_decisions;
    double conflicts = (double)solver->mab_conflicts;
    double score = 0;

    if (conflicts > 0 && decisions > 0) {
        score = log2(decisions) / log2(conflicts);
    }

    // Update cumulative reward for the heuristic that just ran
    solver->mab_reward[solver->heuristic] += score;

    // Reset per-phase counters for the next phase
    for (all_variables(idx)) {
        solver->mab_chosen[idx] = 0;
    }
    solver->mab_chosen_tot = 0;
    solver->mab_decisions = 0;
    solver->mab_conflicts = 0;

    // -------------------------------------------------------------------------
    // 2. Statistics & History Maintenance
    // -------------------------------------------------------------------------

    // Circular buffers for history (size N=5)
    static double lbd_history[5] = {0};
    static double rate_history[5] = {0};
    static int history_idx = 0;       // Next insertion index
    static int history_count = 0;     // Current number of valid items in buffer

    // Gather current phase metrics
    // Use fast EMA of glue (LBD) as the performance metric for LBD history
    double current_lbd = AVERAGE(fast_glue);
    // Conflict rate: conflicts per decision
    double current_rate = (decisions > 0) ? (conflicts / decisions) : 0;

    // Update history buffers
    lbd_history[history_idx] = current_lbd;
    rate_history[history_idx] = current_rate;

    history_idx = (history_idx + 1) % 5;
    if (history_count < 5) history_count++;

    // -------------------------------------------------------------------------
    // 3. Algorithm Implementation: Inverse-Slope Entropy Switching
    // -------------------------------------------------------------------------

    double slope_S = 0.0;
    double V = 0.0;

    // Step 1: Statistics Calculation
    // Only calculate regression if we have a full window. 
    // If incomplete, slope_S remains 0.0 (Stagnation), triggering exploration.
    if (history_count == 5) {
        // Reconstruct history in chronological order
        // history_idx points to the oldest element (next to be overwritten)
        double y[5];
        double x[5] = {0, 1, 2, 3, 4};

        for (int i = 0; i < 5; i++) {
            y[i] = lbd_history[(history_idx + i) % 5];
        }

        // Linear Regression Slope S of LBD history
        double sum_x = 10.0; // 0+1+2+3+4
        double sum_y = 0.0;
        double sum_xy = 0.0;

        for (int i = 0; i < 5; i++) {
            sum_y += y[i];
            sum_xy += x[i] * y[i];
        }

        // Denominator: N * sum_x2 - (sum_x)^2 = 5 * 30 - 100 = 50
        slope_S = (5.0 * sum_xy - sum_x * sum_y) / 50.0;

        // Normalized Standard Deviation V of conflict rate history
        double r_sum = 0;
        for (int i = 0; i < 5; i++) r_sum += rate_history[i];
        double r_mean = r_sum / 5.0;

        double r_sq_diff = 0;
        for (int i = 0; i < 5; i++) {
            double diff = rate_history[i] - r_mean;
            r_sq_diff += diff * diff;
        }
        double r_std = sqrt(r_sq_diff / 5.0);

        if (r_mean > 1e-9) V = r_std / r_mean;
        (void)V; // Suppress unused variable warning (V calculated per spec but unused in selection)
    }

    // Step 2: Compute Average Quality Q
    unsigned num_heuristics = solver->mab_heuristics;
    if (num_heuristics > 2) num_heuristics = 2; // Safety cap, typically 2 (VSIDS/CHB)

    double Q[2] = {0};
    for (unsigned i = 0; i < num_heuristics; i++) {
        if (solver->mab_select[i] > 0) {
            Q[i] = solver->mab_reward[i] / solver->mab_select[i];
        } else {
            // Unexplored arms get high priority
            Q[i] = 1000.0; 
        }
    }

    unsigned next_heuristic = solver->heuristic;

    // Step 3: Momentum Override
    if (slope_S < -0.25) {
        // Performance improving (LBD decreasing): Deterministically select current mode
        next_heuristic = solver->heuristic;
    } else if (slope_S > 0.25) {
        // Performance degrading (LBD increasing): Deterministically switch to other mode
        next_heuristic = (solver->heuristic + 1) % num_heuristics;
    } else {
        // Step 4: Inverse-Slope Boltzmann Selection (|S| <= 0.25)
        // Stagnation detected
        
        // Ensure unselected arms are tried first
        bool unselected = false;
        for (unsigned i = 0; i < num_heuristics; i++) {
            if (solver->mab_select[i] == 0) {
                next_heuristic = i;
                unselected = true;
                break;
            }
        }

        if (!unselected) {
            // Calculate Temperature T
            double abs_S = fabs(slope_S);
            double T = 0.1 + ((0.25 - abs_S) * 4.0);

            // Calculate Probabilities: P(m) = exp(Q_m / T) / Sum
            double exp_vals[2];
            double sum_exp = 0.0;

            for (unsigned i = 0; i < num_heuristics; i++) {
                exp_vals[i] = exp(Q[i] / T);
                sum_exp += exp_vals[i];
            }

            // Select based on probability
            double r = kissat_pick_double(&solver->random);
            double cumulative = 0.0;

            for (unsigned i = 0; i < num_heuristics; i++) {
                double p = exp_vals[i] / sum_exp;
                cumulative += p;
                if (r < cumulative) {
                    next_heuristic = i;
                    break;
                }
            }
        }
    }

    // Apply selection and increment count for the upcoming phase
    solver->heuristic = next_heuristic;
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
