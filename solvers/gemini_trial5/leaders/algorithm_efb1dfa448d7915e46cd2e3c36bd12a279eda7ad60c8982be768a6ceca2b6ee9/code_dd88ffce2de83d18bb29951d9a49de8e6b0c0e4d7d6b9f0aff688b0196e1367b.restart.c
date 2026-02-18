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
    // Capture metrics for the phase that just finished
    double current_lbd = AVERAGE(fast_glue);
    double current_rate = 0.0;
    if (solver->mab_decisions > 0) {
        current_rate = (double)solver->mab_conflicts / solver->mab_decisions;
    }

    // Update MAB Reward (accumulated score)
    // Using baseline reward formula: log2(decisions) / log2(conflicts)
    if (solver->mab_conflicts > 1 && solver->mab_decisions > 0) {
        double score = log2(solver->mab_decisions) / log2((double)solver->mab_conflicts);
        solver->mab_reward[solver->heuristic] += score;
    }

    // Reset per-variable MAB data and counters
    for (all_variables(idx)) {
        solver->mab_chosen[idx] = 0;
    }
    solver->mab_chosen_tot = 0;
    solver->mab_decisions = 0;
    solver->mab_conflicts = 0;

    // --- Step 1: Maintain circular buffer of average LBD and conflict rate (N=5) ---
    static double lbd_history[5];
    static double rate_history[5];
    static int history_idx = 0;
    static int history_count = 0;
    const int N = 5;

    lbd_history[history_idx] = current_lbd;
    rate_history[history_idx] = current_rate;
    history_idx = (history_idx + 1) % N;
    if (history_count < N) history_count++;

    // Initialization check: ensure both heuristics have been tried at least once
    unsigned stable_restarts = 0;
    for (unsigned i = 0; i < solver->mab_heuristics; i++) {
        stable_restarts += solver->mab_select[i];
    }
    
    if (stable_restarts < solver->mab_heuristics) {
        solver->heuristic = (solver->heuristic + 1) % solver->mab_heuristics;
        solver->mab_select[solver->heuristic]++;
        return;
    }

    // Calculate Slope S (Linear Regression on LBD history)
    double S = 0.0;
    if (history_count >= 2) {
        double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
        for (int i = 0; i < history_count; i++) {
            // Reconstruct chronological order: x=0 is oldest
            int idx = (history_idx - history_count + i + N) % N;
            double x = i;
            double y = lbd_history[idx];
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }
        double n = (double)history_count;
        double denom = n * sum_xx - sum_x * sum_x;
        if (denom != 0) {
            S = (n * sum_xy - sum_x * sum_y) / denom;
        }
    }

    // Calculate Normalized Std Dev V of conflict rate history
    double V = 0.0;
    if (history_count >= 2) {
        double sum_val = 0;
        for (int i = 0; i < history_count; i++) sum_val += rate_history[i];
        double mean = sum_val / history_count;
        
        double sum_sq_diff = 0;
        for (int i = 0; i < history_count; i++) {
            double diff = rate_history[i] - mean;
            sum_sq_diff += diff * diff;
        }
        
        if (mean > 1e-9) {
            double variance = sum_sq_diff / history_count;
            V = sqrt(variance) / mean;
        }
    }

    // --- Step 2: Compute Quality Q ---
    // Q = accumulated_score / number_of_selections
    // We assume standard setup with 2 heuristics (VSIDS and CHB)
    
    unsigned next_heuristic = solver->heuristic;

    // --- Step 3: Momentum Override ---
    if (S < -0.25) {
        // Performance improving -> Deterministically select current mode
        next_heuristic = solver->heuristic;
    } else if (S > 0.25) {
        // Performance degrading -> Deterministically switch to other mode
        // Assuming binary choice (0 vs 1)
        if (solver->mab_heuristics == 2) {
            next_heuristic = 1 - solver->heuristic;
        } else {
            next_heuristic = (solver->heuristic + 1) % solver->mab_heuristics;
        }
    } else {
        // --- Step 4: Adaptive Boltzmann Selection (|S| <= 0.25) ---
        double T = 0.1 + V;
        
        // Calculate Q values and find max for numerical stability
        double Q[2] = {0.0, 0.0};
        double max_exponent = -1e9;
        unsigned limit = (solver->mab_heuristics < 2) ? solver->mab_heuristics : 2;

        for (unsigned i = 0; i < limit; i++) {
            if (solver->mab_select[i] > 0) {
                Q[i] = solver->mab_reward[i] / solver->mab_select[i];
            } else {
                Q[i] = 0.0;
            }
            if (Q[i] / T > max_exponent) max_exponent = Q[i] / T;
        }

        // Calculate Probabilities P(m)
        double exp_val[2] = {0.0, 0.0};
        double sum_exp = 0.0;
        
        for (unsigned i = 0; i < limit; i++) {
            exp_val[i] = exp(Q[i] / T - max_exponent);
            sum_exp += exp_val[i];
        }

        // Select based on probability
        if (sum_exp > 0) {
            double r = kissat_pick_double(&solver->random);
            double cumulative = 0.0;
            bool picked = false;
            for (unsigned i = 0; i < limit; i++) {
                cumulative += exp_val[i] / sum_exp;
                if (r < cumulative) {
                    next_heuristic = i;
                    picked = true;
                    break;
                }
            }
            if (!picked) next_heuristic = limit - 1;
        }
    }

    // Apply selection
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
