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
    // Step 1: Data Collection from the just-executed mode
    // We use the fast glue EMA as a proxy for the average LBD of the phase
    double avg_lbd = AVERAGE(fast_glue);

    // Calculate Conflict Rate (Conflicts / Decisions)
    double decisions = (double)solver->mab_decisions;
    double conflicts = (double)solver->mab_conflicts;
    if (decisions < 1.0) decisions = 1.0;
    double conflict_rate = conflicts / decisions;

    unsigned current_mode = solver->heuristic;

    // Static storage for history (N=5) and Quality Q
    static double history_lbd[5];
    static double history_cr[5]; // Conflict Rate history
    static int history_idx = 0;  // Current write position
    static int history_count = 0; // Number of items in history
    
    static double Q[2] = {0.0, 0.0}; // Quality for modes 0 and 1

    // Update History Circular Buffers
    history_lbd[history_idx] = avg_lbd;
    history_cr[history_idx] = conflict_rate;
    history_idx = (history_idx + 1) % 5;
    if (history_count < 5) history_count++;

    // Step 2: Update Quality Q
    // Immediate reward R = 1 / average_LBD
    double R = (avg_lbd > 1e-6) ? (1.0 / avg_lbd) : 0.0;
    
    // Q_mode = 0.6 * R + 0.4 * Q_mode
    Q[current_mode] = 0.6 * R + 0.4 * Q[current_mode];

    // Step 3 & 4: Calculate S (Slope) and V (Normalized StdDev)
    double S = 0.0;
    double V = 0.0;

    if (history_count >= 2) {
        double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
        double sum_cr = 0, sum_cr_sq = 0;
        int n = history_count;
        
        // Reconstruct chronological order from circular buffer
        // Oldest element is at 'start'
        int start = (history_count < 5) ? 0 : history_idx;

        for (int i = 0; i < n; i++) {
            int idx = (start + i) % 5;
            double x = (double)i;
            double y = history_lbd[idx];
            double cr = history_cr[idx];

            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;

            sum_cr += cr;
            sum_cr_sq += cr * cr;
        }

        // Linear Regression Slope S for LBD
        double denom = n * sum_xx - sum_x * sum_x;
        if (fabs(denom) > 1e-9) {
            S = (n * sum_xy - sum_x * sum_y) / denom;
        }

        // Normalized Standard Deviation V for Conflict Rate
        double mean_cr = sum_cr / n;
        double var_cr = (sum_cr_sq / n) - (mean_cr * mean_cr);
        double std_cr = (var_cr > 0.0) ? sqrt(var_cr) : 0.0;
        
        if (mean_cr > 1e-9) {
            V = std_cr / mean_cr;
        }
    }

    // Determine Next Mode
    unsigned next_heuristic = current_mode;

    // Momentum Override
    if (S < -0.25) {
        // Performance improving (LBD decreasing): deterministically select current
        next_heuristic = current_mode;
    } else if (S > 0.25) {
        // Performance degrading (LBD increasing): deterministically switch
        // Assuming binary choice (0 or 1)
        next_heuristic = 1 - current_mode;
    } else {
        // Stagnation (|S| <= 0.25): Adaptive Boltzmann Selection
        double T = 0.1 + V;
        
        // Calculate probabilities P(m) = exp(Q_m / T) / Sum(...)
        // Use softmax stability trick (subtract max Q)
        double q0 = Q[0];
        double q1 = Q[1];
        double max_q = (q0 > q1) ? q0 : q1;
        
        double exp0 = exp((q0 - max_q) / T);
        double exp1 = exp((q1 - max_q) / T);
        double sum_exp = exp0 + exp1;
        
        double p0 = exp0 / sum_exp;
        
        // Random selection
        double rand_val = kissat_pick_double(&solver->random);
        if (rand_val < p0) {
            next_heuristic = 0;
        } else {
            next_heuristic = 1;
        }
    }

    // Safety check for heuristic bounds
    if (next_heuristic >= solver->mab_heuristics) {
        next_heuristic = 0;
    }

    // Apply selection
    solver->heuristic = next_heuristic;
    solver->mab_select[solver->heuristic]++;

    // Reset solver MAB counters for the next phase
    solver->mab_decisions = 0;
    solver->mab_conflicts = 0;
    
    // Clear per-variable chosen counts
    for (all_variables(idx)) {
        solver->mab_chosen[idx] = 0;
    }
    solver->mab_chosen_tot = 0;
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
