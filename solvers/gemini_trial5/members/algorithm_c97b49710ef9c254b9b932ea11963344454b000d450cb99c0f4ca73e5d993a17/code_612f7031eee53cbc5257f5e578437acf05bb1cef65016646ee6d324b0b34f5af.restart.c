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
    // ------------------------------------------------------------------
    // 1. Update Rewards & MAB Housekeeping
    // ------------------------------------------------------------------

    double decisions = solver->mab_decisions;
    double conflicts = (double)solver->mab_conflicts;

    // Prevent log domain errors or division by zero
    if (decisions < 1.0) decisions = 1.0;
    if (conflicts < 1.0) conflicts = 1.0;

    // Calculate reward based on search efficiency (log scale decisions per conflict)
    // Matches baseline logic
    double reward = 0.0;
    if (conflicts > 1.0) {
        reward = log2(decisions) / log2(conflicts);
    } else {
        // If 0 or 1 conflicts, efficiency is based on decisions made
        reward = log2(decisions);
    }

    // Update cumulative reward for the heuristic that just finished
    solver->mab_reward[solver->heuristic] += reward;

    // Save conflicts for history update before resetting counters
    double phase_conflicts = conflicts;

    // Reset per-variable MAB tracking (Standard Kissat housekeeping)
    for (all_variables(idx)) {
        solver->mab_chosen[idx] = 0;
    }
    solver->mab_chosen_tot = 0;
    solver->mab_decisions = 0;
    solver->mab_conflicts = 0;

    // ------------------------------------------------------------------
    // 2. Algorithm: Inverse-Volatility Boltzmann with Hysteresis
    // ------------------------------------------------------------------

    // Static history buffers (Size N=5) to maintain state between restarts.
    // Note: 'static' preserves state across calls for this algorithm.
    static double lbd_history[5] = {0};
    static double conflict_history[5] = {0};
    static int history_idx = 0;   // Circular buffer write pointer
    static int history_count = 0; // Current number of valid entries

    // Step 1: Maintain circular buffer of average LBD and conflict rate
    double current_lbd = AVERAGE(fast_glue);
    
    // Insert into circular buffer
    // history_idx points to the oldest entry (which is overwritten next)
    lbd_history[history_idx] = current_lbd;
    conflict_history[history_idx] = phase_conflicts;
    
    // Advance pointer and count
    history_idx = (history_idx + 1) % 5;
    if (history_count < 5) history_count++;

    unsigned next_heuristic = solver->heuristic;

    // Only apply the advanced strategy if we have a full history window (N=5)
    if (history_count == 5) {
        
        // --- Calculate Slope S (LBD) and Volatility V (Conflicts) ---
        // We need to iterate chronologically. Since history_idx points to the 
        // next write pos (which is the oldest existing entry), the order is:
        // history_idx, history_idx+1, ..., history_idx+4 (modulo 5)
        
        double sum_x = 10.0;  // 0+1+2+3+4
        // double sum_x2 = 30.0; // 0+1+4+9+16
        double sum_y = 0.0;
        double sum_xy = 0.0;
        
        double sum_c = 0.0;
        double sum_c2 = 0.0;

        for (int i = 0; i < 5; i++) {
            int buffer_pos = (history_idx + i) % 5;
            double y = lbd_history[buffer_pos];
            double c = conflict_history[buffer_pos];
            double x = (double)i;

            sum_y += y;
            sum_xy += x * y;
            
            sum_c += c;
            sum_c2 += c * c;
        }

        // Calculate Linear Regression Slope S for LBD
        // Formula: (N * SumXY - SumX * SumY) / (N * SumX2 - SumX^2)
        // Denominator = 5 * 30 - 10 * 10 = 50
        double S = (5.0 * sum_xy - 10.0 * sum_y) / 50.0;

        // Calculate Normalized Standard Deviation V for Conflicts
        double mean_c = sum_c / 5.0;
        double var_c = (sum_c2 / 5.0) - (mean_c * mean_c);
        if (var_c < 0) var_c = 0; // Correct floating point inaccuracies
        double std_c = sqrt(var_c);
        double V = (mean_c > 1e-6) ? (std_c / mean_c) : 0.0;

        // --- Step 2: Compute average Quality 'Q' ---
        // Q = accumulated_score / number_of_selections
        double Q[2];
        // Heuristic 0 (VSIDS), 1 (CHB)
        for (int h = 0; h < 2; h++) {
            if (solver->mab_select[h] > 0) {
                Q[h] = solver->mab_reward[h] / (double)solver->mab_select[h];
            } else {
                Q[h] = 0.0; 
            }
        }

        // --- Step 3: Momentum Override ---
        if (S < -0.25) {
            // Performance improving (LBD decreasing): deterministically select current mode
            next_heuristic = solver->heuristic;
        } 
        else if (S > 0.25) {
            // Performance degrading (LBD increasing): deterministically switch to other mode
            next_heuristic = 1 - solver->heuristic;
        } 
        else {
            // --- Step 4: Stagnation (|S| <= 0.25) -> Inverse-Volatility Boltzmann ---
            
            // Temperature T = 0.5 / (V + 0.05)
            // Low volatility (stagnation) -> High Temperature (Exploration)
            double T = 0.5 / (V + 0.05);

            // Apply boredom penalty to the Quality Q of the currently active mode
            // Use temporary array for calculation to not corrupt global stats
            double Q_calc[2];
            Q_calc[0] = Q[0];
            Q_calc[1] = Q[1];
            
            Q_calc[solver->heuristic] *= 0.85;

            // Calculate Probabilities: P(m) = exp(Q_m / T) / Sum(exp(Q_k / T))
            // Optimization: subtract max exponent to avoid overflow
            double e0 = Q_calc[0] / T;
            double e1 = Q_calc[1] / T;
            double max_e = (e0 > e1) ? e0 : e1;
            
            double exp0 = exp(e0 - max_e);
            double exp1 = exp(e1 - max_e);
            double sum_exp = exp0 + exp1;
            
            double p0 = exp0 / sum_exp;
            
            // Select mode based on probability
            double r = kissat_pick_double(&solver->random);
            if (r < p0) {
                next_heuristic = 0;
            } else {
                next_heuristic = 1;
            }
        }

    } else {
        // Initialization/Warmup phase (History < 5)
        // Ensure both heuristics are sampled to establish baseline Q values
        if (solver->mab_select[0] == 0) next_heuristic = 0;
        else if (solver->mab_select[1] == 0) next_heuristic = 1;
        else {
            // Alternate during warmup
            next_heuristic = 1 - solver->heuristic;
        }
    }

    // Apply the selected heuristic
    solver->heuristic = next_heuristic;
    
    // Update selection count for the chosen heuristic (for the upcoming phase)
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
