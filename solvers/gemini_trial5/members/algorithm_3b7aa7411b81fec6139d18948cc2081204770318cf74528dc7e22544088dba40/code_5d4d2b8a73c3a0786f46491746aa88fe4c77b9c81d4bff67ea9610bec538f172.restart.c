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
    // Implementation of Volatility-Gated Momentum Switching

    // --- 1. Update Rewards (Baseline Logic) ---
    // Calculate reward based on decisions and conflicts since last update.
    // We use the baseline formula: log2(decisions) / log2(conflicts).
    double decisions = (double)solver->mab_decisions;
    double conflicts = (double)solver->mab_conflicts;
    
    double reward = 0.0;
    // Protect against division by zero and log domain errors
    if (conflicts > 1.0 && decisions > 0.0) {
        reward = log2(decisions) / log2(conflicts);
    }
    
    // Update the cumulative reward for the current heuristic
    solver->mab_reward[solver->heuristic] += reward;

    // --- 2. Reset MAB Counters ---
    // Standard cleanup for the next phase
    for (all_variables(idx)) {
        solver->mab_chosen[idx] = 0;
    }
    solver->mab_chosen_tot = 0;
    solver->mab_decisions = 0;
    solver->mab_conflicts = 0;

    // --- 3. Maintain History (N=5) ---
    // We use static variables to maintain history across restarts since we cannot modify the solver struct.
    static double history_lbd[5];
    static double history_cr[5];
    static int history_idx = 0;
    static int history_count = 0;

    // Capture current metrics
    // LBD: Use fast moving average of glue
    double current_lbd = AVERAGE(fast_glue);
    // Conflict Rate: Conflicts per decision in the last phase
    double current_cr = (decisions > 0.0) ? (conflicts / decisions) : 0.0;

    history_lbd[history_idx] = current_lbd;
    history_cr[history_idx] = current_cr;

    history_idx = (history_idx + 1) % 5;
    if (history_count < 5) history_count++;

    // --- 4. Calculate Metrics & Select Mode ---
    unsigned next_heuristic = solver->heuristic;
    
    // Only apply the algorithm if we have a full history buffer
    if (history_count == 5) {
        // Unroll circular buffer to logical order: 0 (oldest) to 4 (newest)
        double y_lbd[5];
        double y_cr[5];
        
        // history_idx points to the oldest element (next to be overwritten)
        for (int i = 0; i < 5; i++) {
            int idx = (history_idx + i) % 5;
            y_lbd[i] = history_lbd[idx];
            y_cr[i] = history_cr[idx];
        }

        // Step 1a: Linear Regression Slope 'S' of LBD
        // x = {0, 1, 2, 3, 4}
        // Formula: S = (N*Sum(xy) - Sum(x)*Sum(y)) / (N*Sum(x^2) - Sum(x)^2)
        double sum_x = 10.0;   // 0+1+2+3+4
        double sum_x2 = 30.0;  // 0+1+4+9+16
        double sum_y = 0.0;
        double sum_xy = 0.0;
        
        for (int i = 0; i < 5; i++) {
            sum_y += y_lbd[i];
            sum_xy += i * y_lbd[i];
        }
        
        // Denominator = 5*30 - 100 = 50
        double S = (5.0 * sum_xy - sum_x * sum_y) / 50.0;

        // Step 1b: Normalized Standard Deviation 'V' of Conflict Rate
        double sum_cr = 0.0;
        for (int i = 0; i < 5; i++) sum_cr += y_cr[i];
        double mean_cr = sum_cr / 5.0;
        
        double sum_sq_diff = 0.0;
        for (int i = 0; i < 5; i++) {
            double diff = y_cr[i] - mean_cr;
            sum_sq_diff += diff * diff;
        }
        
        double V = 0.0;
        if (mean_cr > 1e-9) {
            double std_dev = sqrt(sum_sq_diff / 5.0);
            V = std_dev / mean_cr;
        }

        // Step 2: Average Quality 'Q'
        // Q = accumulated_score / number_of_selections
        double Q[2] = {0.0, 0.0};
        // Assuming binary heuristics (0 and 1) representing Stable and Focused strategies
        for (unsigned i = 0; i < 2 && i < solver->mab_heuristics; i++) {
            if (solver->mab_select[i] > 0) {
                Q[i] = solver->mab_reward[i] / solver->mab_select[i];
            }
        }

        // Step 3: Noise-Aware Momentum
        bool deterministic = false;
        
        if (S < -0.15) {
            // Performance improving (Slope negative): preserve locality
            next_heuristic = solver->heuristic;
            deterministic = true;
        } else if (S > 0.30) {
            // Performance degrading (Slope positive)
            if (V < 0.15) {
                // Low volatility (clear signal): switch mode
                next_heuristic = (solver->heuristic == 0) ? 1 : 0;
                deterministic = true;
            }
            // else: high volatility, ambiguous -> Fall through to Step 4
        }

        // Step 4: Adaptive Boltzmann Selection
        if (!deterministic) {
            double T = 0.1 + V;
            
            double exp0 = exp(Q[0] / T);
            double exp1 = exp(Q[1] / T);
            double sum_exp = exp0 + exp1;
            
            if (sum_exp > 0.0) {
                double p0 = exp0 / sum_exp;
                double rand_val = kissat_pick_double(&solver->random);
                if (rand_val < p0) {
                    next_heuristic = 0;
                } else {
                    next_heuristic = 1;
                }
            } else {
                // Fallback if math fails
                next_heuristic = solver->heuristic;
            }
        }
    } else {
        // Warmup phase: Simple alternation to gather initial data for both modes
        next_heuristic = (solver->heuristic == 0) ? 1 : 0;
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
