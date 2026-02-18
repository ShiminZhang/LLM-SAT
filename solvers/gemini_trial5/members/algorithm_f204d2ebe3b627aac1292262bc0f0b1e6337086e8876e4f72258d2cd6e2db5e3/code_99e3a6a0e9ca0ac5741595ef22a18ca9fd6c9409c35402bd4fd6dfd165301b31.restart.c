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
    // 1. Update Rewards (Standard MAB Bookkeeping)
    // ------------------------------------------------------------------
    
    double reward = 0.0;
    // Prevent division by zero or log error
    if (solver->mab_conflicts > 1) {
        double num = (solver->mab_decisions > 1) ? log2(solver->mab_decisions) : 0.0;
        double den = log2(solver->mab_conflicts);
        reward = num / den;
    } else if (solver->mab_decisions > 0) {
        // Fallback for high efficiency (0 or 1 conflicts)
        reward = 10.0; 
    }

    // Update cumulative reward for the heuristic that just ran
    solver->mab_reward[solver->heuristic] += reward;
    
    // Reset MAB counters for the next phase
    for (all_variables(idx)) {
        solver->mab_chosen[idx] = 0;
    }
    solver->mab_chosen_tot = 0;
    solver->mab_decisions = 0;
    solver->mab_conflicts = 0;

    // ------------------------------------------------------------------
    // 2. Algorithm: Stagnation-Boosted Momentum Switching
    // ------------------------------------------------------------------

    // Static history buffers (Circular) N=5
    static double hist_lbd[5] = {0};
    static double hist_cr[5] = {0};
    static int hist_idx = 0;
    static int hist_cnt = 0;
    
    // Static state for rate calculation
    static uint64_t last_conflicts = 0;
    static uint64_t last_decisions = 0;

    // --- Step 1: Metrics & History ---

    // Current LBD (using fast glue EMA as proxy for recent LBD performance)
    double current_lbd = AVERAGE(fast_glue);
    
    // Current Conflict Rate calculation
    uint64_t curr_conflicts = CONFLICTS;
    uint64_t curr_decisions = DECISIONS;
    
    double d_conflicts = (double)(curr_conflicts - last_conflicts);
    double d_decisions = (double)(curr_decisions - last_decisions);
    
    // Update tracking variables
    last_conflicts = curr_conflicts;
    last_decisions = curr_decisions;

    double current_cr = 0.0;
    if (d_decisions > 0) {
        current_cr = d_conflicts / d_decisions;
    }

    // Update circular buffers
    hist_lbd[hist_idx] = current_lbd;
    hist_cr[hist_idx] = current_cr;
    hist_idx = (hist_idx + 1) % 5;
    if (hist_cnt < 5) hist_cnt++;

    // Ensure initialization of all heuristics before applying advanced logic
    if (solver->mab_select[0] == 0) {
        solver->heuristic = 0;
        solver->mab_select[0]++;
        return;
    }
    if (solver->mab_select[1] == 0) {
        solver->heuristic = 1;
        solver->mab_select[1]++;
        return;
    }

    // If history not full, stick to current
    if (hist_cnt < 5) {
        solver->mab_select[solver->heuristic]++;
        return;
    }

    // --- Compute Slope S of LBD History ---
    // Unroll circular buffer to chronological order: y[0] is oldest
    double y[5];
    for (int i = 0; i < 5; i++) {
        y[i] = hist_lbd[(hist_idx + i) % 5];
    }

    // Linear regression slope S for x={0,1,2,3,4}
    // S = (5 * Sum(xy) - 10 * Sum(y)) / 50
    double sum_y = 0.0;
    double sum_xy = 0.0;
    for (int i = 0; i < 5; i++) {
        sum_y += y[i];
        sum_xy += i * y[i];
    }
    double slope_s = (5.0 * sum_xy - 10.0 * sum_y) / 50.0;

    // --- Compute Volatility V of Conflict Rate ---
    double sum_cr = 0.0;
    for (int i = 0; i < 5; i++) sum_cr += hist_cr[i];
    double mean_cr = sum_cr / 5.0;

    double sum_sq_diff = 0.0;
    for (int i = 0; i < 5; i++) {
        double diff = hist_cr[i] - mean_cr;
        sum_sq_diff += diff * diff;
    }
    double std_dev = sqrt(sum_sq_diff / 5.0);
    double volatility_v = (mean_cr > 1e-9) ? (std_dev / mean_cr) : 0.0;

    // --- Step 2: Average Quality Q ---
    double Q[2];
    for (unsigned i = 0; i < 2; i++) {
        // Safe division guarded by initialization checks above
        Q[i] = solver->mab_reward[i] / solver->mab_select[i];
    }

    // --- Step 3: Momentum Override ---
    unsigned current_mode = solver->heuristic;
    unsigned next_mode = current_mode;

    if (slope_s < -0.25) {
        // Improving (Slope < -0.25) -> Deterministically select current
        next_mode = current_mode;
    } else if (slope_s > 0.25) {
        // Degrading (Slope > 0.25) -> Deterministically switch
        next_mode = 1 - current_mode;
    } else {
        // --- Step 4: Stagnation Handling (|S| <= 0.25) ---
        
        // Stagnation Severity lambda = 1 - (|S| / 0.25)
        double abs_s = (slope_s < 0) ? -slope_s : slope_s;
        double lambda = 1.0 - (abs_s / 0.25);
        
        // Temperature T = 0.1 + V + (0.5 * lambda)
        double T = 0.1 + volatility_v + (0.5 * lambda);
        
        // Boredom Penalty: Q'_c = 0.9 * Q_c
        double Q_prime[2];
        Q_prime[current_mode] = 0.9 * Q[current_mode];
        Q_prime[1 - current_mode] = Q[1 - current_mode];
        
        // Softmax Selection: P(m) = exp(Q'_m / T) / Sum
        // Use log-sum-exp stabilization
        double max_q = (Q_prime[0] > Q_prime[1]) ? Q_prime[0] : Q_prime[1];
        
        double exp0 = exp((Q_prime[0] - max_q) / T);
        double exp1 = exp((Q_prime[1] - max_q) / T);
        double sum_exp = exp0 + exp1;
        
        double p0 = exp0 / sum_exp;
        
        // Random selection
        double r = kissat_pick_double(&solver->random);
        if (r < p0) next_mode = 0;
        else next_mode = 1;
    }

    // Apply selection
    solver->heuristic = next_mode;
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
