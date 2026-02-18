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

    // --- Standard MAB Reward Update ---
    // Calculate score for the phase just finished using baseline metric: log2(decisions)/log2(conflicts)
    double reward = 0;
    if (solver->mab_conflicts > 0) {
        double n = (double)solver->mab_decisions;
        double k = (double)solver->mab_conflicts;
        // Clamp to avoid log(0) or log(1) -> 0 division issues
        if (n < 2.0) n = 2.0;
        if (k < 2.0) k = 2.0;
        reward = log2(n) / log2(k);
    }
    solver->mab_reward[solver->heuristic] += reward;

    // --- Step 1: Metrics Extraction ---
    // Extract current phase metrics before resetting counters
    // LBD Proxy: Fast Exponential Moving Average of Glue (available via macro)
    double current_lbd = AVERAGE(fast_glue);
    
    // Conflict Rate: Conflicts per Decision
    double current_rate = 0.0;
    if (solver->mab_decisions > 0) {
        current_rate = (double)solver->mab_conflicts / (double)solver->mab_decisions;
    }

    // --- Reset MAB Counters for next phase ---
    for (all_variables(idx)) {
        solver->mab_chosen[idx] = 0;
    }
    solver->mab_chosen_tot = 0;
    solver->mab_decisions = 0;
    solver->mab_conflicts = 0;

    // --- Robust Median-Filtered Momentum Strategy Implementation ---

    // Static history buffers (Circular Buffer size N=12)
    // Note: Using static variables as we cannot modify the solver struct fields.
    static double hist_lbd[12];
    static double hist_rate[12];
    static int hist_count = 0;
    static int hist_head = 0;

    // Update Circular Buffers
    hist_lbd[hist_head] = current_lbd;
    hist_rate[hist_head] = current_rate;
    hist_head = (hist_head + 1) % 12;
    if (hist_count < 12) hist_count++;

    // Initialization phase: if not enough history, use simple alternation
    if (hist_count < 3) {
        solver->heuristic = 1 - solver->heuristic;
        solver->mab_select[solver->heuristic]++;
        return;
    }

    // Linearize history for processing
    double lbd_linear[12];
    double rate_linear[12];
    for (int i = 0; i < hist_count; i++) {
        int idx = (hist_head - hist_count + i + 12) % 12;
        lbd_linear[i] = hist_lbd[idx];
        rate_linear[i] = hist_rate[idx];
    }

    // Step 1a: Median-of-3 Smoothing on LBD
    double lbd_smoothed[12];
    // Preserve endpoints
    lbd_smoothed[0] = lbd_linear[0];
    lbd_smoothed[hist_count - 1] = lbd_linear[hist_count - 1];
    
    for (int i = 1; i < hist_count - 1; i++) {
        double a = lbd_linear[i - 1];
        double b = lbd_linear[i];
        double c = lbd_linear[i + 1];
        // Median calculation
        double m = b;
        if (a > b) {
            if (b > c) m = b;
            else if (a > c) m = c;
            else m = a;
        } else {
            if (b < c) m = b;
            else if (a < c) m = c;
            else m = a;
        }
        lbd_smoothed[i] = m;
    }

    // Step 1b: Linear Regression Slope 'S' on Smoothed LBD
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    double n_val = (double)hist_count;
    
    for (int i = 0; i < hist_count; i++) {
        double x = (double)i;
        double y = lbd_smoothed[i];
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
    }
    
    double slope_s = 0.0;
    double denom = (n_val * sum_xx - sum_x * sum_x);
    if (denom != 0.0) {
        slope_s = (n_val * sum_xy - sum_x * sum_y) / denom;
    }

    // Step 1c: Coefficient of Variation 'V' on Conflict Rate
    double rate_sum = 0;
    for (int i = 0; i < hist_count; i++) rate_sum += rate_linear[i];
    double rate_mean = rate_sum / n_val;
    
    double rate_var_sum = 0;
    for (int i = 0; i < hist_count; i++) {
        double d = rate_linear[i] - rate_mean;
        rate_var_sum += d * d;
    }
    double rate_std = sqrt(rate_var_sum / n_val);
    
    // Calculate V (prevent division by zero)
    double v_val = (rate_mean > 1e-9) ? (rate_std / rate_mean) : 0.0;

    // --- Step 2: Compute Average Quality 'Q' ---
    // Q = accumulated_score / number_of_selections
    double q[2];
    // Assuming 2 heuristics: 0 (VSIDS) and 1 (CHB)
    for (unsigned i = 0; i < 2; i++) {
        if (solver->mab_select[i] > 0) {
            q[i] = solver->mab_reward[i] / solver->mab_select[i];
        } else {
            q[i] = 1.0; // Optimistic initialization for unselected heuristics
        }
    }

    // --- Step 3 & 4: Selection Logic ---
    unsigned next_heuristic = solver->heuristic;

    if (slope_s < -0.25) {
        // Step 3: Performance Improving (LBD decreasing) -> Deterministically select current mode
        next_heuristic = solver->heuristic;
    } else if (slope_s > 0.25) {
        // Step 3: Performance Degrading (LBD increasing) -> Deterministically switch mode
        next_heuristic = 1 - solver->heuristic;
    } else {
        // Step 4: Stagnation (|S| <= 0.25) -> Adaptive Boltzmann Selection
        double t = 0.1 + v_val;
        
        // Calculate exponentials: exp(Q/T)
        double e0 = exp(q[0] / t);
        double e1 = exp(q[1] / t);
        
        // Safety against overflow/infinity
        if (isinf(e0) && isinf(e1)) { e0 = 1.0; e1 = 1.0; }
        else if (isinf(e0)) { e0 = 1.0; e1 = 0.0; }
        else if (isinf(e1)) { e0 = 0.0; e1 = 1.0; }
        
        double sum_e = e0 + e1;
        double p0 = 0.5;
        if (sum_e > 0.0) {
            p0 = e0 / sum_e;
        }
        
        double rnd = kissat_pick_double(&solver->random);
        if (rnd < p0) {
            next_heuristic = 0;
        } else {
            next_heuristic = 1;
        }
    }

    // Apply Selection
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
