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
    // Static variables for Momentum-Driven Softmax Bandit
    // These persist across restart calls to maintain state
    static double fast_ema[2] = {0};
    static double slow_ema[2] = {0};
    static uint64_t last_glue_low_count = 0;
    static int stagnation_counter = 0;
    static bool initialized = false;

    // Constants defined by the algorithm
    const double alpha_fast = 0.3;
    const double alpha_slow = 0.05;

    // Current state
    unsigned current_h = solver->heuristic;
    unsigned n_heuristics = solver->mab_heuristics;
    
    // Safety cap for static array usage (typically 2 heuristics: VSIDS and CHB)
    if (n_heuristics > 2) n_heuristics = 2; 

    // Step 1: Calculate Reward R
    // R = (conflicts / max(1, sum_of_LBDs))
    // We approximate sum_of_LBDs using the EMA of fast glue * conflicts
    // as we don't have an explicit sum_of_LBDs field for the interval.
    double conflicts = (double)solver->mab_conflicts;
    double avg_glue = AVERAGE(fast_glue);
    if (avg_glue < 1.0) avg_glue = 1.0; 
    
    double sum_lbd = avg_glue * conflicts;
    if (sum_lbd < 1.0) sum_lbd = 1.0;
    
    double R = conflicts / sum_lbd;

    // Step 2: Update Persistent EMAs for the currently active mode
    if (!initialized) {
        // Initialize EMAs with the first reward to avoid ramp-up bias
        for (unsigned i = 0; i < 2; i++) {
            fast_ema[i] = R;
            slow_ema[i] = R;
        }
        // Initialize glue counter based on current global stats
        // Mode 1 is stable mode, where MAB operates
        last_glue_low_count = solver->statistics.used[1].glue[1] + 
                              solver->statistics.used[1].glue[2];
        initialized = true;
    } else {
        // Update EMAs: New = Alpha * R + (1 - Alpha) * Old
        fast_ema[current_h] = alpha_fast * R + (1.0 - alpha_fast) * fast_ema[current_h];
        slow_ema[current_h] = alpha_slow * R + (1.0 - alpha_slow) * slow_ema[current_h];
    }

    // Step 3: Update Global Stagnation Counter
    // Check for new glue clauses (LBD <= 2) in stable mode (index 1)
    uint64_t current_glue_low = solver->statistics.used[1].glue[1] + 
                                solver->statistics.used[1].glue[2];
    
    // Check if new low glues were learned during this interval
    if (current_glue_low > last_glue_low_count) {
        stagnation_counter = 0; // Reset if progress made
    } else {
        stagnation_counter++;   // Increment if stagnating
    }
    last_glue_low_count = current_glue_low;

    // Step 4: Calculate Priority P for each available mode
    double P[2] = {0};
    for (unsigned i = 0; i < n_heuristics; i++) {
        // P = Slow_EMA + 1.5 * (Fast_EMA - Slow_EMA)
        // This rewards positive momentum
        P[i] = slow_ema[i] + 1.5 * (fast_ema[i] - slow_ema[i]);
    }

    // Step 5: Apply Penalty
    if (stagnation_counter > 6) {
        // If stagnating, penalize the currently active mode
        P[current_h] *= 0.5;
    }

    // Step 6: Softmax Selection
    // Calculate selection probability S_i = exp(P_i) / sum(exp(P_k))
    double exp_P[2];
    double sum_exp = 0.0;
    
    // Find max P for numerical stability in exp calculation
    double max_P = P[0];
    for (unsigned i = 1; i < n_heuristics; i++) {
        if (P[i] > max_P) max_P = P[i];
    }
    
    for (unsigned i = 0; i < n_heuristics; i++) {
        exp_P[i] = exp(P[i] - max_P);
        sum_exp += exp_P[i];
    }

    // Randomly sample the next restart mode based on distribution S
    double rand_val = kissat_pick_double(&solver->random);
    double cumulative = 0.0;
    unsigned next_h = 0;
    
    for (unsigned i = 0; i < n_heuristics; i++) {
        double prob = exp_P[i] / sum_exp;
        cumulative += prob;
        if (rand_val < cumulative) {
            next_h = i;
            break;
        }
    }
    // Fallback for floating point edge cases
    if (cumulative <= rand_val && n_heuristics > 0) {
        next_h = n_heuristics - 1;
    }

    solver->heuristic = next_h;

    // Housekeeping: Reset MAB tracking variables for the next interval
    for (all_variables(idx)) {
        solver->mab_chosen[idx] = 0;
    }
    solver->mab_chosen_tot = 0;
    solver->mab_decisions = 0;
    solver->mab_conflicts = 0;

    // Update selection count for reporting
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
