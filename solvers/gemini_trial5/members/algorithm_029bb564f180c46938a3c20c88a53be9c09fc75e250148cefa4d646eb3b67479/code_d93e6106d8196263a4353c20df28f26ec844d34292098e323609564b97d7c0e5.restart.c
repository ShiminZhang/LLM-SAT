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
    // Static state variables for the Momentum Reset Stagnation Bandit algorithm
    // These persist across function calls to maintain history (EMAs, stagnation, previous stats)
    static double fast_ema[2] = {0};
    static double slow_ema[2] = {0};
    static int stagnation_counter = 0;
    static uint64_t prev_lbd_sum = 0;
    static uint64_t prev_small_glue_count = 0;
    static bool mab_initialized = false;

    // 1. Calculate Reward R
    // We need the sum of LBDs and count of small glue clauses for the just-completed interval.
    // Calculate current global sums from statistics.
    
    uint64_t current_lbd_sum = 0;
    uint64_t current_small_glue_count = 0;
    
    // Iterate through glue values (0-127) for stable mode (index 1).
    // solver->statistics.used[1].glue is the array of glue counts.
    for (int i = 0; i <= 127; i++) {
        uint64_t count = solver->statistics.used[1].glue[i];
        current_lbd_sum += count * i;
        if (i <= 2) {
            current_small_glue_count += count;
        }
    }

    // Handle initialization phase (first run)
    if (!mab_initialized) {
        mab_initialized = true;
        prev_lbd_sum = current_lbd_sum;
        prev_small_glue_count = current_small_glue_count;
        
        // Reset counters and return, keeping current heuristic
        for (all_variables(idx)) {
            solver->mab_chosen[idx] = 0;
        }
        solver->mab_chosen_tot = 0;
        solver->mab_decisions = 0;
        solver->mab_conflicts = 0;
        return;
    }

    // Calculate deltas for the interval
    uint64_t delta_lbd_sum = 0;
    if (current_lbd_sum > prev_lbd_sum) {
        delta_lbd_sum = current_lbd_sum - prev_lbd_sum;
    }
    
    uint64_t delta_small_glue = 0;
    if (current_small_glue_count > prev_small_glue_count) {
        delta_small_glue = current_small_glue_count - prev_small_glue_count;
    }

    // Update previous values for next call
    prev_lbd_sum = current_lbd_sum;
    prev_small_glue_count = current_small_glue_count;

    // Calculate Reward R = conflicts / max(1, sum_of_LBDs)
    double conflicts = (double)solver->mab_conflicts;
    double lbd_sum = (double)delta_lbd_sum;
    if (lbd_sum < 1.0) lbd_sum = 1.0;
    double R = conflicts / lbd_sum;

    // 2. Update EMAs for the currently active mode
    unsigned h = solver->heuristic;
    // Ensure heuristic index is within bounds (typically 0 or 1)
    if (h > 1) h = 0; 

    // Fast_EMA (alpha=0.3) and Slow_EMA (alpha=0.05)
    fast_ema[h] += 0.3 * (R - fast_ema[h]);
    slow_ema[h] += 0.05 * (R - slow_ema[h]);

    // 3. Update Global Stagnation Counter
    // Increment if zero new glue clauses (LBD <= 2) were learned
    if (delta_small_glue == 0) {
        stagnation_counter++;
    } else {
        stagnation_counter = 0;
    }

    // 4. Calculate Priority P for each available mode
    double P[2];
    for (unsigned i = 0; i < 2; i++) {
        P[i] = slow_ema[i] + 1.5 * (fast_ema[i] - slow_ema[i]);
    }

    // 5. Apply Penalty and Momentum Reset
    // If Stagnation_Counter > 5 and the mode is currently active
    if (stagnation_counter > 5) {
        // Apply harsh penalty
        P[h] *= 0.3;
        // Force momentum reset
        fast_ema[h] = slow_ema[h];
    }

    // 6. Select Next Mode
    // Select mode corresponding to the highest Priority P
    if (P[1] > P[0]) {
        solver->heuristic = 1;
    } else {
        solver->heuristic = 0;
    }

    // Standard MAB bookkeeping
    solver->mab_select[solver->heuristic]++;
    
    // Reset interval counters
    for (all_variables(idx)) {
        solver->mab_chosen[idx] = 0;
    }
    solver->mab_chosen_tot = 0;
    solver->mab_decisions = 0;
    solver->mab_conflicts = 0;
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
