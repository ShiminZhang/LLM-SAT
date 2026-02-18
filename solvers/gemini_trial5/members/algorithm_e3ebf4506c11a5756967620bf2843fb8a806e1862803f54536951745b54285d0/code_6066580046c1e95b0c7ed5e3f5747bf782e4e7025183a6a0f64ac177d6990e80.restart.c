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
    // Static state variables for Asymmetric Momentum Stagnation Bandit
    static double fast_ema[2] = {0};
    static double slow_ema[2] = {0};
    static int stagnation_counter = 0;
    static uint64_t last_sum_lbds = 0;
    static uint64_t last_small_glue = 0;

    // 1. Calculate statistics for the just-completed interval
    // We access the stable mode statistics (index 1) to get LBD sums
    uint64_t current_sum_lbds = 0;
    uint64_t current_small_glue = 0;

    // Iterate through glue histogram (0 to 127 as per API reference MAX_GLUE_USED)
    for (int i = 0; i <= 127; i++) {
        uint64_t count = solver->statistics.used[1].glue[i];
        current_sum_lbds += i * count;
        if (i <= 2) {
            current_small_glue += count;
        }
    }

    // Calculate deltas (work done during this interval)
    uint64_t delta_sum_lbds = current_sum_lbds - last_sum_lbds;
    uint64_t delta_small_glue = current_small_glue - last_small_glue;

    // Update history for next call
    last_sum_lbds = current_sum_lbds;
    last_small_glue = current_small_glue;

    // 2. Calculate Reward R
    // R = conflicts / max(1, sum_of_LBDs)
    // solver->mab_conflicts tracks conflicts since last MAB update
    double sum_lbd_safe = (delta_sum_lbds > 0) ? (double)delta_sum_lbds : 1.0;
    double R = (double)solver->mab_conflicts / sum_lbd_safe;

    // 3. Update EMAs for the current heuristic
    unsigned h = solver->heuristic;

    // Check initialization (if this heuristic ran for the first time)
    if (solver->mab_select[h] == 0) {
        fast_ema[h] = R;
        slow_ema[h] = R;
    } else {
        // Compare R to current Fast_EMA to determine alpha
        double alpha_fast = (R > fast_ema[h]) ? 0.4 : 0.15;
        
        // Update Fast_EMA
        fast_ema[h] += alpha_fast * (R - fast_ema[h]);
        
        // Update Slow_EMA (fixed alpha = 0.05)
        slow_ema[h] += 0.05 * (R - slow_ema[h]);
    }

    // 4. Update Global Stagnation Counter
    // Increment if zero new glue clauses (LBD <= 2) were learned
    if (delta_small_glue == 0) {
        stagnation_counter++;
    } else {
        stagnation_counter = 0;
    }

    // Increment selection count for the heuristic that just ran
    solver->mab_select[h]++;

    // 5. Calculate Priorities and Select Next Mode
    double best_p = -1.0;
    unsigned best_h = 0;
    bool found_unexplored = false;

    for (unsigned i = 0; i < solver->mab_heuristics; i++) {
        // Exploration: Ensure each heuristic is tried at least once
        if (solver->mab_select[i] == 0) {
            best_h = i;
            found_unexplored = true;
            break;
        }

        // Calculate Priority P
        // P = Slow_EMA + 1.5 * (Fast_EMA - Slow_EMA)
        double p = slow_ema[i] + 1.5 * (fast_ema[i] - slow_ema[i]);

        // Apply penalty if Stagnation_Counter > 6 and mode is currently active
        // "currently active" refers to the mode solver->heuristic that just ran
        if (stagnation_counter > 6 && i == solver->heuristic) {
            p *= 0.5;
        }

        if (p > best_p) {
            best_p = p;
            best_h = i;
        }
    }

    // 6. Set the next restart mode
    if (found_unexplored) {
        solver->heuristic = best_h;
    } else {
        solver->heuristic = best_h;
    }

    // 7. Reset MAB tracking variables for the next interval
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
