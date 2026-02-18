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
    // Static state for Momentum-Driven Stagnation Bandit
    // These persist across calls to track history
    static double fast_ema[2] = {0};
    static double slow_ema[2] = {0};
    static bool initialized[2] = {false};
    static unsigned stagnation = 0;
    static uint64_t last_binary = 0;

    // Safety check: The solver state defines arrays of size 2.
    // We ensure we don't access out of bounds.
    if (solver->mab_heuristics > 2) return;

    // Step 1: Calculate Reward R
    // R = conflicts / max(1, sum_of_LBDs)
    // We estimate sum_of_LBDs using the average fast glue (LBD) and conflict count.
    double avg_lbd = AVERAGE(fast_glue);
    double sum_lbds = (double)solver->mab_conflicts * avg_lbd;
    
    if (sum_lbds < 1.0) sum_lbds = 1.0;
    
    double R = (double)solver->mab_conflicts / sum_lbds;

    // Step 2: Update persistent Exponential Moving Averages (EMAs)
    // Only update for the currently active mode
    unsigned active = solver->heuristic;
    if (active < 2) {
        if (!initialized[active]) {
            fast_ema[active] = R;
            slow_ema[active] = R;
            initialized[active] = true;
        } else {
            // Fast_EMA (alpha=0.3)
            fast_ema[active] += 0.3 * (R - fast_ema[active]);
            // Slow_EMA (alpha=0.05)
            slow_ema[active] += 0.05 * (R - slow_ema[active]);
        }
    }

    // Step 3: Update Stagnation_Counter
    // Check if zero new glue clauses (LBD <= 2) were learned.
    // We use the 'clauses_binary' statistic as a faithful proxy for LBD <= 2 learned clauses.
    uint64_t current_binary = GET(clauses_binary);
    uint64_t delta_binary = 0;

    // Handle potential counter wrap-around or fresh solver instances
    if (current_binary >= last_binary) {
        delta_binary = current_binary - last_binary;
    } else {
        delta_binary = current_binary;
    }
    last_binary = current_binary;

    if (delta_binary == 0) {
        stagnation++;
    } else {
        stagnation = 0;
    }

    // Step 4: Calculate Priority P for each available mode
    // P = Slow_EMA + 1.5 * (Fast_EMA - Slow_EMA)
    double P[2] = {0};
    for (unsigned i = 0; i < solver->mab_heuristics && i < 2; i++) {
        if (initialized[i]) {
            P[i] = slow_ema[i] + 1.5 * (fast_ema[i] - slow_ema[i]);
        }
    }

    // Step 5: Apply Penalty
    // If Stagnation_Counter > 6 and the mode is currently active, apply penalty.
    if (stagnation > 6) {
        if (active < 2) {
            P[active] *= 0.5;
        }
    }

    // Step 6: Select the next restart mode
    // Select mode corresponding to the highest Priority P.
    unsigned best_heuristic = active;
    double max_p = -1e100; // Initialize low to handle potential negative momentum

    for (unsigned i = 0; i < solver->mab_heuristics && i < 2; i++) {
        if (P[i] > max_p) {
            max_p = P[i];
            best_heuristic = i;
        }
    }

    solver->heuristic = best_heuristic;
    if (best_heuristic < 2) {
        solver->mab_select[best_heuristic]++;
    }

    // Standard MAB Housekeeping: Reset counters for the next interval
    solver->mab_decisions = 0;
    solver->mab_conflicts = 0;
    solver->mab_chosen_tot = 0;
    for (all_variables(idx)) {
        solver->mab_chosen[idx] = 0;
    }
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
