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
    // Static state to maintain the bandit's memory across solver restarts
    static double r_prev[2] = {0.0, 0.0};
    static int consecutive_count = 0;
    static uint64_t last_ticks = 0;
    static double global_best_lbd = 1e10;
    static unsigned global_min_unassigned = 0;
    static bool initialized = false;

    // Initialize global tracking on the very first execution
    if (!initialized) {
        global_min_unassigned = solver->vars;
        initialized = true;
    }

    // --- Step 1: Compute the raw efficiency metric E ---
    // E = log2(conflicts_generated + 1) / (elapsed_ticks + 1) * (1.0 / max(1, avg_lbd))
    uint64_t current_ticks = solver->ticks;
    uint64_t elapsed_ticks = (current_ticks > last_ticks) ? (current_ticks - last_ticks) : 0;
    last_ticks = current_ticks;

    double avg_lbd = AVERAGE(fast_glue);
    double denom_lbd = (avg_lbd < 1.0) ? 1.0 : avg_lbd;
    
    double E = (log2((double)solver->mab_conflicts + 1.0) / ((double)elapsed_ticks + 1.0)) * (1.0 / denom_lbd);

    // --- Step 2: Calculate a Progress Multiplier M ---
    // If solver improved global 'best_lbd' or 'best_assigned_trail', M = 1.3; else M = 0.7
    bool improved = false;
    if (avg_lbd < global_best_lbd) {
        global_best_lbd = avg_lbd;
        improved = true;
    }
    // Improvement in assigned trail means fewer unassigned variables than the previous minimum
    if (solver->unassigned < global_min_unassigned) {
        global_min_unassigned = solver->unassigned;
        improved = true;
    }
    double M = improved ? 1.3 : 0.7;

    // --- Step 3: Calculate Reward R and Trend T ---
    // R = E * M. T = R - R_prev. If R_prev is 0, T = 0. Update R_prev = R.
    unsigned active = solver->heuristic;
    if (active > 1) active = 0; // Safety clamp for mode index
    
    double R = E * M;
    double T = 0.0;
    if (r_prev[active] != 0.0) {
        T = R - r_prev[active];
    }
    r_prev[active] = R;

    // --- Step 4: Update Q-value with dampened PD-update rule ---
    // Dead-zoning: If |T| < 0.05 * R, set T = 0
    if (fabs(T) < (0.05 * R)) {
        T = 0.0;
    }
    // Q_new = 0.80 * Q_old + 0.20 * (R + 1.5 * T)
    double Q_old = solver->mab_reward[active];
    double Q_new = 0.80 * Q_old + 0.20 * (R + 1.5 * T);
    solver->mab_reward[active] = Q_new;

    // --- Step 5: Calculate Selection Score S for both modes ---
    // Active mode: S = Q_new * (0.98 ^ consecutive_count)
    // Inactive mode: S = Q_stored
    unsigned inactive = 1 - active;
    double S_active = Q_new * pow(0.98, (double)consecutive_count);
    double S_inactive = solver->mab_reward[inactive];

    // --- Step 6: Select mode with highest S ---
    unsigned next_mode;
    if (S_inactive > S_active) {
        next_mode = inactive;
        consecutive_count = 0; // Reset on mode change
    } else {
        next_mode = active;
        consecutive_count++;   // Increment on mode persistence
    }

    // Apply the selection to the solver
    solver->heuristic = next_mode;
    solver->mab_select[next_mode]++;

    // --- Housekeeping: Reset MAB tracking for the next phase ---
    solver->mab_conflicts = 0;
    solver->mab_decisions = 0;
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
