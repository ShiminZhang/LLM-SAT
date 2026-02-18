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
    // Static state variables to persist across restart calls
    // Note: These assume single solver instance or sequential usage. 
    // Reset logic is included for safety.
    static double prev_raw_reward[2] = {0.0, 0.0};
    static uint64_t last_mab_ticks = 0;
    static unsigned consecutive_count = 0;

    // Safety check: Detect if solver instance changed or reset (time regression)
    if (solver->ticks < last_mab_ticks) {
        last_mab_ticks = 0;
        prev_raw_reward[0] = 0.0;
        prev_raw_reward[1] = 0.0;
        consecutive_count = 0;
    }

    // Step 1: Compute the raw reward R
    // R = log2(conflicts_generated + 1) / (elapsed_ticks + 1)
    uint64_t current_ticks = solver->ticks;
    uint64_t delta_ticks = current_ticks - last_mab_ticks;
    last_mab_ticks = current_ticks;

    double conflicts = (double)solver->mab_conflicts;
    double ticks_duration = (double)delta_ticks;
    
    // Use log2 from math.h, add 1.0 to avoid domain errors
    double R = log2(conflicts + 1.0) / (ticks_duration + 1.0);

    // Identify current and inactive modes
    unsigned current_mode = solver->heuristic;
    unsigned inactive_mode = 1 - current_mode; // Assumes 2 heuristics (0 and 1)

    // Step 2: Calculate the performance trend T
    // T = R - R_prev
    double R_prev = prev_raw_reward[current_mode];
    double T = R - R_prev;
    prev_raw_reward[current_mode] = R; // Update stored reward

    // Step 3: Update the Q-value (score)
    // Q_new = 0.85 * Q_old + 0.15 * (R + 2.0 * T)
    // We use solver->mab_reward to store the Q-values
    double Q_old = solver->mab_reward[current_mode];
    double Q_new = 0.85 * Q_old + 0.15 * (R + 2.0 * T);
    solver->mab_reward[current_mode] = Q_new;

    // Step 4: Calculate the Selection Score S
    // For current mode: S = Q_new * (0.95 ^ consecutive_count)
    double S_current = Q_new * pow(0.95, (double)consecutive_count);
    
    // For inactive mode: S = Q_current (stored)
    double S_inactive = solver->mab_reward[inactive_mode];

    // Initialization safeguard: If inactive mode has never been selected, 
    // force its selection to gather initial data.
    if (solver->mab_select[inactive_mode] == 0) {
        S_inactive = 1.0e100; // Arbitrary large value to ensure selection
    }

    // Step 5: Compare and Select
    unsigned next_mode = current_mode;

    if (S_inactive > S_current) {
        // Check stability period (minimum 3 consecutive runs)
        if (consecutive_count < 3) {
            next_mode = current_mode; // Force stay
        } else {
            next_mode = inactive_mode; // Switch
        }
    } else {
        // Current mode score is higher or equal
        next_mode = current_mode;
    }

    // Update consecutive_count logic
    if (next_mode != current_mode) {
        consecutive_count = 0;
    } else {
        consecutive_count++;
    }

    // Apply selection
    solver->heuristic = next_mode;
    solver->mab_select[next_mode]++;

    // Reset MAB tracking variables for the next phase
    solver->mab_conflicts = 0;
    solver->mab_decisions = 0;

    // Clear per-variable chosen counts (standard cleanup)
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
