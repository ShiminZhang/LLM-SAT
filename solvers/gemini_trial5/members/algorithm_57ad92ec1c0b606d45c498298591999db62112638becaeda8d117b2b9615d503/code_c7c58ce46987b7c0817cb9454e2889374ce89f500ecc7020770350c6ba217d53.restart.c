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
    // Static state variables to persist across function calls
    // We use static because we cannot modify the solver struct
    static double last_reward[2] = {0.0, 0.0};
    static uint64_t last_ticks = 0;
    static unsigned consecutive_count = 0;

    // Step 1: Compute the raw reward R
    // R = (conflicts_generated * 10000.0) / (elapsed_ticks + 1)
    
    uint64_t current_ticks = solver->ticks;
    uint64_t elapsed_ticks = 0;
    
    if (current_ticks >= last_ticks) {
        elapsed_ticks = current_ticks - last_ticks;
    } else {
        elapsed_ticks = current_ticks; // Handle wrap-around or reset
    }
    last_ticks = current_ticks;

    double conflicts = (double)solver->mab_conflicts;
    double R = (conflicts * 10000.0) / (double)(elapsed_ticks + 1);

    // Identify the mode (heuristic) that just finished
    unsigned current_mode = solver->heuristic;
    if (current_mode >= 2) current_mode = 0; // Safety clamp for array access

    // Step 2: Calculate the performance trend T
    // T = R - R_prev
    double R_prev = last_reward[current_mode];
    double T = R - R_prev;
    
    // Update R_prev to R for next time
    last_reward[current_mode] = R;

    // Step 3: Update the Q-value (score) for the current mode
    // Q_new = 0.85 * Q_old + 0.15 * (R + 2.0 * T)
    double Q_old = solver->mab_reward[current_mode];
    double Q_new = 0.85 * Q_old + 0.15 * (R + 2.0 * T);
    solver->mab_reward[current_mode] = Q_new;

    // Step 4: Calculate the Selection Score S for both modes
    unsigned best_mode = current_mode;
    double max_S = -1e100; // Initialize to negative infinity

    unsigned n = solver->mab_heuristics;
    if (n > 2) n = 2; // Ensure we don't exceed static array or solver array bounds

    for (unsigned i = 0; i < n; i++) {
        double S;
        double Q = solver->mab_reward[i];

        if (i == current_mode) {
            // For the mode just finished: S = Q_new * (0.95 ^ consecutive_count)
            S = Q * pow(0.95, (double)consecutive_count);
        } else {
            // For the inactive mode: S = Q_current
            S = Q;
        }

        // Step 5: Select the mode with the highest S
        if (S > max_S) {
            max_S = S;
            best_mode = i;
        }
    }

    // Update consecutive_count
    if (best_mode != current_mode) {
        consecutive_count = 0;
    } else {
        consecutive_count++;
    }

    // Apply selection
    solver->heuristic = best_mode;
    solver->mab_select[best_mode]++;

    // Reset MAB tracking variables for the next phase
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
