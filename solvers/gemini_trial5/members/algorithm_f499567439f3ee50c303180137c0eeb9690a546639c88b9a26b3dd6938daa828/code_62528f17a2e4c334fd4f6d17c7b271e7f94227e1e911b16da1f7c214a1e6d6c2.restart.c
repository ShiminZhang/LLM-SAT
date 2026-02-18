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
  // Static state variables to track history across phases
  // Note: These persist across solver instances due to API constraints preventing struct modification
  static double previous_rewards[2] = {0.0, 0.0};
  static int consecutive_count = 0;

  // Step 1: Compute the raw reward R for the just-completed phase
  // R = log2(conflicts_generated + 1) / (elapsed_ticks + 1)
  // We use mab_decisions as the proxy for elapsed_ticks in the current phase
  double conflicts = (double) solver->mab_conflicts;
  double ticks = solver->mab_decisions;
  double R = log2 (conflicts + 1.0) / (ticks + 1.0);

  unsigned current_mode = solver->heuristic;

  // Step 2: Calculate the performance trend T
  // T = R - R_prev
  double R_prev = previous_rewards[current_mode];
  double T = R - R_prev;

  // Update R_prev to R
  previous_rewards[current_mode] = R;

  // Step 3: Update the Q-value (score) for the current mode
  // Q_new = 0.85 * Q_old + 0.15 * (R + 2.0 * T)
  double Q_old = solver->mab_reward[current_mode];
  double Q_new = 0.85 * Q_old + 0.15 * (R + 2.0 * T);
  solver->mab_reward[current_mode] = Q_new;

  // Step 4: Determine a dynamic decay factor lambda based on the Trend T
  // If T > 0 (accelerating), lambda = 0.98; if T <= 0 (decelerating), lambda = 0.92
  double lambda = (T > 0.0) ? 0.98 : 0.92;

  // Calculate the Selection Score S
  double scores[2];
  unsigned num_heuristics = solver->mab_heuristics;
  if (num_heuristics > 2)
    num_heuristics = 2; // Safety bound matching static arrays

  for (unsigned i = 0; i < num_heuristics; i++) {
    if (i == current_mode) {
      // For the mode just finished: S = Q_new * (lambda ^ consecutive_count)
      scores[i] = Q_new * pow (lambda, consecutive_count);
    } else {
      // For the inactive mode: S = Q_current
      scores[i] = solver->mab_reward[i];
    }
  }

  // Step 5: Select the mode with the highest S
  unsigned best_mode = current_mode;
  double best_score = -1.0; 

  if (num_heuristics > 0) {
    best_mode = 0;
    best_score = scores[0];
  }

  for (unsigned i = 1; i < num_heuristics; i++) {
    if (scores[i] > best_score) {
      best_score = scores[i];
      best_mode = i;
    }
  }

  // If the mode changes, reset consecutive_count to 0; otherwise, increment it
  if (best_mode != current_mode) {
    consecutive_count = 0;
  } else {
    consecutive_count++;
  }

  // Apply selection
  solver->heuristic = best_mode;
  solver->mab_select[best_mode]++;

  // Reset MAB tracking variables for the next phase
  solver->mab_decisions = 0;
  solver->mab_conflicts = 0;
  solver->mab_chosen_tot = 0;

  for (all_variables (idx)) {
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
