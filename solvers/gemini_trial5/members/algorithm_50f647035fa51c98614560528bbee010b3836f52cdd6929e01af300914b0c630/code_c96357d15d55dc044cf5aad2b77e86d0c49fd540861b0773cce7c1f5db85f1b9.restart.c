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
    // Static state for Balanced Fast-Adapt MAB
    static double r_prev[2] = {0.0, 0.0};
    static int consecutive_count = 0;

    // Ensure we are working with the expected number of heuristics (typically 2)
    if (solver->mab_heuristics != 2) return;

    unsigned current_mode = solver->heuristic;
    unsigned other_mode = !current_mode; // Assumes 0 and 1 are the only modes

    // Step 1: Compute the raw reward R
    // R = log2(conflicts_generated + 1) / (elapsed_ticks + 1)
    // Using mab_decisions as the measure of elapsed work (ticks)
    double conflicts = (double)solver->mab_conflicts;
    double ticks = solver->mab_decisions;
    
    double R = 0.0;
    if (ticks + 1.0 > 0.0) {
        R = log2(conflicts + 1.0) / (ticks + 1.0);
    }

    // Step 2: Calculate the performance trend T
    // T = R - R_prev
    double T = R - r_prev[current_mode];
    
    // Update R_prev to R for this mode
    r_prev[current_mode] = R;

    // Step 3: Update the Q-value
    // Q_new = 0.75 * Q_old + 0.25 * (R + T)
    // solver->mab_reward is used to store the Q-values
    double Q_old = solver->mab_reward[current_mode];
    double Q_new = 0.75 * Q_old + 0.25 * (R + T);
    
    solver->mab_reward[current_mode] = Q_new;

    // Step 4: Calculate the Selection Score S
    double S[2];
    
    // For the mode just finished: S = Q_new * (0.95 ^ consecutive_count)
    S[current_mode] = Q_new * pow(0.95, (double)consecutive_count);
    
    // For the inactive mode: S = Q_current
    S[other_mode] = solver->mab_reward[other_mode];

    // Step 5: Select the mode with the highest S
    unsigned next_mode = (S[1] > S[0]) ? 1 : 0;

    // Update consecutive_count logic
    if (next_mode != current_mode) {
        consecutive_count = 0;
    } else {
        consecutive_count++;
    }

    // Apply selection
    solver->heuristic = next_mode;
    solver->mab_select[next_mode]++;

    // Reset phase counters for the next run
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
