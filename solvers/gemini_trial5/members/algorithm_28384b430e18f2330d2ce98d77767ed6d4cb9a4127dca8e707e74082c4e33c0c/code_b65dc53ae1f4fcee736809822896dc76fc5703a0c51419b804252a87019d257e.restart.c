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

     // Static variables to track fatigue (consecutive mode selections)
     static unsigned last_phase_mode = 0;
     static int consecutive_modes = 0;
     static bool mab_initialized = false;

     // Initialize tracking on first execution
     if (!mab_initialized) {
         last_phase_mode = solver->heuristic;
         consecutive_modes = 0;
         mab_initialized = true;
     }

     // Step 4 (Part A): Update Fatigue Counter based on the COMPLETED phase
     // This tracks how many times the current mode has been chosen consecutively
     if (solver->heuristic == last_phase_mode) {
         consecutive_modes++;
     } else {
         consecutive_modes = 1;
         last_phase_mode = solver->heuristic;
     }

     // Step 1: Compute Metrics
     // L: Average LBD of the phase (Approximated by Fast Glue EMA)
     double L = AVERAGE(fast_glue);
     // D: Average decision level (Approximated by Level EMA)
     double D = AVERAGE(level);
     // N: Number of variables
     double N = (double)solver->vars;
     
     // d_ratio = D / N (Depth Ratio)
     double d_ratio = (N > 0) ? (D / N) : 0.0;

     // Step 2: Calculate Base Reward R = 1 / L
     // Prevent division by zero; LBD is typically >= 1
     double R = (L > 1e-6) ? (1.0 / L) : 1.0;

     // Step 3: Apply Asymmetric Incentives
     // If the completed phase was STABLE (Arm 0), reward deep search trees.
     // This distinguishes the Stable objective from Focused (which targets low LBD).
     if (solver->heuristic == 0) {
         R = R * (1.0 + (1.5 * d_ratio));
     }

     // Step 4 (Part B): Apply Fatigue Penalty
     // If current mode chosen > 4 times in a row, decay reward to force switching
     if (consecutive_modes > 4) {
         R = R * 0.6;
     }

     // Step 5: Update MAB Statistics
     solver->mab_reward[solver->heuristic] += R;
     solver->mab_select[solver->heuristic]++;

     // Reset phase counters (decisions, conflicts, chosen vars)
     solver->mab_decisions = 0;
     solver->mab_conflicts = 0;
     solver->mab_chosen_tot = 0;
     for (all_variables(idx)) {
         solver->mab_chosen[idx] = 0;
     }

     // Step 6: Deterministic Mean-Based Selection
     // Ensure initial exploration of both arms (Stable=0, Focused=1)
     if (solver->mab_select[0] == 0) {
         solver->heuristic = 0;
         return;
     }
     if (solver->mab_select[1] == 0) {
         solver->heuristic = 1;
         return;
     }

     // Calculate expected mean reward E = Cumulative Reward / Count
     double E0 = solver->mab_reward[0] / solver->mab_select[0];
     double E1 = solver->mab_reward[1] / solver->mab_select[1];

     // Select mode with strictly higher E. Tie defaults to Stable (0).
     if (E1 > E0) {
         solver->heuristic = 1;
     } else {
         solver->heuristic = 0;
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
