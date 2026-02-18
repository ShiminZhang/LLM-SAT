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
    if (solver->mab_heuristics < 2) return;

    // Step 1: Calculate Yield 'Y' (log of conflicts per second)
    // We use decisions as a proxy for time since exact timing can be nondeterministic/expensive.
    double decisions = solver->mab_decisions;
    double conflicts = (double)solver->mab_conflicts;

    // Avoid division by zero and log(0)
    if (decisions < 1.0) decisions = 1.0;
    if (conflicts < 1.0) conflicts = 1.0;

    // Y = log(conflicts / time) -> log(conflicts / decisions)
    // Using natural log as per standard mathematical interpretation of "log"
    double yield = log(conflicts / decisions);

    // Step 2: Maintain EMAs (Static state to persist across calls)
    // Index 0: VSIDS, Index 1: CHB
    static double ema_fast[2] = {0.0, 0.0};
    static double ema_slow[2] = {0.0, 0.0};
    static bool initialized[2] = {false, false};

    unsigned current = solver->heuristic;
    // Safety check for heuristic index
    if (current >= 2) current = 0;

    // Update EMAs only for the mode that just finished
    const double alpha_fast = 0.20;
    const double alpha_slow = 0.05;

    if (!initialized[current]) {
        ema_fast[current] = yield;
        ema_slow[current] = yield;
        initialized[current] = true;
    } else {
        ema_fast[current] = (1.0 - alpha_fast) * ema_fast[current] + alpha_fast * yield;
        ema_slow[current] = (1.0 - alpha_slow) * ema_slow[current] + alpha_slow * yield;
    }

    // Step 3 & 4: Calculate Momentum and Projected Score
    double P[2] = {0.0, 0.0}; // Projected Score
    double M[2] = {0.0, 0.0}; // Momentum

    for (unsigned i = 0; i < 2; i++) {
        if (initialized[i]) {
            M[i] = ema_fast[i] - ema_slow[i];
            P[i] = ema_fast[i] + (1.5 * M[i]);
        } else {
            // Set uninitialized scores very low to ensure they are prioritized 
            // by the initialization logic below
            P[i] = -1e100;
        }
    }

    // Step 5: Switching Logic
    unsigned next = current;
    unsigned other = 1 - current;

    // Force initialization of the inactive mode if it hasn't run yet
    if (!initialized[other]) {
        next = other;
    } else {
        // "If the inactive mode's P is higher than the current mode's P"
        if (P[other] > P[current]) {
            double diff = P[other] - P[current];
            
            // "switch ONLY if the difference ... is greater than 0.1" (Hysteresis)
            bool hysteresis_check = (diff > 0.1);
            
            // "OR if the current mode's Momentum M is strictly negative" (Trend Bailout)
            bool trend_bailout = (M[current] < 0.0);

            if (hysteresis_check || trend_bailout) {
                next = other;
            }
        }
    }

    // Apply the selection
    solver->heuristic = next;
    solver->mab_select[next]++;

    // Update cumulative reward (optional, but maintains solver state consistency)
    solver->mab_reward[current] += yield;

    // Reset MAB counters for the next phase (Standard Kissat reset logic)
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
