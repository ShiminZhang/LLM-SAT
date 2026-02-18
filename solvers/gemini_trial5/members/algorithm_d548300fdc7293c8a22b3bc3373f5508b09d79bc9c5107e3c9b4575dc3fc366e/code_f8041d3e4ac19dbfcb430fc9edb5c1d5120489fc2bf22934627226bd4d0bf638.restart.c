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
    // Algorithm: High-Frequency Reactive MACD
    
    // 1. Static State Maintenance
    static double fast_ema[2] = {0.0, 0.0};
    static double slow_ema[2] = {0.0, 0.0};
    static uint64_t last_ticks = 0;
    
    // Tracking for "active for more than 5 consecutive intervals"
    static unsigned last_heuristic = 0;
    static int consecutive_active = 0;
    
    static bool initialized = false;

    // Constants
    const double alpha_fast = 0.40;
    const double alpha_slow = 0.15;

    // Initialize state on first run
    if (!initialized) {
        last_ticks = solver->ticks;
        last_heuristic = solver->heuristic;
        consecutive_active = 0;
        initialized = true;
    }

    // Update active counter
    unsigned current_h = solver->heuristic;
    if (current_h == last_heuristic) {
        consecutive_active++;
    } else {
        consecutive_active = 1;
        last_heuristic = current_h;
    }

    // 2. Calculate Yield 'Y' (log of conflicts per second)
    // We use solver->ticks as a proxy for time (seconds)
    uint64_t current_ticks = solver->ticks;
    uint64_t delta_ticks = current_ticks - last_ticks;
    last_ticks = current_ticks;

    // Prevent division by zero and handle small deltas
    double duration = (delta_ticks > 0) ? (double)delta_ticks : 1.0;
    double conflicts = (double)solver->mab_conflicts;

    // Y = log(conflicts / time). Using +1.0 for safety with log.
    double yield = log((conflicts + 1.0) / duration);

    // Update EMAs only for the mode that just finished (current_h)
    // Ensure heuristic index is within bounds (assuming 2 heuristics: 0 and 1)
    if (current_h > 1) current_h = 0; 

    fast_ema[current_h] = alpha_fast * yield + (1.0 - alpha_fast) * fast_ema[current_h];
    slow_ema[current_h] = alpha_slow * yield + (1.0 - alpha_slow) * slow_ema[current_h];

    // 3. Calculate Momentum M = Fast - Slow
    double momentum[2];
    for (unsigned i = 0; i < 2; i++) {
        momentum[i] = fast_ema[i] - slow_ema[i];
    }

    // 4. Compute Projected Score P = Fast + (1.5 * M)
    double projected_score[2];
    for (unsigned i = 0; i < 2; i++) {
        projected_score[i] = fast_ema[i] + (1.5 * momentum[i]);
    }

    // 5. Select Mode with highest P
    unsigned next_h = (projected_score[0] >= projected_score[1]) ? 0 : 1;

    // Override: Force switch if current mode has negative Momentum (M < 0) 
    // and has been active for more than 5 consecutive intervals.
    // This acts as a 'stop-loss' mechanism.
    if (momentum[current_h] < 0 && consecutive_active > 5) {
        next_h = 1 - current_h; // Switch to the other heuristic
    }

    // Apply selection
    solver->heuristic = next_h;
    solver->mab_select[next_h]++;
    
    // Update tracking for next iteration
    // If we switched, the next call will see a change in heuristic relative to 'last_heuristic'
    // effectively resetting the counter in the next call. 
    // However, we update last_heuristic here to track what we *chose*.
    // Actually, consecutive_active tracks how long the *incoming* mode was active.
    // The counter update logic at the top handles the continuity.
    // We just need to ensure last_heuristic matches what we set now for the check in the next call.
    last_heuristic = next_h; 

    // Reset MAB tracking variables (Standard Kissat MAB maintenance)
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
