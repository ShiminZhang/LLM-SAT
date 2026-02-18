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
    // Static state for MACD-Driven Momentum Switching
    // Index 0: Heuristic 0 (e.g., VSIDS)
    // Index 1: Heuristic 1 (e.g., CHB)
    static double fast_ema[2] = {0.0, 0.0};
    static double slow_ema[2] = {0.0, 0.0};
    static uint64_t last_ticks = 0;
    static int consecutive_active = 0;
    static unsigned last_heuristic = 0;
    static bool initialized = false;

    // 1. Calculate Yield 'Y'
    // Algorithm: log(conflicts) / seconds
    // We use solver->ticks as a deterministic proxy for time/work.
    uint64_t current_ticks = solver->ticks;
    uint64_t delta_ticks = current_ticks - last_ticks;
    last_ticks = current_ticks;

    // Avoid division by zero
    if (delta_ticks == 0) delta_ticks = 1;

    double conflicts = (double)solver->mab_conflicts;
    double yield = 0.0;

    // Calculate Yield: log(conflicts) / ticks
    // We scale by 1,000,000 to keep the small floating point values readable.
    // This does not affect relative comparisons.
    if (conflicts > 0) {
        yield = (kissat_logn(solver->mab_conflicts) * 1000000.0) / (double)delta_ticks;
    }

    // Identify current mode
    unsigned h = solver->heuristic;
    if (h >= 2) h = 0; // Safety fallback if more than 2 heuristics defined

    // Initialize EMAs on first run
    if (!initialized) {
        fast_ema[0] = fast_ema[1] = yield;
        slow_ema[0] = slow_ema[1] = yield;
        last_heuristic = h;
        consecutive_active = 0;
        initialized = true;
    }

    // 2. Update EMAs for the mode that just finished
    // Fast EMA (alpha=0.20)
    fast_ema[h] = 0.80 * fast_ema[h] + 0.20 * yield;
    // Slow EMA (alpha=0.05)
    slow_ema[h] = 0.95 * slow_ema[h] + 0.05 * yield;

    // Update consecutive active counter
    if (h == last_heuristic) {
        consecutive_active++;
    } else {
        consecutive_active = 1;
        last_heuristic = h;
    }

    // 3. Calculate Momentum (M) and Projected Score (P)
    double momentum[2];
    double projected[2];

    for (unsigned i = 0; i < 2; i++) {
        momentum[i] = fast_ema[i] - slow_ema[i];
        // P = Fast_EMA + (1.5 * M)
        projected[i] = fast_ema[i] + (1.5 * momentum[i]);
    }

    // 4. Select Mode
    unsigned next_h = h;

    // Default: Select mode with highest Projected Score
    if (projected[0] >= projected[1]) {
        next_h = 0;
    } else {
        next_h = 1;
    }

    // 5. Stop-loss Mechanism
    // Override if current mode has negative Momentum and active > 5 intervals
    if (momentum[h] < 0 && consecutive_active > 5) {
        next_h = 1 - h;
    }

    // Apply selection
    solver->heuristic = next_h;

    // Update selection statistics
    if (next_h < solver->mab_heuristics) {
        solver->mab_select[next_h]++;
    }

    // 6. Reset MAB counters (Standard Kissat housekeeping)
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
