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
    // Static state for Acceleration-Sensitive MACD Strategy
    // We maintain state for 2 heuristics (0=VSIDS, 1=CHB)
    static double fast_ema[2];
    static double slow_ema[2];
    static double signal_ema[2];
    static bool initialized[2];
    static uint64_t last_ticks;
    static int consecutive_active;
    static unsigned last_heuristic;
    static bool global_init = false;

    // Initialize statics on very first load or if solver appears reset
    // We use mab_select counts to detect a fresh solver instance
    if (!global_init || (solver->mab_select[0] == 0 && solver->mab_select[1] == 0)) {
        fast_ema[0] = fast_ema[1] = 0.0;
        slow_ema[0] = slow_ema[1] = 0.0;
        signal_ema[0] = signal_ema[1] = 0.0;
        initialized[0] = initialized[1] = false;
        last_ticks = solver->ticks;
        consecutive_active = 0;
        last_heuristic = solver->heuristic;
        global_init = true;
    }

    // Step 2: Calculate Yield 'Y'
    // Yield = log(conflicts per second). 
    // We use solver->ticks as a proxy for time.
    uint64_t current_ticks = solver->ticks;
    uint64_t delta_ticks = current_ticks - last_ticks;
    if (delta_ticks < 1) delta_ticks = 1; // Prevent division by zero
    last_ticks = current_ticks;

    unsigned conflicts = solver->mab_conflicts;
    
    // Calculate conflicts per tick. Add epsilon to avoid log(0).
    double cps = (double)conflicts / (double)delta_ticks;
    if (cps < 1e-10) cps = 1e-10;
    double yield = log(cps);

    unsigned h = solver->heuristic;

    // Update EMAs for the mode that just finished (h)
    // Step 1: Maintain EMAs (Fast=0.20, Slow=0.05, Signal=0.10)
    if (!initialized[h]) {
        // Initialize EMAs with the first yield value to avoid convergence lag
        fast_ema[h] = yield;
        slow_ema[h] = yield;
        signal_ema[h] = 0.0;
        initialized[h] = true;
    } else {
        // Fast EMA (alpha=0.20)
        fast_ema[h] = 0.80 * fast_ema[h] + 0.20 * yield;
        // Slow EMA (alpha=0.05)
        slow_ema[h] = 0.95 * slow_ema[h] + 0.05 * yield;
        
        // Step 3: Divergence and Signal Update
        double divergence = fast_ema[h] - slow_ema[h];
        // Signal EMA (alpha=0.10)
        signal_ema[h] = 0.90 * signal_ema[h] + 0.10 * divergence;
    }

    // Calculate Momentum for current mode for Stop-Loss check
    // M = D + (D - Signal)
    double current_divergence = fast_ema[h] - slow_ema[h];
    double current_momentum = current_divergence + (current_divergence - signal_ema[h]);

    // Update consecutive intervals counter
    if (h == last_heuristic) {
        consecutive_active++;
    } else {
        consecutive_active = 1;
        last_heuristic = h;
    }

    // Step 4: Compute Projected Score P for both modes
    // P = Fast_EMA + (1.5 * M)
    double p_score[2];
    for (unsigned i = 0; i < 2; i++) {
        double d = fast_ema[i] - slow_ema[i];
        double m = d + (d - signal_ema[i]);
        p_score[i] = fast_ema[i] + (1.5 * m);
    }

    // Step 5: Select mode
    // Default: Select mode with highest Projected Score
    unsigned next_h;
    if (p_score[0] >= p_score[1]) {
        next_h = 0;
    } else {
        next_h = 1;
    }

    // Stop-Loss Override
    // "Force a switch if current mode has negative Momentum (M < 0) and has been active for more than 5 consecutive intervals"
    if (current_momentum < 0 && consecutive_active > 5) {
        next_h = !h; // Switch heuristic (0->1 or 1->0)
    }

    // Apply selection
    solver->heuristic = next_h;
    solver->mab_select[next_h]++;

    // Reset MAB tracking variables for next phase
    solver->mab_decisions = 0;
    solver->mab_conflicts = 0;
    solver->mab_chosen_tot = 0;
    
    // Clear per-variable chosen counts
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
