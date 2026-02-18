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

    // Static state for Asymmetric MACD Momentum Scoring
    // Index 0 = Heuristic 0 (e.g., VSIDS/Stable)
    // Index 1 = Heuristic 1 (e.g., CHB/Focused)
    static double ema_fast[2] = {0.0, 0.0};
    static double ema_slow[2] = {0.0, 0.0};
    static int consecutive_active[2] = {0, 0};
    static uint64_t last_ticks = 0;
    static bool initialized = false;

    // 1. Capture current interval statistics
    unsigned conflicts = solver->mab_conflicts;
    uint64_t current_ticks = solver->ticks;
    unsigned current_mode = solver->heuristic;

    // Safety check for mode bounds (Kissat typically has 2 heuristics)
    if (current_mode > 1) current_mode = 0;

    // 2. Reset MAB tracking variables (Required for Kissat internal correctness)
    for (all_variables(idx)) {
        solver->mab_chosen[idx] = 0;
    }
    solver->mab_chosen_tot = 0;
    solver->mab_decisions = 0;
    solver->mab_conflicts = 0;

    // 3. Handle Initialization
    if (!initialized) {
        last_ticks = current_ticks;
        initialized = true;
        // Cannot calculate yield on first run, keep current heuristic
        solver->mab_select[current_mode]++;
        return;
    }

    // 4. Calculate Yield Y
    // Yield = log(conflicts / time). Using solver->ticks as a robust proxy for time.
    uint64_t delta_ticks = current_ticks - last_ticks;
    last_ticks = current_ticks;

    double yield = 0.0;
    if (delta_ticks > 0) {
        double rate = (double)conflicts / (double)delta_ticks;
        // Clamp rate to avoid log(0) or extreme negative values
        if (rate < 1e-10) rate = 1e-10;
        yield = log(rate);
    } else {
        // Fallback for zero time delta
        yield = -23.0; // approx log(1e-10)
    }

    // 5. Update EMAs (Only for the finished mode)
    const double alpha_fast = 0.20;
    const double alpha_slow = 0.05;

    ema_fast[current_mode] = alpha_fast * yield + (1.0 - alpha_fast) * ema_fast[current_mode];
    ema_slow[current_mode] = alpha_slow * yield + (1.0 - alpha_slow) * ema_slow[current_mode];

    // Update consecutive active counters
    consecutive_active[current_mode]++;
    consecutive_active[1 - current_mode] = 0;

    // 6. Calculate Momentum (M) and Projected Score (P)
    double P[2] = {0.0, 0.0};
    double M[2] = {0.0, 0.0};

    for (int i = 0; i < 2; i++) {
        M[i] = ema_fast[i] - ema_slow[i];
        
        // Asymmetric weighting
        if (M[i] > 0) {
            P[i] = ema_fast[i] + (0.8 * M[i]); // Conservative growth
        } else {
            P[i] = ema_fast[i] + (3.0 * M[i]); // Aggressive penalty
        }
    }

    // 7. Selection Logic
    unsigned next_mode = current_mode;

    // Stop-loss mechanism: force switch if current mode has negative Momentum 
    // and has been active for more than 5 consecutive intervals.
    if (M[current_mode] < 0 && consecutive_active[current_mode] > 5) {
        next_mode = 1 - current_mode;
    } else {
        // Select mode with highest Projected Score
        if (P[0] >= P[1]) {
            next_mode = 0;
        } else {
            next_mode = 1;
        }
    }

    // 8. Apply selection and update stats
    solver->heuristic = next_mode;
    solver->mab_select[next_mode]++;
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
