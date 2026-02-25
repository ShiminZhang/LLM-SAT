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
    // Static state for the MAB-based heuristic selection strategy
    static double fast_ema[2] = {0.0, 0.0};
    static double slow_ema[2] = {0.0, 0.0};
    static double volatility[2] = {0.0, 0.0};
    static uint64_t inactivity_age[2] = {0, 0};
    static uint64_t active_intervals[2] = {0, 0};
    static uint64_t last_ticks = 0;
    static bool initialized = false;

    // Step 1: Calculate Yield Y
    // Yield Y = (conflicts / solver->ticks) / (1.0 + average_LBD)
    // We use the delta of ticks to represent the work done in the current interval.
    uint64_t current_ticks = solver->ticks;
    uint64_t interval_ticks = current_ticks - last_ticks;
    if (interval_ticks == 0) interval_ticks = 1;
    last_ticks = current_ticks;

    double conflicts = (double)solver->mab_conflicts;
    double average_lbd = AVERAGE(slow_glue);
    double Y = (conflicts / (double)interval_ticks) / (1.0 + average_lbd);

    // Identify current mode (active) and other mode (inactive)
    unsigned h = solver->heuristic;
    if (h >= 2) h = 0; // Ensure binary choice safety
    unsigned other = 1 - h;

    // Initialize EMAs on the first run to prevent bias from zero-initialization
    if (!initialized) {
        fast_ema[0] = fast_ema[1] = Y;
        slow_ema[0] = slow_ema[1] = Y;
        initialized = true;
    }

    // Step 2: Update the active mode's metrics
    // Fast EMA (alpha=0.20)
    fast_ema[h] = (0.80 * fast_ema[h]) + (0.20 * Y);
    // Slow EMA (alpha=0.05)
    slow_ema[h] = (0.95 * slow_ema[h]) + (0.05 * Y);
    // Momentum M = Fast_EMA - Slow_EMA
    double M_active = fast_ema[h] - slow_ema[h];
    // Volatility V = EMA(|Y - Fast_EMA|, alpha=0.10)
    double diff = fabs(Y - fast_ema[h]);
    volatility[h] = (0.90 * volatility[h]) + (0.10 * diff);
    // Update active streak
    active_intervals[h]++;

    // Step 3: Apply 'Stale Data' correction to the inactive mode
    // Multiply its EMAs by 0.95 (decay) and increment its 'inactivity_age' counter
    fast_ema[other] *= 0.95;
    slow_ema[other] *= 0.95;
    inactivity_age[other]++;
    // Reset the streak for the mode that was not running
    active_intervals[other] = 0;

    // Step 4: Compute the Projected Score for both modes
    // P = Fast_EMA + (k * M), where k = 1.5 / (1.0 + V)
    uint64_t total_intervals = 0;
    for (unsigned i = 0; i < solver->mab_heuristics; i++) {
        total_intervals += solver->mab_select[i];
    }
    if (total_intervals == 0) total_intervals = 1;

    double P_final[2];
    for (unsigned i = 0; i < 2; i++) {
        double k = 1.5 / (1.0 + volatility[i]);
        double M = fast_ema[i] - slow_ema[i];
        double P = fast_ema[i] + (k * M);
        
        if (i != h) {
            // For the inactive mode, add an exploration bonus
            P_final[i] = P + sqrt((double)inactivity_age[i] / (double)total_intervals);
        } else {
            P_final[i] = P;
        }
    }

    // Step 5: Apply Three-State Gating logic
    unsigned next_h = h;
    if (M_active > (0.05 * slow_ema[h])) {
        // [Improving]: Deterministically stay in the current mode
        next_h = h;
    } else if (M_active < -0.10 && active_intervals[h] > 5) {
        // [Degrading]: Deterministically switch to the other mode (Stop-Loss)
        next_h = other;
    } else {
        // [Stagnant]: Perform Boltzmann Selection between modes using P_final values
        // Temperature T = 0.1 + V
        double T = 0.1 + volatility[h];
        
        // Softmax probabilities for selection
        double max_p = (P_final[0] > P_final[1]) ? P_final[0] : P_final[1];
        double e0 = exp((P_final[0] - max_p) / T);
        double e1 = exp((P_final[1] - max_p) / T);
        double prob0 = e0 / (e0 + e1);
        
        if (kissat_pick_double(&solver->random) < prob0) {
            next_h = 0;
        } else {
            next_h = 1;
        }
    }

    // Reset 'inactivity_age' for the selected mode
    inactivity_age[next_h] = 0;

    // Apply heuristic selection and update Kissat statistics
    solver->heuristic = next_h;
    solver->mab_select[next_h]++;

    // Standard Kissat Housekeeping: Reset MAB interval counters
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
