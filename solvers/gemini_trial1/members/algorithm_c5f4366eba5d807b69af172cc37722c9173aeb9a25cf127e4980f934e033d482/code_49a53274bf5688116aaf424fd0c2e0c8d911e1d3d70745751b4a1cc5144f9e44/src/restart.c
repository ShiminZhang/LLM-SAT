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

  // Standard Kissat checks for restart enablement and decision level
  if (!GET_OPTION (restart))
    return false;
  if (!solver->level)
    return false;
  if (CONFLICTS < solver->limits.restart.conflicts)
    return false;

  // Preserve standard behavior for stable mode (Reluctant Doubling)
  // The LBD-based algorithm is intended for the focused mode.
  if (solver->stable)
    return kissat_reluctant_triggered (&solver->reluctant);

  // --- Reactive Gradient LBD Restarting Algorithm ---

  // Step 1: Retrieve metrics
  // avg_fast and avg_slow are exponential moving averages of LBD
  const double avg_fast = AVERAGE (fast_glue);
  const double avg_slow = AVERAGE (slow_glue);

  // Reconstruct 'raw_lbd' from the EMA state.
  // Since we cannot modify the solver struct to store the raw LBD of the last conflict,
  // we reverse-engineer it from the change in the EMA.
  // EMA update formula: new = (1 - alpha) * old + alpha * raw
  // Therefore: raw = (new - (1 - alpha) * old) / alpha
  
  static double last_avg_fast = 0;
  static uint64_t last_conflicts_count = 0;
  
  // Detect reset or new solve
  if (CONFLICTS < last_conflicts_count) {
    last_avg_fast = avg_fast;
  }

  double raw_lbd = avg_fast; // Fallback if no recent change
  double alpha = 1.0 / (double) GET_OPTION (emafast);
  
  // Only calculate raw if we have history and a conflict occurred since last check
  if (CONFLICTS > last_conflicts_count && last_avg_fast > 0) {
      // Prevent division by zero or extreme values if alpha is weird
      if (alpha > 0.0) {
          raw_lbd = (avg_fast - (1.0 - alpha) * last_avg_fast) / alpha;
          // Clamp to reasonable range to avoid numerical noise spikes
          if (raw_lbd < 1.0) raw_lbd = 1.0;
      }
  }

  // Update static history for next call
  last_avg_fast = avg_fast;
  last_conflicts_count = CONFLICTS;

  // Calculate effective fast metric
  // fast_metric = (0.75 * avg_fast) + (0.25 * raw_lbd)
  double fast_metric = (0.75 * avg_fast) + (0.25 * raw_lbd);

  // Step 2: Persistent Circular Buffer
  // "On each function call, append the current fast_metric"
  static double buffer[10] = {0};
  static int head = 0;
  static bool filled = false;
  static uint64_t buffer_reset_check = 0;

  if (CONFLICTS < buffer_reset_check) {
      head = 0;
      filled = false;
  }
  buffer_reset_check = CONFLICTS;

  // Calculate gradient
  // If buffer not filled, assume gradient is 0 (oldest = current)
  double oldest = filled ? buffer[head] : fast_metric;
  
  buffer[head] = fast_metric;
  head = (head + 1) % 10;
  if (head == 0) filled = true;

  double gradient = fast_metric - oldest;

  // Step 3: Dynamic Margin
  // Base margin M (typically 1.05, derived from options)
  double M = (100.0 + (double) GET_OPTION (restartmargin)) / 100.0;
  
  // M' = M - (gradient * 2.0)
  double M_prime = M - (gradient * 2.0);

  // Step 4: Clamp M' to [0.8, 1.5]
  if (M_prime < 0.8) M_prime = 0.8;
  if (M_prime > 1.5) M_prime = 1.5;

  // Step 5: Trigger Restart
  // Trigger if fast_metric > avg_slow * M'
  double limit = avg_slow * M_prime;

  // Logging
  kissat_extremely_verbose (solver,
                            "restart gradient: fast=%g raw=%g metric=%g grad=%g "
                            "M=%g M'=%g limit=%g slow=%g",
                            avg_fast, raw_lbd, fast_metric, gradient, 
                            M, M_prime, limit, avg_slow);

  return (fast_metric > limit);
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
    // Reset MAB tracking variables
    unsigned stable_restarts = 0;
    solver->mab_reward[solver->heuristic] += log2(solver->mab_decisions) / log2(solver->mab_conflicts);
    
    // Clear per-variable MAB data
    for (all_variables(idx)) {
        solver->mab_chosen[idx] = 0;
    }
    solver->mab_chosen_tot = 0;
    solver->mab_decisions = 0;
    solver->mab_conflicts = 0;
    
    // Count stable restarts across all heuristics
    for (unsigned i = 0; i < solver->mab_heuristics; i++) {
        stable_restarts += solver->mab_select[i];
    }

    // Track recent gains with momentum
    static double recent_gains[10] = {0};
    static int gain_index = 0;
    static double momentum = 1.0;

    double current_gain = solver->mab_reward[solver->heuristic] / solver->mab_select[solver->heuristic];
    recent_gains[gain_index] = current_gain;
    gain_index = (gain_index + 1) % 10;

    // Compute average gain over recent window
    double avg_gain = 0;
    for (int i = 0; i < 10; i++) {
        avg_gain += recent_gains[i];
    }
    avg_gain /= 10;

    // Update momentum based on performance
    if (current_gain > avg_gain) {
        momentum *= 1.1;
    } else {
        momentum *= 0.9;
    }

    // Compute adaptive exploration parameter
    double adaptive_c = solver->mabc / (momentum * (stable_restarts + 1));

    // Select next heuristic
    if (stable_restarts < solver->mab_heuristics) {
        // Exploration phase: alternate between first two heuristics
        solver->heuristic = solver->heuristic == 0 ? 1 : 0;
    } else {
        // UCB-based selection
        double ucb[2];
        solver->heuristic = 0;
        for (unsigned i = 0; i < solver->mab_heuristics; i++) {
            ucb[i] = solver->mab_reward[i] / solver->mab_select[i] 
                   + sqrt(adaptive_c * log(stable_restarts + 1) / solver->mab_select[i]);
            if (i != 0 && ucb[i] > ucb[solver->heuristic]) {
                solver->heuristic = i;
            }
        }
    }
    
    // Update selection count for chosen heuristic
    solver->mab_select[solver->heuristic]++;
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
