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

  // Static state for LBD variance tracking and restart timing
  // We use static variables to persist state without modifying struct kissat
  static double lbd_variance = 0.0;
  static double last_fast_glue = 0.0;
  static uint64_t last_conflicts_seen = 0;
  static uint64_t conflict_at_last_restart = 0;
  static uint64_t last_restarts_count = 0;

  // Handle solver reset or new instance (heuristic detection)
  if (CONFLICTS < last_conflicts_seen) {
    lbd_variance = 0.0;
    last_fast_glue = 0.0;
    last_conflicts_seen = 0;
    conflict_at_last_restart = 0;
    last_restarts_count = 0;
  }

  // Sync with external restarts or restarts triggered by other means
  if (solver->statistics.restarts > last_restarts_count) {
    last_restarts_count = solver->statistics.restarts;
    // If we missed the exact restart moment, ensure we don't calculate 
    // an excessively large 'conflicts_since_restart'
    if (conflict_at_last_restart < CONFLICTS)
      conflict_at_last_restart = CONFLICTS;
  }

  // Step 1: Augment the solver state to track `lbd_variance`
  // We approximate the "current LBD" deviation by observing the change in the fast EMA.
  const double fast = AVERAGE (fast_glue);
  const double slow = AVERAGE (slow_glue);

  if (CONFLICTS > last_conflicts_seen) {
    if (last_conflicts_seen == 0) {
        last_fast_glue = fast;
    }

    // Reconstruct approximate deviation from EMA change:
    // fast_new = (1-alpha)*fast_old + alpha*sample
    // sample - fast_old = (fast_new - fast_old) / alpha
    double ema_window = (double) GET_OPTION (emafast);
    double alpha_fast = 1.0 / ema_window;
    
    double diff = fast - last_fast_glue;
    double estimated_deviation = diff * ema_window; // equivalent to diff / alpha

    // Update variance EMA (alpha=0.05)
    double var_alpha = 0.05;
    lbd_variance += var_alpha * ((estimated_deviation * estimated_deviation) - lbd_variance);

    last_fast_glue = fast;
    last_conflicts_seen = CONFLICTS;
  }

  // Standard global conflict limit check (preserves restartint logic)
  if (CONFLICTS < solver->limits.restart.conflicts)
    return false;

  // Stable mode: Reluctant Doubling (preserve existing strategy for stable)
  if (solver->stable) {
    bool res = kissat_reluctant_triggered (&solver->reluctant);
    if (res) {
        conflict_at_last_restart = CONFLICTS;
        // last_restarts_count will be updated on next call via check above
    }
    return res;
  }

  // Step 2: Calculate Coefficient of Variation (CV)
  double cv = sqrt (lbd_variance) / (slow + 1e-9);

  // Step 3: Determine dynamic threshold scaler S
  double S = 1.0;
  if (cv > 0.5)
    S = 1.15; // High volatility: suppress restarts
  else if (cv < 0.15)
    S = 0.85; // Low volatility: encourage restarts

  // Step 4: Dynamic conflict floor and Trigger
  double min_k = fast * 3.5;
  uint64_t conflicts_since_restart = CONFLICTS - conflict_at_last_restart;

  // Trigger only if we passed the floor AND the LBD trend justifies it
  bool trigger = (conflicts_since_restart > min_k) && (fast > slow * S);

  if (trigger) {
    kissat_extremely_verbose (solver,
        "restart triggered: fast=%.2f slow=%.2f CV=%.2f S=%.2f since=%" PRIu64 " floor=%.2f",
        fast, slow, cv, S, conflicts_since_restart, min_k);
    conflict_at_last_restart = CONFLICTS;
  }

  return trigger;
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
