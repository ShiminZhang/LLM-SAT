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

  // --- Asymmetric Efficiency-Trend Protected Restarts ---

  // Static state variables for efficiency tracking
  static double average_efficiency = 0.0;
  static uint64_t last_restarts_count = 0;
  static uint64_t props_at_start = 0;
  static uint64_t confs_at_start = 0;
  static double last_seen_eff = 0.0;

  // Static state variables for glue protection
  static uint64_t prev_glue2_count = 0;
  static uint64_t last_glue2_conflict = 0;
  static bool state_initialized = false;

  // Heuristic reset detection: if global conflicts are lower than our start marker,
  // the solver has likely been reset. Reset static state.
  if (CONFLICTS < confs_at_start) {
      average_efficiency = 0.0;
      last_restarts_count = 0;
      props_at_start = 0;
      confs_at_start = 0;
      last_seen_eff = 0.0;
      prev_glue2_count = 0;
      last_glue2_conflict = 0;
      state_initialized = false;
  }

  // Initialize state on first run
  if (!state_initialized) {
      props_at_start = GET (propagations);
      confs_at_start = CONFLICTS;
      // Sum LBD 2 usage from both focused (0) and stable (1) modes
      prev_glue2_count = solver->statistics.used[0].glue[2] + solver->statistics.used[1].glue[2];
      state_initialized = true;
  }

  // Step 2: Update average_efficiency on restart boundaries
  // We detect a completed interval if the global restart count increased
  uint64_t current_restarts = GET (restarts);
  if (current_restarts > last_restarts_count) {
      if (average_efficiency == 0.0) {
          average_efficiency = last_seen_eff;
      } else {
          // Asymmetric EMA update
          // If efficiency is rising (interval > average), adapt fast (alpha=0.05)
          // If efficiency is dropping, decay slow (alpha=0.001)
          double alpha = (last_seen_eff > average_efficiency) ? 0.05 : 0.001;
          average_efficiency = (1.0 - alpha) * average_efficiency + alpha * last_seen_eff;
      }
      
      // Start new interval
      props_at_start = GET (propagations);
      confs_at_start = CONFLICTS;
      last_restarts_count = current_restarts;
  }

  // Step 1: Calculate interval_efficiency
  uint64_t d_props = GET (propagations) - props_at_start;
  uint64_t d_confs = CONFLICTS - confs_at_start;

  // Prevent division by zero or immediate triggers
  if (d_confs == 0) return false;

  double interval_efficiency = (double) d_props / (double) d_confs;
  last_seen_eff = interval_efficiency; // Save for next update

  // Step 3: Determine Scaling Factor S
  double S = 1.0;
  if (average_efficiency > 1e-9) {
      if (interval_efficiency > 1.1 * average_efficiency)
          S = 1.2; // Inhibit restart (high efficiency)
      else if (interval_efficiency < 0.9 * average_efficiency)
          S = 0.8; // Encourage restart (thrashing)
  }

  // Step 5: Glue Protection
  // Check if a clause with LBD <= 2 (specifically LBD=2) was learned recently
  uint64_t current_glue2 = solver->statistics.used[0].glue[2] + solver->statistics.used[1].glue[2];
  
  if (current_glue2 > prev_glue2_count) {
      last_glue2_conflict = CONFLICTS;
      prev_glue2_count = current_glue2;
  }

  // If LBD<=2 learned within last 50 conflicts, force trigger = false
  if (CONFLICTS >= last_glue2_conflict && (CONFLICTS - last_glue2_conflict < 50)) {
      return false;
  }

  // Step 4: Check Trigger
  const double fast = AVERAGE (fast_glue);
  const double slow = AVERAGE (slow_glue);
  
  // Standard Glucose condition modified by S and factor 0.8
  double limit = S * 0.8 * slow;

  kissat_extremely_verbose (solver,
      "restart check eff=%.2f avg=%.2f S=%.2f limit=%.2f fast=%.2f",
      interval_efficiency, average_efficiency, S, limit, fast);

  return (fast > limit);
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
