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
  
  // Basic restart checks
  if (!GET_OPTION (restart))
    return false;
  if (!solver->level)
    return false;
  
  // Respect the global restart scheduling limit (conflict count)
  if (CONFLICTS < solver->limits.restart.conflicts)
    return false;

  // Stable mode uses Reluctant Doubling (geometric restarts)
  if (solver->stable)
    return kissat_reluctant_triggered (&solver->reluctant);

  // Focused mode: Decision-Cost Augmented Efficiency Restarts

  // Static state variables to track interval efficiency and glue protection
  // (Using static variables to avoid modifying solver struct)
  static uint64_t last_restarts_mark = 0;
  static uint64_t base_propagations = 0;
  static uint64_t base_decisions = 0;
  static uint64_t base_conflicts = 0;
  static double average_efficiency = 0.0;
  static bool avg_eff_initialized = false;

  static uint64_t last_lbd2_total_count = 0;
  static uint64_t last_lbd2_conflict = 0;
  static bool lbd_initialized = false;

  // Get current statistics
  const uint64_t current_restarts = GET (restarts);
  const uint64_t current_props = GET (propagations);
  const uint64_t current_decs = GET (decisions);
  const uint64_t current_confs = CONFLICTS;

  // Detect restart occurrence to reset interval baselines
  if (current_restarts > last_restarts_mark) {
    last_restarts_mark = current_restarts;
    base_propagations = current_props;
    base_decisions = current_decs;
    base_conflicts = current_confs;
  }

  // Step 1: Calculate interval_efficiency
  // Ratio of (propagations since last restart) to (decisions + conflicts + 1 since last restart)
  uint64_t delta_props = current_props - base_propagations;
  uint64_t delta_decs = current_decs - base_decisions;
  uint64_t delta_confs = current_confs - base_conflicts;

  double interval_efficiency = (double) delta_props / (double) (delta_decs + delta_confs + 1);

  // Step 2: Update global average_efficiency (EMA alpha=0.01)
  if (!avg_eff_initialized) {
    average_efficiency = interval_efficiency;
    avg_eff_initialized = true;
  } else {
    average_efficiency += 0.01 * (interval_efficiency - average_efficiency);
  }

  // Step 3: Determine dynamic scaling factor S
  double S = 1.0;
  // Check against average efficiency (guard against zero)
  if (average_efficiency > 1e-9) {
    if (interval_efficiency > 1.1 * average_efficiency) {
      S = 1.2; // High efficiency: inhibit restart
    } else if (interval_efficiency < 0.9 * average_efficiency) {
      S = 0.8; // Low efficiency (drifting/thrashing): encourage restart
    }
  }

  // Step 4: Check the standard Glucose restart condition
  // trigger = fast_LBD_EMA > (S * 0.8 * slow_LBD_EMA)
  const double fast = AVERAGE (fast_glue);
  const double slow = AVERAGE (slow_glue);
  
  bool trigger = (fast > (S * 0.8 * slow));

  // Step 5: Implement 'Glue Protection'
  // If a clause with LBD <= 2 was learned within the last 50 conflicts, force trigger = false
  
  // Calculate current total of LBD 1 and 2 clauses from usage statistics
  // Summing across both modes [0] and [1] to catch all recent learnings
  uint64_t current_lbd2 = 0;
  current_lbd2 += solver->statistics.used[0].glue[1] + solver->statistics.used[0].glue[2];
  current_lbd2 += solver->statistics.used[1].glue[1] + solver->statistics.used[1].glue[2];

  // Initialize LBD counter on first run
  if (!lbd_initialized) {
    last_lbd2_total_count = current_lbd2;
    lbd_initialized = true;
  }

  // If count of low-LBD clauses increased, update timestamp
  if (current_lbd2 > last_lbd2_total_count) {
    last_lbd2_total_count = current_lbd2;
    last_lbd2_conflict = current_confs;
  }

  // Check protection window (50 conflicts)
  if (current_confs - last_lbd2_conflict < 50) {
    trigger = false;
  }

  kissat_extremely_verbose (solver,
                            "restart check: eff=%.4f avg=%.4f S=%.1f fast=%.2f slow=%.2f trigger=%d",
                            interval_efficiency, average_efficiency, S, fast, slow, trigger);

  // Step 6: Return trigger
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
