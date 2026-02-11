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
  
  // Stable mode delegation: Use standard Reluctant Doubling
  if (solver->stable)
    return kissat_reluctant_triggered (&solver->reluctant);

  // --- ACES: Asymmetric Continuous Efficiency Scaling Strategy ---

  // Static state variables to track history across calls
  // Note: These persist across the process lifetime.
  static uint64_t last_restarts = 0;
  static uint64_t props_at_restart = 0;
  static uint64_t confs_at_restart = 0;
  static double average_efficiency = 0.0;
  static uint64_t last_low_glue_count = 0;
  static uint64_t last_low_glue_conflict = 0;

  // Access current solver statistics
  const uint64_t current_restarts = solver->statistics.restarts;
  const uint64_t current_props = solver->statistics.propagations;
  const uint64_t current_confs = CONFLICTS;

  // Reset static state if the solver seems to have been reset (conflict count dropped)
  if (current_confs < confs_at_restart) {
      last_restarts = 0;
      props_at_restart = 0;
      confs_at_restart = 0;
      average_efficiency = 0.0;
      last_low_glue_count = 0;
      last_low_glue_conflict = 0;
  }

  // Step 2: Update global average_efficiency
  // We check if a restart occurred since the last call to this function.
  // If so, the previous interval is complete, and we update the EMA.
  if (current_restarts > last_restarts) {
      double dp = (double)(current_props - props_at_restart);
      double dc = (double)(current_confs - confs_at_restart);
      if (dc > 0) {
          double eff = dp / dc;
          if (average_efficiency == 0.0) {
              average_efficiency = eff;
          } else {
              // Alpha = 0.01 as specified
              average_efficiency = 0.01 * eff + 0.99 * average_efficiency;
          }
      }
      
      // Reset baseline for the new interval
      props_at_restart = current_props;
      confs_at_restart = current_confs;
      last_restarts = current_restarts;
  }

  // Step 1: Calculate interval_efficiency
  // Based on propagations and conflicts since the last restart (current running interval)
  double dp = (double)(current_props - props_at_restart);
  double dc = (double)(current_confs - confs_at_restart);
  double interval_efficiency = (dc > 0) ? (dp / dc) : 0.0;

  // Initialize average if not yet set (first interval)
  if (average_efficiency == 0.0) {
      average_efficiency = (interval_efficiency > 0) ? interval_efficiency : 1.0;
  }

  // Step 3: Calculate Ratio R and Scaling S
  double R = 1.0;
  if (average_efficiency > 1e-9) {
      R = interval_efficiency / average_efficiency;
  }

  double S;
  if (R < 1.0) {
      // Thrashing: S = max(0.6, 1.0 - 0.5 * (1.0 - R))
      double val = 1.0 - 0.5 * (1.0 - R);
      S = (val < 0.6) ? 0.6 : val;
  } else {
      // High efficiency: S = min(1.4, 1.0 + 0.25 * (R - 1.0))
      double val = 1.0 + 0.25 * (R - 1.0);
      S = (val > 1.4) ? 1.4 : val;
  }

  // Step 4: Glucose restart condition with Scaling S
  const double fast = AVERAGE (fast_glue);
  const double slow = AVERAGE (slow_glue);
  const double threshold = S * 0.8 * slow;
  
  bool trigger = (fast > threshold);

  kissat_extremely_verbose (solver,
                            "ACES: R=%.2f S=%.2f fast=%.2f slow=%.2f thr=%.2f trig=%d",
                            R, S, fast, slow, threshold, trigger);

  // Step 5: Glue Protection
  // Check if a clause with LBD <= 2 was learned within the last 50 conflicts.
  // We sum the usage counts for glue 1 and 2 from both modes (0=focused, 1=stable).
  uint64_t current_low_glue = 
      solver->statistics.used[0].glue[1] + solver->statistics.used[0].glue[2] +
      solver->statistics.used[1].glue[1] + solver->statistics.used[1].glue[2];

  // If the count increased, a low-glue clause was recently learned
  if (current_low_glue > last_low_glue_count) {
      last_low_glue_count = current_low_glue;
      last_low_glue_conflict = current_confs;
  }

  // If restart is triggered, check if we are within the protection window
  if (trigger) {
      if (current_confs >= last_low_glue_conflict && 
          (current_confs - last_low_glue_conflict < 50)) {
          trigger = false;
          kissat_extremely_verbose (solver, "ACES: Restart blocked by Glue Protection");
      }
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
