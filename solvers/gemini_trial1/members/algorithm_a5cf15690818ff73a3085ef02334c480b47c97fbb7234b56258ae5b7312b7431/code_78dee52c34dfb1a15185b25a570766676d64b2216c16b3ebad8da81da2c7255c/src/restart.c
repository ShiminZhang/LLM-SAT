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

  // Preserve Reluctant Doubling for Stable Mode
  if (solver->stable)
    return kissat_reluctant_triggered (&solver->reluctant);

  // --- Tiered LBD-Sensitive Restart Protection Implementation ---

  // Static state for Efficiency Tracking
  static uint64_t last_restarts_count = 0;
  static uint64_t base_props = 0;
  static uint64_t base_confs = 0;
  static double average_efficiency = 0.0;
  static bool avg_init = false;

  // Static state for Tiered Glue Protection (LBD detection)
  static uint64_t saved_used[2][128]; // Snapshot of LBD histogram: [mode][glue]
  static bool hist_init = false;
  static uint64_t protect_until = 0;

  // Initialize histogram snapshot on first run
  if (!hist_init) {
    for (int m = 0; m < 2; m++) {
      for (int g = 0; g < 128; g++) {
        saved_used[m][g] = solver->statistics.used[m].glue[g];
      }
    }
    hist_init = true;
  }

  // Step 1: Calculate interval_efficiency
  uint64_t current_restarts = solver->statistics.restarts;
  uint64_t current_props = solver->statistics.propagations;
  uint64_t current_confs = CONFLICTS;

  // Detect new restart phase
  if (current_restarts > last_restarts_count) {
    last_restarts_count = current_restarts;
    base_props = current_props;
    base_confs = current_confs;
    // Note: protect_until persists across restarts based on conflict count
  }

  double interval_efficiency = 0.0;
  uint64_t delta_props = current_props - base_props;
  uint64_t delta_confs = current_confs - base_confs;

  if (delta_confs > 0) {
    interval_efficiency = (double) delta_props / (double) delta_confs;
  }

  // Step 2: Update average_efficiency (EMA alpha=0.01)
  if (!avg_init) {
    average_efficiency = interval_efficiency;
    avg_init = true;
  } else if (delta_confs > 0) {
    // Update EMA
    average_efficiency += 0.01 * (interval_efficiency - average_efficiency);
  }

  // Step 3: Determine dynamic scaling factor S
  double S = 1.0;
  if (avg_init) {
    if (interval_efficiency > 1.1 * average_efficiency) {
      S = 1.2; // High efficiency: inhibit restart
    } else if (interval_efficiency < 0.9 * average_efficiency) {
      S = 0.8; // Thrashing: encourage restart
    }
  }

  // Step 4: Check standard Glucose restart condition
  const double fast = AVERAGE (fast_glue);
  const double slow = AVERAGE (slow_glue);
  
  // trigger = fast_LBD_EMA > (S * 0.8 * slow_LBD_EMA)
  bool trigger = (fast > (S * 0.8 * slow));

  // Step 5: Tiered Glue Protection
  int last_lbd = -1;

  // Detect most recently learned clause by checking histogram changes
  for (int m = 0; m < 2; m++) {
    for (int g = 0; g < 128; g++) {
      uint64_t curr = solver->statistics.used[m].glue[g];
      if (curr > saved_used[m][g]) {
        last_lbd = g;
        saved_used[m][g] = curr;
      }
    }
  }

  if (last_lbd != -1) {
    int W = 0;
    if (last_lbd <= 2) {
      W = 80;
    } else if (last_lbd == 3) {
      W = 30;
    }
    
    if (W > 0) {
      protect_until = CONFLICTS + W;
    }
  }

  // Apply protection
  if (CONFLICTS < protect_until) {
    trigger = false;
  }

  kissat_extremely_verbose (solver,
                            "restart check: eff=%.2f avg=%.2f S=%.1f "
                            "fast=%.2f slow=%.2f limit=%.2f protect=%" PRIu64 " -> %s",
                            interval_efficiency, average_efficiency, S,
                            fast, slow, (S * 0.8 * slow), protect_until,
                            trigger ? "YES" : "NO");

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
