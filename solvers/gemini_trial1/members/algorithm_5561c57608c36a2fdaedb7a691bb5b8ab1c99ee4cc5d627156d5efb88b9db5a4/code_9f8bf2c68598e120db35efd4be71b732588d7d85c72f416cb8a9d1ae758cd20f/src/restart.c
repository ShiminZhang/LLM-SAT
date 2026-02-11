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

  // --- Algorithm Implementation: Batched LBD-Trend Restarting ---

  // Static state to maintain history across function calls (as we cannot modify solver struct)
  // Note: These statics assume a single-threaded solver execution flow.
  static uint64_t last_used[2][128]; 
  static uint64_t accumulated_lbd = 0;
  static uint64_t accumulated_count = 0;
  static uint64_t batch_start_conflicts = 0;
  static double ema_fast = 0.0;
  static double ema_slow = 0.0;
  static bool emas_initialized = false;
  static double multiplier = 1.0;
  static uint64_t last_conflicts_check = 0;

  // Detect reset or new solver instance (simple heuristic based on conflict count drop)
  if (CONFLICTS < last_conflicts_check) {
    for (int m = 0; m < 2; m++)
      for (int g = 0; g < 128; g++)
        last_used[m][g] = 0;
    accumulated_lbd = 0;
    accumulated_count = 0;
    batch_start_conflicts = CONFLICTS;
    ema_fast = 0.0;
    ema_slow = 0.0;
    emas_initialized = false;
    multiplier = 1.0;
  }
  last_conflicts_check = CONFLICTS;

  // Initialize batch start if needed
  if (batch_start_conflicts == 0 && CONFLICTS > 0)
      batch_start_conflicts = CONFLICTS;

  // Step 1 Part A: Accumulate LBD of learned clauses
  // We infer learned clause LBDs by diffing the global histogram 'solver->statistics.used'
  for (int mode = 0; mode < 2; mode++) {
    for (int g = 0; g < 128; g++) {
      uint64_t current = solver->statistics.used[mode].glue[g];
      uint64_t diff = current - last_used[mode][g];
      if (diff > 0) {
        // g represents the LBD value (or close approximation based on histogram index)
        accumulated_lbd += diff * g;
        accumulated_count += diff;
        last_used[mode][g] = current;
      }
    }
  }

  // Step 1 Part B: Batch Update (every 50 conflicts)
  if (CONFLICTS >= batch_start_conflicts + 50) {
    if (accumulated_count > 0) {
      double mean_lbd = (double)accumulated_lbd / accumulated_count;

      if (!emas_initialized) {
        ema_fast = mean_lbd;
        ema_slow = mean_lbd;
        emas_initialized = true;
      } else {
        // Update EMAs with specified alphas
        ema_fast += 0.15 * (mean_lbd - ema_fast);
        ema_slow += 0.01 * (mean_lbd - ema_slow);
      }

      // Step 2: Compute Trend Ratio R
      double R = (ema_slow > 1e-9) ? (ema_fast / ema_slow) : 1.0;

      // Step 3: Determine Dynamic Multiplier M
      if (R < 0.85) {
        multiplier = 1.4; // Improving: extend run
      } else if (R > 1.15) {
        multiplier = 0.7; // Degrading: curtail run
      } else {
        multiplier = 1.0;
      }
      
      kissat_extremely_verbose(solver, 
        "Batched LBD Update: mean=%.2f, fast=%.2f, slow=%.2f, R=%.2f, M=%.2f", 
        mean_lbd, ema_fast, ema_slow, R, multiplier);
    }

    // Reset batch accumulators
    accumulated_lbd = 0;
    accumulated_count = 0;
    batch_start_conflicts = CONFLICTS;
  }

  // Stable Mode: Defer to Reluctant Doubling (Standard Kissat Behavior)
  if (solver->stable)
    return kissat_reluctant_triggered (&solver->reluctant);

  // Step 4 & 5: Focused Mode Trigger Logic
  // Retrieve scheduled limit and apply dynamic multiplier
  uint64_t limit = solver->limits.restart.conflicts;
  double threshold = (double)limit * multiplier;

  kissat_extremely_verbose (solver,
                            "restart check: conflicts=%" PRIu64 
                            " limit=%" PRIu64 " M=%.2f threshold=%.2f",
                            CONFLICTS, limit, multiplier, threshold);

  // Trigger restart if current conflicts exceed the dynamic threshold
  return (CONFLICTS > threshold);
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
