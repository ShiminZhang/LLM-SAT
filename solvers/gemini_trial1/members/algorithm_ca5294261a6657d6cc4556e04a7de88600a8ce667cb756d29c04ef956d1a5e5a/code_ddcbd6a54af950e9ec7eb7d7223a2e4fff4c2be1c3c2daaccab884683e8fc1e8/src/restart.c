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

  // Static state for Clamped MACD-LBD Momentum Restarting algorithm.
  // Using static variables as we cannot modify the solver struct.
  static double fast_ema = 0.0;
  static double slow_ema = 0.0;
  static double last_solver_fast_glue = 0.0;
  static uint64_t last_restarts_count = 0;
  static uint64_t last_restart_at_conflicts = 0;
  static bool initialized = false;

  // Handle reset logic (e.g., if solver is reset or new instance starts)
  if (CONFLICTS == 0 && initialized) {
    fast_ema = 0.0;
    slow_ema = 0.0;
    last_solver_fast_glue = 0.0;
    last_restarts_count = 0;
    last_restart_at_conflicts = 0;
    initialized = false;
  }

  // Step 1: Maintain two persistent EMAs
  double current_solver_fast = AVERAGE (fast_glue);

  if (!initialized) {
    // Initialization
    fast_ema = current_solver_fast;
    slow_ema = current_solver_fast;
    last_solver_fast_glue = current_solver_fast;
    last_restarts_count = GET (restarts);
    last_restart_at_conflicts = CONFLICTS;
    initialized = true;
  } else {
    // Detect if solver's internal EMA changed to infer the latest LBD input.
    // This acts as a proxy since we cannot hook directly into the analyze function.
    if (current_solver_fast != last_solver_fast_glue) {
      double window = (double) GET_OPTION (emafast);
      double alpha_solver = 1.0 / window;
      
      // Reverse engineer the input: input = (new - old) / alpha + old
      double input_lbd = (current_solver_fast - last_solver_fast_glue) / alpha_solver + last_solver_fast_glue;

      // Clamp input LBD to max 20
      if (input_lbd > 20.0) input_lbd = 20.0;
      if (input_lbd < 0.0) input_lbd = 0.0;

      // Update custom EMAs
      // Fast EMA alpha = 0.05
      fast_ema += 0.05 * (input_lbd - fast_ema);
      // Slow EMA alpha = 0.002
      slow_ema += 0.002 * (input_lbd - slow_ema);

      last_solver_fast_glue = current_solver_fast;
    }
  }

  // Update restart tracking to handle Step 3
  uint64_t current_restarts = GET (restarts);
  if (current_restarts > last_restarts_count) {
    last_restarts_count = current_restarts;
    last_restart_at_conflicts = CONFLICTS;
  }

  // Step 3: Enforce minimum run length of 50 conflicts
  if ((CONFLICTS - last_restart_at_conflicts) < 50)
    return false;

  // Step 2: Calculate Momentum Ratio R
  double R = 1.0;
  if (slow_ema > 1e-9)
    R = fast_ema / slow_ema;

  kissat_extremely_verbose (solver,
                            "MACD Restart: R=%.2f (fast=%.2f, slow=%.2f) mode=%s",
                            R, fast_ema, slow_ema, solver->stable ? "stable" : "focused");

  // Step 5: Stable Mode
  if (solver->stable) {
    bool reluctant = kissat_reluctant_triggered (&solver->reluctant);
    // Restart if R > 1.35 OR reluctant doubling triggered
    if (R > 1.35) return true;
    return reluctant;
  }

  // Step 4: Focused Mode
  // Trigger restart if R > 1.15
  if (R > 1.15) return true;
  // Inhibit restart if R < 0.85
  if (R < 0.85) return false;

  return false;
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
