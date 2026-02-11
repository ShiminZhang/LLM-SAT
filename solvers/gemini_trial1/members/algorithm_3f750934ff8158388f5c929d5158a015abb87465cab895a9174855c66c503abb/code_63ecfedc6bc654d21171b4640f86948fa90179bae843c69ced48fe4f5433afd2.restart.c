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
  
  // Static state for MACD Stable-Phase Extension Strategy
  // These persist across calls to maintain the EMAs and restart history
  static double macd_fast_ema = 0.0;
  static double macd_slow_ema = 0.0;
  static uint64_t macd_last_clauses_learned = 0;
  static bool macd_initialized = false;
  static uint64_t conflicts_at_last_restart = 0;
  static uint64_t last_restarts_count = 0;

  // Standard checks from baseline
  assert (solver->unassigned);
  if (!GET_OPTION (restart))
    return false;
  if (!solver->level)
    return false;

  // Sync with external restarts
  // If the global restart count increased without us triggering it, reset our local counter
  if (solver->statistics.restarts > last_restarts_count) {
      conflicts_at_last_restart = CONFLICTS;
      last_restarts_count = solver->statistics.restarts;
  }

  // Step 1: Maintain two persistent EMAs
  // We use the solver's fast glue average as a proxy for the "latest learned clause LBD"
  // because raw access to the last learned clause is not exposed.
  // We update only when new clauses are learned to respect the time-step.
  if (solver->statistics.clauses_learned > macd_last_clauses_learned) {
      double input = AVERAGE (fast_glue);
      
      if (!macd_initialized) {
          macd_fast_ema = input;
          macd_slow_ema = input;
          macd_initialized = true;
          // Initialize conflict tracking
          conflicts_at_last_restart = CONFLICTS;
      } else {
          // fast_ema (alpha=0.05)
          macd_fast_ema += 0.05 * (input - macd_fast_ema);
          // slow_ema (alpha=0.002)
          macd_slow_ema += 0.002 * (input - macd_slow_ema);
      }
      macd_last_clauses_learned = solver->statistics.clauses_learned;
  }

  // Cannot make decisions without initialization
  if (!macd_initialized) return false;

  // Step 2: Calculate the momentum ratio R
  double R = 1.0;
  if (macd_slow_ema > 1e-9) {
      R = macd_fast_ema / macd_slow_ema;
  }

  // Step 3: Enforce a minimum run length
  // If conflicts since the last restart < 50, return false.
  if ((CONFLICTS - conflicts_at_last_restart) < 50) {
      return false;
  }

  bool decision = false;

  if (!solver->stable) {
      // Step 4: Focused Mode
      if (R > 1.15) {
          // Trigger a restart if quality is degrading
          decision = true;
      } else if (R < 0.85) {
          // Inhibit restarting if in a hot streak
          decision = false;
      } else {
          // Default behavior for neutral R in Focused mode (implied inhibit/continue)
          decision = false;
      }
  } else {
      // Step 5: Stable Mode
      
      // Independent emergency restart check
      if (R > 1.40) {
          decision = true;
      } else {
          // Check standard reluctant doubling conflict limit
          if (kissat_reluctant_triggered (&solver->reluctant)) {
              // Only trigger if R >= 0.90
              if (R >= 0.90) {
                  decision = true;
              } else {
                  // If R < 0.90, extend the search
                  decision = false;
              }
          } else {
              // Limit not reached
              decision = false;
          }
      }
  }

  // If we decided to restart, update local tracking
  // We anticipate the increment of solver->statistics.restarts by the caller
  if (decision) {
      conflicts_at_last_restart = CONFLICTS;
      last_restarts_count = solver->statistics.restarts + 1;
  }

  return decision;
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
