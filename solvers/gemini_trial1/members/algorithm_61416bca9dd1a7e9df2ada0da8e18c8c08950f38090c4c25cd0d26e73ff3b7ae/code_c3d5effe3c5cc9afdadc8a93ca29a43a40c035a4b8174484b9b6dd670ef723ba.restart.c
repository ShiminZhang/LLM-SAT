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

  // Hard constraints (cannot override)
  if (!GET_OPTION (restart))
    return false;
  if (!solver->level)
    return false;

  // --- Step 1: Maintain LBD Variance EMA ---
  // We use static variables to maintain state across calls since we cannot modify the solver struct.
  // Note: This assumes single-threaded usage per process or external synchronization, which is standard for Kissat.
  static double lbd_var_ema = 1.0;
  static double last_fast_glue = 0.0;
  static uint64_t last_conflicts = 0;
  static bool initialized = false;
  static int laminar_skips = 0;

  const uint64_t conflicts = CONFLICTS;
  const double fast = AVERAGE (fast_glue);
  const double slow = AVERAGE (slow_glue);

  if (!initialized) {
      lbd_var_ema = 1.0;
      last_fast_glue = fast;
      last_conflicts = conflicts;
      initialized = true;
  }

  // Update metrics on every new conflict
  if (conflicts > last_conflicts) {
      // Reverse engineer the raw LBD deviation from the change in the fast EMA.
      // EMA formula: new = old + alpha * (raw - old)
      // Therefore: raw - new = (new - old) * (1/alpha - 1)
      // This gives us (current_lbd - fast_lbd_ema) directly.
      
      double window = (double) GET_OPTION (emafast);
      if (window < 1.0) window = 1.0; // Safety check
      
      double change = fast - last_fast_glue;
      double raw_diff = change * (window - 1.0);
      
      // Clamp the difference: min(abs(raw_diff), 5.0)
      double abs_diff = (raw_diff < 0) ? -raw_diff : raw_diff;
      double clamped_diff = (abs_diff > 5.0) ? 5.0 : abs_diff;
      
      // Update Variance EMA with alpha=0.15
      lbd_var_ema = 0.85 * lbd_var_ema + 0.15 * (clamped_diff * clamped_diff);
      
      last_fast_glue = fast;
      last_conflicts = conflicts;
  }

  // --- Determine Standard Kissat Restart Condition ---
  bool standard_decision = false;
  
  if (conflicts >= solver->limits.restart.conflicts) {
      if (solver->stable) {
          standard_decision = kissat_reluctant_triggered (&solver->reluctant);
      } else {
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
                            
          standard_decision = (limit <= fast);
      }
  }

  // --- Step 2 & 4: Laminar Search Override ---
  // If standard restart condition is TRUE, check if we should suppress it.
  if (standard_decision) {
      // Laminar condition: fast < 0.75*slow AND var < 1.0
      bool laminar = (fast < 0.75 * slow) && (lbd_var_ema < 1.0);
      
      if (laminar) {
          if (laminar_skips < 5) {
              laminar_skips++;
              kissat_extremely_verbose(solver, "Laminar override: skipping restart (%d/5)", laminar_skips);
              return false; // OVERRIDE: Skip restart to extend burst
          }
      }
      
      // If we proceed to restart, reset skips
      laminar_skips = 0;
      return true;
  }

  // --- Step 3 & 5: Turbulent Search Override ---
  // If standard restart condition is FALSE, check if we should force it.
  if (!standard_decision) {
      // Turbulent condition: fast > 1.1*slow AND var > 3.5
      bool turbulent = (fast > 1.1 * slow) && (lbd_var_ema > 3.5);
      
      if (turbulent) {
          kissat_extremely_verbose(solver, "Turbulent override: forcing restart (var %.2f)", lbd_var_ema);
          laminar_skips = 0; // Reset skips on any restart
          return true; // OVERRIDE: Force restart to escape local minimum
      }
  }

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
