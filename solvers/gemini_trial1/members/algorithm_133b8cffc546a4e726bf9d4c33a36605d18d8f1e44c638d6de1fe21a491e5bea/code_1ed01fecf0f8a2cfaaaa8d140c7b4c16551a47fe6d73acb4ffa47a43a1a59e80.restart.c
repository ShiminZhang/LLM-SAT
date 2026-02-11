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

  // --- Heuristic State (Static) ---
  // Since we cannot modify the solver struct, we use static variables to maintain
  // the state required for the variance heuristic.
  static double lbd_var_ema = 0.0;
  static double last_fast_glue = 0.0;
  static uint64_t last_conflict_count = 0;
  static int extension_count = 0;

  // --- Data Gathering ---
  double fast = AVERAGE (fast_glue);
  double slow = AVERAGE (slow_glue);
  uint64_t conflicts = CONFLICTS;

  // Initialize history on first execution
  if (last_conflict_count == 0) {
      last_fast_glue = fast;
      last_conflict_count = conflicts;
  }

  // --- Step 1: Update Variance EMA ---
  // We approximate the "on every conflict" update by calculating the drift since the last restart check.
  // We reverse-engineer the "current" LBD behavior from the change in the fast EMA.
  
  uint64_t delta = conflicts - last_conflict_count;
  double estimated_current_lbd = fast; // Fallback

  if (delta > 0) {
      double ema_fast_window = (double) GET_OPTION (emafast);
      double alpha = 1.0 / ema_fast_window;
      
      // fast_new = fast_old * (1-alpha)^delta + current_avg * (1 - (1-alpha)^delta)
      // We solve for current_avg to estimate the recent LBD quality.
      double decay = pow(1.0 - alpha, (double)delta);
      double inverted_decay = 1.0 - decay;

      if (inverted_decay > 1e-9) {
          estimated_current_lbd = (fast - last_fast_glue * decay) / inverted_decay;
      }
      
      // Clamp to prevent numerical artifacts
      if (estimated_current_lbd < 1.0) estimated_current_lbd = 1.0;

      // Update Variance: lbd_var_ema = 0.95 * lbd_var_ema + 0.05 * (diff * diff)
      double diff = estimated_current_lbd - fast;
      lbd_var_ema = 0.95 * lbd_var_ema + 0.05 * (diff * diff);

      // Update history
      last_fast_glue = fast;
      last_conflict_count = conflicts;
  }

  // --- Step 2 & 3: Define Search Conditions ---
  // Laminar: stable trajectory finding high-quality clauses
  bool laminar = (fast < 0.75 * slow) && (lbd_var_ema < 1.0);
  // Turbulent: thrashing locally without finding glue clauses
  bool turbulent = (fast > 1.1 * slow) && (lbd_var_ema > 3.5);

  // --- Standard Kissat Restart Logic ---
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
  
  bool standard_restart = (limit <= fast);

  // --- Step 4: Laminar Override ---
  if (standard_restart) {
      if (laminar) {
          // Check for Glue Clause (LBD <= 2)
          if (estimated_current_lbd <= 2.0) {
              extension_count = 0;
              return false; // OVERRIDE: Skip restart (infinite extension)
          }
          // Check for generic extension cap
          if (extension_count < 8) {
              extension_count++;
              return false; // OVERRIDE: Skip restart
          }
      }
      return true; // Allow restart
  }
  
  // --- Step 5: Turbulent Override ---
  if (!standard_restart) {
      if (turbulent) {
          return true; // OVERRIDE: Force restart
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
