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

  // --- Cooldown-Gated Turbulent Restart State ---
  // Static variables used to maintain state without modifying solver struct.
  // Note: These persist across function calls.
  static double lbd_var_ema = 0.0;
  static uint64_t last_forced_restart = 0;
  static uint64_t last_conflict_count = 0;
  static int consecutive_skips = 0;
  static bool last_stable = false;

  // Reset EMA on mode switch to avoid artifacts from statistical shifts
  if (solver->stable != last_stable) {
      lbd_var_ema = 0.0;
      last_stable = solver->stable;
  }

  // Step 1: Update LBD Variance EMA
  // We use the difference between the fast (short-term) and slow (long-term)
  // glue averages as a proxy for the variance/instability of the search.
  uint64_t current_conflicts = CONFLICTS;
  double fast = AVERAGE (fast_glue);
  double slow = AVERAGE (slow_glue);

  // Update only if conflicts have progressed
  if (current_conflicts > last_conflict_count) {
      double diff = fast - slow; 
      lbd_var_ema = 0.95 * lbd_var_ema + 0.05 * (diff * diff);
      last_conflict_count = current_conflicts;
  }

  // Step 2 & 3: Define Search States
  // Laminar: Stable trajectory (low fast glue, low variance)
  bool laminar = (fast < 0.75 * slow) && (lbd_var_ema < 1.0);
  // Turbulent: Chaotic search (high fast glue, high variance)
  bool turbulent = (fast > 1.1 * slow) && (lbd_var_ema > 3.5);

  // Calculate Standard Restart Condition
  // This mirrors the baseline logic but separates calculation from return
  bool standard_trigger = false;
  bool schedule_reached = (current_conflicts >= solver->limits.restart.conflicts);

  const double margin = (100.0 + GET_OPTION (restartmargin)) / 100.0;
  const double limit = margin * slow;

  if (schedule_reached) {
      if (solver->stable) {
          standard_trigger = kissat_reluctant_triggered (&solver->reluctant);
      } else {
          standard_trigger = (limit <= fast);
          // Logging preserved from baseline
          kissat_extremely_verbose (solver,
                                    "restart glue limit %g = "
                                    "%.02f * %g (slow glue) %c %g (fast glue)",
                                    limit, margin, slow,
                                    (limit > fast    ? '>'
                                     : limit == fast ? '='
                                                     : '<'),
                                    fast);
      }
  }

  // Step 4: Laminar Override
  // If the standard condition suggests a restart, we check if we should
  // delay it to exploit a laminar (high-quality) search phase.
  if (standard_trigger) {
      if (laminar) {
          if (consecutive_skips < 5) {
              consecutive_skips++;
              kissat_extremely_verbose(solver, "Laminar override: skipping restart (%d/5)", consecutive_skips);
              return false; // OVERRIDE: Skip restart
          }
          consecutive_skips = 0; // Reset after limit
      } else {
          consecutive_skips = 0; // Reset if not laminar
      }
      return true; // Proceed with standard restart
  }

  // Step 5: Turbulent Override (Cooldown Valve)
  // If standard condition did NOT trigger (either schedule not reached or heuristic said no),
  // we check if we should force a restart to escape turbulence.
  if (turbulent) {
      if (current_conflicts >= last_forced_restart + 200) {
          last_forced_restart = current_conflicts;
          kissat_extremely_verbose(solver, "Turbulent override: forcing restart");
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
