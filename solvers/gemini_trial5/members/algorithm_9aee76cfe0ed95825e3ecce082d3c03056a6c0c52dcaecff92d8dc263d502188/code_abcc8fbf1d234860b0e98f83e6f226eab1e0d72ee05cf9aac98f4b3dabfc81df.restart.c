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
  const double fast = AVERAGE (fast_glue);
  const double slow = AVERAGE (slow_glue);
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
  return (limit <= fast);
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
    // Static variables to persist state across function calls
    // Using static allows us to maintain the score state 'S' without modifying the solver struct.
    static double heuristic_scores[2] = {0.0, 0.0};
    static bool initialized = false;

    // Step 1: Calculate the raw performance metric 'P' for the just-concluded restart interval
    // Formula: P = 10000 / (average_LBD * sqrt(average_trail_level) + 1)
    
    // We use the Exponential Moving Averages (EMAs) provided by Kissat.
    // 'fast_glue' tracks the LBD (Literal Block Distance).
    double avg_lbd = AVERAGE(fast_glue);
    
    // 'level' tracks the decision level, which we interpret as 'average_trail_level'.
    // Note: AVERAGE(level) is typically available when !QUIET. 
    double avg_trail_level = AVERAGE(level);

    // Ensure the value under sqrt is non-negative
    if (avg_trail_level < 0.0) avg_trail_level = 0.0;

    // Calculate P
    double denom = avg_lbd * sqrt(avg_trail_level) + 1.0;
    double P = 10000.0 / denom;

    // Initialize scores if this is the first execution
    if (!initialized) {
        heuristic_scores[0] = P;
        heuristic_scores[1] = P;
        initialized = true;
    }

    // Step 2: Update the exponential moving average Score 'S'
    // Identify current heuristic (0=VSIDS, 1=CHB)
    unsigned h = solver->heuristic;
    // Safety check for heuristic index
    if (h >= 2) h = 0;

    double s_old = heuristic_scores[h];
    double s_new;

    // Asymmetric Learning Rate
    if (P > s_old) {
        // Performance Improving: alpha = 0.4
        // S_new = 0.6 * S_old + 0.4 * P
        s_new = 0.6 * s_old + 0.4 * P;
    } else {
        // Performance Degrading: alpha = 0.1
        // S_new = 0.9 * S_old + 0.1 * P
        s_new = 0.9 * s_old + 0.1 * P;
    }

    // Update the score for the current heuristic
    heuristic_scores[h] = s_new;

    // Step 3: Calculate the discrete derivative 'Delta'
    double delta = s_new - s_old;

    // Step 4: Execute Mode Selection
    // "Switch mode" implies switching the heuristic (arm).
    
    // Define a small epsilon for "approx 0" comparison
    double epsilon = 1e-6;

    if (delta > epsilon) {
        // Performance Accelerating (Delta > 0): deterministic keep
        // Do nothing, keep current solver->heuristic
    } 
    else if (delta < -epsilon) {
        // Performance Decelerating (Delta < 0): switch mode with probability
        double abs_delta = fabs(delta);
        
        // prob = min(1.0, abs(Delta) * 5.0)
        double prob = abs_delta * 5.0;
        if (prob > 1.0) prob = 1.0;

        // Switch proportional to the rate of degradation
        double rand_val = kissat_pick_double(&solver->random);
        if (rand_val < prob) {
            // Switch to the other heuristic (assuming 2 heuristics)
            solver->heuristic = 1 - h;
        }
    } 
    else {
        // Delta approx 0: revert to comparison of S_stable vs S_focused
        // We compare the scores stored in heuristic_scores
        if (heuristic_scores[0] > heuristic_scores[1]) {
            solver->heuristic = 0;
        } else if (heuristic_scores[1] > heuristic_scores[0]) {
            solver->heuristic = 1;
        }
        // If equal, keep current
    }

    // Update selection statistics for the chosen heuristic
    if (solver->heuristic < 2) {
        solver->mab_select[solver->heuristic]++;
    }

    // Reset MAB tracking variables for the next interval
    // This ensures the solver's internal MAB counters are clean,
    // even though our algorithm primarily uses global EMAs.
    solver->mab_decisions = 0;
    solver->mab_conflicts = 0;
    solver->mab_chosen_tot = 0;
    
    // Clear per-variable chosen counts (standard practice in restart_mab)
    for (all_variables(idx)) {
        solver->mab_chosen[idx] = 0;
    }
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
