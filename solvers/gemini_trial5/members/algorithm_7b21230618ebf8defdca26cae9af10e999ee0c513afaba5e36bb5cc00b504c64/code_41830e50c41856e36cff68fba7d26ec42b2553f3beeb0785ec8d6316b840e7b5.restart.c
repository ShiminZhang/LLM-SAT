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
    // Step 1: Calculate reward R for the just-completed restart interval
    // Algorithm: R = (conflicts / max(1, sum_of_LBDs))
    // Since sum_of_LBDs is not directly tracked, we approximate it using the 
    // exponential moving average of glues: sum ~ conflicts * AVERAGE(fast_glue).
    // Therefore, R ~ 1.0 / AVERAGE(fast_glue).
    
    double avg_glue = AVERAGE(fast_glue);
    if (avg_glue < 1.0) avg_glue = 1.0; // Safety floor
    double R = 1.0 / avg_glue;

    // Step 2: Update persistent Exponential Moving Averages (EMA)
    // Fast_EMA (alpha=0.3) and Slow_EMA (alpha=0.05)
    static double fast_ema[2] = {0};
    static double slow_ema[2] = {0};
    static bool initialized = false;
    
    unsigned current_h = solver->heuristic;
    // Safety check for heuristic index
    if (current_h >= 2) current_h = 0;

    if (!initialized) {
        // Initialize EMAs on first run with current R to avoid zero-bias
        for (int i = 0; i < 2; i++) {
            fast_ema[i] = R;
            slow_ema[i] = R;
        }
        initialized = true;
    } else {
        // Update EMAs for the currently active mode
        fast_ema[current_h] += 0.3 * (R - fast_ema[current_h]);
        slow_ema[current_h] += 0.05 * (R - slow_ema[current_h]);
    }

    // Step 3: Update global Stagnation_Counter
    // Increment if zero new glue clauses (LBD <= 2) were learned during this interval.
    // We access the stable mode statistics (index 1) for glue/LBD counts.
    // We sum counts for LBD 1 and LBD 2.
    static uint64_t last_lbd2_count = 0;
    static int stagnation_counter = 0;
    static bool stats_initialized = false;

    // solver->statistics.used[1] refers to stable mode statistics
    uint64_t current_lbd2_count = solver->statistics.used[1].glue[1] + 
                                  solver->statistics.used[1].glue[2];

    if (!stats_initialized) {
        last_lbd2_count = current_lbd2_count;
        stats_initialized = true;
        stagnation_counter = 0; 
    } else {
        uint64_t new_lbd2 = current_lbd2_count - last_lbd2_count;
        if (new_lbd2 == 0) {
            stagnation_counter++;
        } else {
            stagnation_counter = 0;
        }
        last_lbd2_count = current_lbd2_count;
    }

    // Step 4: Calculate Priority P for each available mode
    double P[2] = {0};
    unsigned num_heuristics = solver->mab_heuristics;
    if (num_heuristics > 2) num_heuristics = 2; // Clamp to array size

    for (unsigned i = 0; i < num_heuristics; i++) {
        double fast = fast_ema[i];
        double slow = slow_ema[i];
        
        // Asymmetric momentum: exploit positive breakouts
        if (fast > slow) {
            P[i] = fast + 2.0 * (fast - slow);
        } else {
            P[i] = fast;
        }

        // Step 5: Apply penalty if Stagnation_Counter > 6 and mode is currently active
        if (i == current_h && stagnation_counter > 6) {
            P[i] *= 0.5;
        }
    }

    // Step 6: Select the next restart mode corresponding to the highest Priority P
    unsigned best_h = current_h;
    double max_p = -1.0; 

    for (unsigned i = 0; i < num_heuristics; i++) {
        if (P[i] > max_p) {
            max_p = P[i];
            best_h = i;
        }
    }

    solver->heuristic = best_h;
    solver->mab_select[best_h]++;

    // Housekeeping: Reset MAB interval tracking variables
    // This ensures the next interval's stats are calculated correctly
    solver->mab_decisions = 0;
    solver->mab_conflicts = 0;
    solver->mab_chosen_tot = 0;
    
    // Clear per-variable chosen counts (standard Kissat MAB practice)
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
