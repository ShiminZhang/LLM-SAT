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
    // Reset MAB tracking variables
    // This cleanup is necessary to ensure the underlying heuristics (like CHB)
    // start fresh for the next interval, consistent with Kissat's MAB implementation.
    for (all_variables(idx)) {
        solver->mab_chosen[idx] = 0;
    }
    solver->mab_chosen_tot = 0;
    solver->mab_decisions = 0;
    solver->mab_conflicts = 0;

    // Step 1: Calculate the raw performance metric 'P'
    // P = 10000 / (average_LBD * sqrt(average_trail_level) + 1)
    // We use fast_glue as the proxy for the "just-concluded restart interval" LBD.
    const double avg_lbd = AVERAGE(fast_glue);
    const double avg_trail = AVERAGE(level);
    
    // Ensure input to sqrt is non-negative
    const double safe_avg_trail = (avg_trail < 0) ? 0 : avg_trail;
    const double P = 10000.0 / (avg_lbd * sqrt(safe_avg_trail) + 1.0);

    // Step 2: Update the exponential moving average Score 'S'
    // S_new = 0.8 * S_old + 0.2 * P
    const unsigned h = solver->heuristic;
    const double S_old = solver->mab_reward[h];
    const double S_new = 0.8 * S_old + 0.2 * P;
    
    solver->mab_reward[h] = S_new;

    // Step 3: Calculate the discrete derivative 'Delta'
    const double Delta = S_new - S_old;

    // Step 4: Execute Mode Selection
    // Calculate Relative Change 'R' = Delta / S_old
    double R = 0.0;
    if (S_old > 1e-9) {
        R = Delta / S_old;
    }

    // Threshold check: R > -0.05 (Improvement/Stagnation) vs R <= -0.05 (Degradation)
    if (R <= -0.05) {
        // Significant Degradation
        // Compare current score 'S_new' against the inactive mode's score 'S_inactive'
        // Assuming binary heuristics (0 and 1)
        const unsigned other_h = h ^ 1;
        const double S_inactive = solver->mab_reward[other_h];

        if (S_new < S_inactive) {
            // Switch mode immediately (Current mode is degrading and historically worse)
            solver->heuristic = other_h;
        } else {
            // Switch with probability 'prob' = min(1.0, abs(R) * 10.0)
            double abs_R = (R < 0) ? -R : R;
            double prob = abs_R * 10.0;
            if (prob > 1.0) prob = 1.0;

            if (kissat_pick_double(&solver->random) < prob) {
                solver->heuristic = other_h;
            }
        }
    }
    // Else: Deterministically keep the current mode to preserve search locality.

    // Update selection count for the chosen heuristic
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
