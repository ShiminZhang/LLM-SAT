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

void restart_mab (kissat *solver) {
  /* Multi-armed bandit (Thompson sampling) to pick a restart heuristic.
     This function is called from 'kissat_restart' (stable mode only). */

  /* Keep state across calls. */
  static double S[3] = { 1.0, 1.0, 1.0 };
  static double F[3] = { 1.0, 1.0, 1.0 };

  static double global_moving_average = 0.0;
  static double max_R = 1e-6;

  static uint64_t last_window_conflicts = 0;
  static uint64_t last_decay_conflicts = 0;

  /* We cannot use 'solver->statistics.clauses_reduced' (does not exist in this
     code base).  Use a safe proxy based only on available signals. */
  static unsigned chosen_arm = 0;
  static double accumulated_ge = 0.0;
  static uint64_t ge_count = 0;

  static bool initialized = false;

  if (!initialized) {
    last_window_conflicts = CONFLICTS;
    last_decay_conflicts = CONFLICTS;
    initialized = true;
  }

  /* Geometric-efficiency proxy based on glue and current decision level. */
  const double current_lbd = AVERAGE (fast_glue);
  const double ge = current_lbd / (current_lbd + (double) solver->level + 1.0);
  accumulated_ge += ge;
  ge_count++;

  /* Evaluate and update bandit every window. */
  if (CONFLICTS >= last_window_conflicts + GE_WINDOW_SIZE) {
    const double average_ge =
        ge_count ? (accumulated_ge / (double) ge_count) : 0.0;

    /* Reward proxy: higher GE and lower average LBD is better.
       (No clause-deletion statistic available here.) */
    double average_lbd = AVERAGE (slow_glue);
    if (average_lbd < 1.0)
      average_lbd = 1.0;

    const double reward_R = average_ge / average_lbd;

    if (reward_R > max_R)
      max_R = reward_R;

    if (reward_R > global_moving_average) {
      const double bonus = (max_R > 0) ? (reward_R / max_R) : 0.0;
      S[chosen_arm] += 1.0 + bonus;
    } else {
      F[chosen_arm] += 1.0;
    }

    global_moving_average = 0.95 * global_moving_average + 0.05 * reward_R;

    /* Thompson sampling: sample Beta(S[i],F[i]) via Gamma sums. */
    double max_theta = -1.0;
    unsigned best_arm = 0;

    for (unsigned i = 0; i < 3; i++) {
      double x = 0.0;
      for (unsigned j = 0; j < (unsigned) S[i]; j++)
        x -= log (kissat_pick_double (&solver->random) + 1e-9);

      double y = 0.0;
      for (unsigned j = 0; j < (unsigned) F[i]; j++)
        y -= log (kissat_pick_double (&solver->random) + 1e-9);

      const double theta = (x + y > 0.0) ? (x / (x + y)) : 0.0;
      if (theta > max_theta) {
        max_theta = theta;
        best_arm = i;
      }
    }

    chosen_arm = best_arm;

    /* Reset window accumulators. */
    last_window_conflicts = CONFLICTS;
    accumulated_ge = 0.0;
    ge_count = 0;
  }

  /* Decay counts occasionally to adapt. */
  if (CONFLICTS >= last_decay_conflicts + GE_DECAY_INTERVAL) {
    for (unsigned i = 0; i < 3; i++) {
      S[i] *= GE_DECAY_FACTOR;
      F[i] *= GE_DECAY_FACTOR;
      if (S[i] < 1.0)
        S[i] = 1.0;
      if (F[i] < 1.0)
        F[i] = 1.0;
    }
    last_decay_conflicts = CONFLICTS;
  }

  /* Map chosen arm to solver heuristic.
     Keep it conservative: only use heuristics already supported by the solver.
     Here we assume:
       0 -> keep current heuristic (no change)
       1 -> switch to 1
       2 -> switch to 2
     If your solver uses different heuristic IDs, adjust these constants. */
  if (chosen_arm == 0) {
    /* no change */
  } else if (chosen_arm == 1) {
    solver->heuristic = 1;
  } else {
    solver->heuristic = 2;
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
