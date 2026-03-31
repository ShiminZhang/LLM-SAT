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
  /* Pick a restart heuristic ("arm") using a simple Thompson-sampling style
     bandit.  This function only changes 'solver-&gt;heuristic' and does not
     depend on non-existing statistics fields. */

  /* Heuristic IDs used in this solver build:
     0 = stable (reluctant/Luby-like)
     1 = focused (EMA/Glucose-like)
     2 = aggressive (more frequent restarts; implemented elsewhere) */

  static double S[3] = { 1.0, 1.0, 1.0 };
  static double F[3] = { 1.0, 1.0, 1.0 };

  static uint64_t last_window_conflicts = 0;
  static uint64_t last_decay_conflicts = 0;
  static unsigned chosen_arm = 0;
  static bool initialized = false;

  if (!initialized) {
    last_window_conflicts = CONFLICTS;
    last_decay_conflicts = CONFLICTS;
    chosen_arm = solver->heuristic % 3u;
    initialized = true;
  }

  /* Use a cheap reward proxy based on glue evolution.
     Lower fast glue relative to slow glue is considered better. */
  const double fast = AVERAGE (fast_glue);
  const double slow = AVERAGE (slow_glue);
  const double denom = (slow > 1e-9 ? slow : 1e-9);
  const double reward = 1.0 - fast / denom; /* higher is better */

  if (CONFLICTS >= last_window_conflicts + GE_WINDOW_SIZE) {
    /* Update success/failure for the previously chosen arm. */
    if (reward > 0.0)
      S[chosen_arm] += 1.0;
    else
      F[chosen_arm] += 1.0;

    /* Thompson sampling: sample Beta(S[i],F[i]) via Gamma sums.
       We approximate Gamma(k,1) for integer k by sum of -log(U). */
    double best_theta = -1.0;
    unsigned best_arm = 0;

    for (unsigned i = 0; i < 3; i++) {
      unsigned si = (unsigned) S[i];
      unsigned fi = (unsigned) F[i];
      if (!si) si = 1;
      if (!fi) fi = 1;

      double x = 0.0;
      for (unsigned j = 0; j < si; j++)
        x -= log (kissat_pick_double (&solver->random) + 1e-12);

      double y = 0.0;
      for (unsigned j = 0; j < fi; j++)
        y -= log (kissat_pick_double (&solver->random) + 1e-12);

      const double theta = (x + y > 0.0) ? x / (x + y) : 0.0;
      if (theta > best_theta) {
        best_theta = theta;
        best_arm = i;
      }
    }

    chosen_arm = best_arm;
    solver->heuristic = chosen_arm;

    last_window_conflicts = CONFLICTS;
  }

  /* Decay to keep adaptation responsive. */
  if (CONFLICTS >= last_decay_conflicts + GE_DECAY_INTERVAL) {
    for (unsigned i = 0; i < 3; i++) {
      S[i] *= GE_DECAY_FACTOR;
      F[i] *= GE_DECAY_FACTOR;
      if (S[i] < 1.0) S[i] = 1.0;
      if (F[i] < 1.0) F[i] = 1.0;
    }
    last_decay_conflicts = CONFLICTS;
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
