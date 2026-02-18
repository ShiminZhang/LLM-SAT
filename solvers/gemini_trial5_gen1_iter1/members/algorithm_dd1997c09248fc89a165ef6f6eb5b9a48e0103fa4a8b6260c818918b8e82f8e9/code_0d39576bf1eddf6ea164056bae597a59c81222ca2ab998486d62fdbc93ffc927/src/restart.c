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

static void restart_mab (kissat *solver)
{
  /* This file already has a proper `bool kissat_restarting(kissat*)`.
     The injected code accidentally redefined it with a different return type
     and also referenced non-existent fields/macros.  This helper implements
     a lightweight MAB-style heuristic switch using only existing state. */

  assert (solver);
  assert (solver-&gt;stable);
  assert (solver-&gt;mab);

  /* Keep simple per-heuristic scores locally (no struct changes). */
  enum { NUM_ARMS = 2 };
  static double q[NUM_ARMS] = { 0.0, 0.0 };
  static unsigned consecutive = 0;
  static uint64_t last_conflicts = 0;
  static uint64_t last_ticks = 0;

  /* Map current heuristic to an arm (0/1). */
  const unsigned cur_heuristic = solver-&gt;heuristic;
  const unsigned cur_arm = (cur_heuristic ? 1u : 0u);
  const unsigned other_arm = 1u - cur_arm;

  /* Derive a reward from progress since last call (conflicts per tick).
     Use deltas to avoid relying on non-existent averages fields. */
  const uint64_t now_conflicts = CONFLICTS;
  const uint64_t now_ticks = solver-&gt;ticks;

  uint64_t dconf = now_conflicts - last_conflicts;
  uint64_t dtick = now_ticks - last_ticks;

  last_conflicts = now_conflicts;
  last_ticks = now_ticks;

  if (!dtick) dtick = 1;

  const double rate = (double) dconf / (double) dtick;
  const double reward = log2 ((double) dconf + 1.0) / ((double) dtick);

  /* Trend based on reward delta. */
  static double prev_reward = 0.0;
  const double trend = reward - prev_reward;
  prev_reward = reward;

  /* Exponential moving update of Q for current arm. */
  const double alpha = 0.1;
  const double q_old = q[cur_arm];
  const double q_new = (1.0 - alpha) * q_old + alpha * (reward + 2.0 * trend);
  q[cur_arm] = q_new;

  /* Selection score with mild penalty for sticking too long. */
  const double decay = pow (0.95, (double) consecutive);
  const double s_cur = q_new * decay;
  const double s_other = q[other_arm];

  /* Switch heuristic if the other arm looks better. */
  if (s_other &gt; s_cur) {
    solver-&gt;heuristic = other_arm; /* assumes two heuristics 0/1 */
    consecutive = 0;
  } else {
    consecutive++;
  }

  /* Silence unused-variable warnings if compiled without assertions/logging. */
  (void) rate;
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
