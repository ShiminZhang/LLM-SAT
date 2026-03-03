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
  /* This function is called from 'kissat_restart' only if
     (solver->stable && solver->mab).  It must not conflict with the existing
     'bool kissat_restarting(kissat*)' function and must only use fields that
     exist in the solver. */

  assert (solver);
  assert (solver->stable);
  assert (solver->unassigned);

  /* Guard against division by zero / invalid log argument. */
  const uint64_t ticks = solver->ticks;
  if (!ticks)
    return;

  const double cps = (double) CONFLICTS / (double) ticks;
  if (!(cps > 0.0))
    return;

  /* Yield proxy: log(conflicts per tick). */
  const double raw_yield = log (cps);

  /* Keep state local to this compilation unit without touching 'kissat'. */
  static double fast_ema_stable = 0.0, slow_ema_stable = 0.0;
  static double fast_ema_focused = 0.0, slow_ema_focused = 0.0;
  static unsigned consecutive_intervals = 0;

  /* Use fixed EMA rates (do not rely on RNG helpers not included here). */
  const double alpha_fast = 0.20;
  const double alpha_slow = 0.05;

  if (solver->stable) {
    fast_ema_stable =
        (1.0 - alpha_fast) * fast_ema_stable + alpha_fast * raw_yield;
    slow_ema_stable =
        (1.0 - alpha_slow) * slow_ema_stable + alpha_slow * raw_yield;
  } else {
    fast_ema_focused =
        (1.0 - alpha_fast) * fast_ema_focused + alpha_fast * raw_yield;
    slow_ema_focused =
        (1.0 - alpha_slow) * slow_ema_focused + alpha_slow * raw_yield;
  }

  const double momentum_stable = fast_ema_stable - slow_ema_stable;
  const double momentum_focused = fast_ema_focused - slow_ema_focused;

  const double projected_score_stable = fast_ema_stable + 1.5 * momentum_stable;
  const double projected_score_focused =
      fast_ema_focused + 1.5 * momentum_focused;

  /* Switch mode if momentum stays bad for a few consecutive intervals. */
  const double MOMENTUM_THRESHOLD = 0.0;
  const unsigned SWITCH_THRESHOLD = 3;

  if (solver->stable) {
    if (momentum_stable < MOMENTUM_THRESHOLD) {
      if (++consecutive_intervals > SWITCH_THRESHOLD) {
        solver->stable = false;
        consecutive_intervals = 0;
      }
    } else
      consecutive_intervals = 0;
  } else {
    if (momentum_focused < MOMENTUM_THRESHOLD) {
      if (++consecutive_intervals > SWITCH_THRESHOLD) {
        solver->stable = true;
        consecutive_intervals = 0;
      }
    } else
      consecutive_intervals = 0;
  }

  /* Periodic exploration: flip mode every MAX_INTERVALS ticks. */
  const uint64_t MAX_INTERVALS = 10;
  if (MAX_INTERVALS && (ticks % MAX_INTERVALS) == 0)
    solver->stable = !solver->stable;

  /* Pick heuristic based on projected score. */
  solver->heuristic = (projected_score_stable > projected_score_focused) ? 1 : 0;
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
