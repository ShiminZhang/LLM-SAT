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
  /* This function is called from 'kissat_restart' when 'solver->stable && solver->mab'.
     It implements the intended EMA/momentum logic without introducing new symbols
     or redefining existing functions. */

  assert(solver);

  /* Use existing averages as proxies (as in the injected code). */
  const double average_LBD = AVERAGE(fast_glue);
  const double average_trail_level = AVERAGE(slow_glue);

  /* Guard against invalid values (sqrt domain / division). */
  const double denom =
      average_LBD * sqrt(average_trail_level > 0.0 ? average_trail_level : 0.0) + 1.0;
  const double P = 10000.0 / denom;

  /* Static state must be initialized with constants. */
  static bool initialized = false;
  static double fast_EMA = 0.0;
  static double slow_EMA = 0.0;
  static double alpha_fast = 0.1;
  static double alpha_slow = 0.05;
  static double previous_P = 0.0;
  static unsigned intervals = 0;

  if (!initialized) {
    fast_EMA = P;
    slow_EMA = P;
    previous_P = P;
    intervals = 0;
    initialized = true;
  }

  /* Update alphas based on recent performance. */
  if (P > previous_P) {
    alpha_fast += 0.01;
    alpha_slow += 0.005;
  } else {
    alpha_fast -= 0.01;
    alpha_slow -= 0.005;
  }

  /* Clamp alphas to [0,1]. */
  if (alpha_fast < 0.0) alpha_fast = 0.0;
  if (alpha_slow < 0.0) alpha_slow = 0.0;
  if (alpha_fast > 1.0) alpha_fast = 1.0;
  if (alpha_slow > 1.0) alpha_slow = 1.0;

  fast_EMA = (1.0 - alpha_fast) * fast_EMA + alpha_fast * P;
  slow_EMA = (1.0 - alpha_slow) * slow_EMA + alpha_slow * P;

  const double momentum = fast_EMA - slow_EMA;
  const double projected_score = fast_EMA + (1.5 * momentum);

  /* Mode switching logic (kept from injected code). */
  if (momentum < 0.0 && intervals > 3)
    solver->stable = !solver->stable;

  if (intervals % 10u == 0u)
    solver->stable = !solver->stable;

  intervals++;
  previous_P = P;

  kissat_extremely_verbose(solver, "restart MAB projected score: %g", projected_score);
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
