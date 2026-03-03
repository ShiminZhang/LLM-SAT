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
  /* Local defaults for parameters that were previously undeclared macros. */
  const double alpha_fast = 0.20;
  const double alpha_slow = 0.05;
  const double alpha_vol  = 0.10;

  const double eps         = 1e-6;
  const double m_threshold = 0.05;
  const double perf_margin = 1.05;   /* require other to be ~5% better */
  const double prob_scale  = 10.0;   /* scale |M| into a probability */
  const double t_base      = 0.10;

  /* Step 1: interval performance metric P. */
  const double conflicts = (double) solver->mab_conflicts;
  const double decisions = (double) solver->mab_decisions;
  const double avg_lbd = AVERAGE (fast_glue);
  const double P =
      (decisions > 0.0) ? (conflicts / decisions) * (1.0 / (avg_lbd + 1.0)) : 0.0;

  /* Persist state across calls (kissat struct is fixed). */
  static double e_fast[2] = {0.0, 0.0};
  static double e_slow[2] = {0.0, 0.0};
  static double volatility[2] = {0.0, 0.0};
  static bool initialized[2] = {false, false};

  const unsigned m = solver->heuristic;
  if (m >= 2)
    return;

  if (!initialized[m]) {
    e_fast[m] = P;
    e_slow[m] = P;
    volatility[m] = 0.0;
    initialized[m] = true;
  } else {
    e_fast[m] = (1.0 - alpha_fast) * e_fast[m] + alpha_fast * P;
    e_slow[m] = (1.0 - alpha_slow) * e_slow[m] + alpha_slow * P;
    volatility[m] = (1.0 - alpha_vol) * volatility[m] +
                    alpha_vol * fabs (P - e_fast[m]);
  }

  /* For reporting/external visibility. */
  solver->mab_reward[m] = e_fast[m];

  /* Step 3: Relative Momentum M. */
  const double M = (e_fast[m] - e_slow[m]) / (e_slow[m] + eps);

  unsigned next_m = m;
  const unsigned other = 1u - m;

  /* Step 4: Trajectory gating. */
  if (M > m_threshold) {
    next_m = m;
  } else if (M < -m_threshold) {
    if (initialized[other] && e_fast[m] < (e_fast[other] * perf_margin)) {
      double prob = fabs (M) * prob_scale;
      if (prob > 1.0)
        prob = 1.0;
      if (kissat_pick_double (&solver->random) < prob)
        next_m = other;
    }
  } else {
    /* Step 5: Stagnant: volatility-scaled Boltzmann selection. */
    const double denom = e_fast[m] + eps;
    const double T = t_base + (volatility[m] / denom);

    /* Guard against pathological tiny/negative temperatures. */
    const double safeT = (T > eps) ? T : eps;

    const double v0 = e_fast[0] / safeT;
    const double v1 = e_fast[1] / safeT;
    const double max_v = (v0 > v1) ? v0 : v1;

    const double exp0 = exp (v0 - max_v);
    const double exp1 = exp (v1 - max_v);
    const double sum_exp = exp0 + exp1;

    if (sum_exp > 0.0) {
      const double prob0 = exp0 / sum_exp;
      const double r = kissat_pick_double (&solver->random);
      next_m = (r < prob0) ? 0u : 1u;
    }
  }

  solver->heuristic = next_m;
  solver->mab_select[next_m]++;

  solver->mab_decisions = 0;
  solver->mab_conflicts = 0;
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
