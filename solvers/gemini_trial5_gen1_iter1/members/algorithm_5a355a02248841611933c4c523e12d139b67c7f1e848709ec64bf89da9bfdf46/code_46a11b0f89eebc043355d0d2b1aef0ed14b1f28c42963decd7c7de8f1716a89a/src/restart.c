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
  /* Simple MAB-style heuristic switcher based on recent trends.
     Kept self-contained and only uses symbols available in this file/includes. */

  enum { N = 10 };
  const double THRESHOLD = 0.5;

  static double lbd_history[N];
  static double conflict_rate_history[N];
  static unsigned pos;
  static unsigned filled;
  static double S;

  /* Update circular buffers. */
  const double lbd = AVERAGE (fast_glue);

  /* 'ticks' might be zero early; avoid division by zero. */
  const double ticks = (double) solver->ticks;
  const double conflict_rate =
      ticks > 0.0 ? (double) solver->statistics.conflicts / ticks : 0.0;

  lbd_history[pos] = lbd;
  conflict_rate_history[pos] = conflict_rate;

  pos = (pos + 1u) % N;
  if (filled < N)
    filled++;

  /* Need enough samples for meaningful decisions. */
  if (filled < 2)
    return;

  /* Compute averages over available samples. */
  double avg_lbd = 0.0;
  for (unsigned i = 0; i < filled; i++)
    avg_lbd += lbd_history[i];
  avg_lbd /= (double) filled;

  /* Linear regression slope of conflict_rate over time index. */
  double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0;
  for (unsigned i = 0; i < filled; i++) {
    const double x = (double) i;
    const double y = conflict_rate_history[i];
    sum_x += x;
    sum_y += y;
    sum_xy += x * y;
    sum_x2 += x * x;
  }

  const double denom = (double) filled * sum_x2 - sum_x * sum_x;
  const double slope = denom != 0.0 ? (((double) filled * sum_xy - sum_x * sum_y) / denom) : 0.0;

  /* Performance metric (avoid sqrt(0) issues; level is unsigned anyway). */
  const double level = (double) solver->level;
  const double P = 10000.0 / (avg_lbd * sqrt (level) + 1.0);

  const double alpha = (slope > THRESHOLD) ? 0.9 : 0.5;
  const double S_new = alpha * S + (1.0 - alpha) * P;
  const double Delta = S_new - S;

  /* Mode selection: probabilistically switch heuristic on degradation. */
  if (Delta < 0.0) {
    double prob = fabs (Delta) * 5.0;
    if (prob > 1.0)
      prob = 1.0;

    if (kissat_pick_double (&solver->random) < prob) {
      /* Toggle heuristic (minimal "switch mode" logic). */
      solver->heuristic ^= 1u;
    }
  } else if (fabs (Delta) < 1e-5) {
    /* Placeholder: no-op for stable/focused comparison. */
  }

  S = S_new;
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
