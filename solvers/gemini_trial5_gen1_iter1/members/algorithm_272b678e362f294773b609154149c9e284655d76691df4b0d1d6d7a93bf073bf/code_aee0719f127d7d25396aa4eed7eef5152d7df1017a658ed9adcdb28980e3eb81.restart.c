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
  /* Simple, self-contained MAB-style heuristic switch used by kissat_restart().
     Keeps per-solver state in static variables (as the injected code did). */

  assert (solver);

  enum { N = 10 };

  /* Circular buffers */
  static double lbd_buffer[N];
  static double conflict_rate_buffer[N];
  static unsigned buffer_index;
  static bool initialized;

  if (!initialized) {
    for (unsigned i = 0; i < N; i++) {
      lbd_buffer[i] = 0.0;
      conflict_rate_buffer[i] = 0.0;
    }
    buffer_index = 0;
    initialized = true;
  }

  /* Record current observations. */
  const double lbd = AVERAGE (fast_glue);

  /* 'ticks' might be zero early; avoid division by zero. */
  const double ticks = solver->ticks ? (double) solver->ticks : 1.0;
  const double conflict_rate = (double) CONFLICTS / ticks;

  lbd_buffer[buffer_index] = lbd;
  conflict_rate_buffer[buffer_index] = conflict_rate;

  /* Advance index after writing; keep last written index for reading. */
  const unsigned last = buffer_index;
  buffer_index = (buffer_index + 1u) % N;

  /* Compute fast/slow EMAs over the buffers (same overall logic). */
  double fast_lbd = 0.0, slow_lbd = 0.0;
  double fast_cr = 0.0, slow_cr = 0.0;

  const double alpha_fast = 0.1;
  const double alpha_slow = 0.01;

  for (unsigned i = 0; i < N; i++) {
    fast_lbd = (1.0 - alpha_fast) * fast_lbd + alpha_fast * lbd_buffer[i];
    slow_lbd = (1.0 - alpha_slow) * slow_lbd + alpha_slow * lbd_buffer[i];

    fast_cr = (1.0 - alpha_fast) * fast_cr + alpha_fast * conflict_rate_buffer[i];
    slow_cr = (1.0 - alpha_slow) * slow_cr + alpha_slow * conflict_rate_buffer[i];
  }

  /* Raw yield based on conflict rate; avoid log(0). */
  const double y = log (conflict_rate_buffer[last] + 1e-10);

  /* Update LBD EMAs with yield (as in injected code). */
  fast_lbd = (1.0 - alpha_fast) * fast_lbd + alpha_fast * y;
  slow_lbd = (1.0 - alpha_slow) * slow_lbd + alpha_slow * y;

  /* Momentum and projected scores. */
  const double momentum_stable = fast_lbd - slow_lbd;
  const double projected_stable = fast_lbd + 1.5 * momentum_stable;

  const double momentum_focused = fast_cr - slow_cr;
  const double projected_focused = fast_cr + 1.5 * momentum_focused;

  /* Choose heuristic based on projected score.
     We map "stable" to heuristic 0 and "focused" to heuristic 1. */
  unsigned chosen = (projected_stable > projected_focused) ? 0u : 1u;

  /* Decay / flip after a few restarts if momentum indicates worsening. */
  if (momentum_stable < 0.0 && solver->statistics.restarts > 3)
    chosen ^= 1u;

  /* Adaptive Boltzmann-like random flip based on variability. */
  double v = 0.0;
  for (unsigned i = 0; i < N; i++) {
    const double d = conflict_rate_buffer[i] - fast_cr;
    v += d * d;
  }
  v = sqrt (v / (double) N);

  const double t = 0.1 + v;
  if (kissat_pick_double (&solver->random) < t)
    chosen ^= 1u;

  solver->heuristic = chosen;
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
