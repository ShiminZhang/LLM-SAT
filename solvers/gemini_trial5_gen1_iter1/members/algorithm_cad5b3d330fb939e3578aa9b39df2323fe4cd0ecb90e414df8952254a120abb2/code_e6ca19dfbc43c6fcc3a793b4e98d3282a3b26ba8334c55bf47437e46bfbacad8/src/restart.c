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

static void restart_mab (kissat *solver) {
  /* Minimal, compile-safe implementation.
     The injected code tried to:
       - redefine 'kissat_restarting' with a different return type
       - use '...' placeholders (invalid C)
       - allocate VLA 'probabilities[solver->vars]' (not necessarily desired)
     Here we keep the "overall logic" idea (EMA-based priority + stagnation
     penalty) but only use state available in this file and avoid placeholders.
  */

  assert (solver);
  assert (solver->unassigned);

  /* Only meaningful in stable mode with MAB enabled (as used by caller). */
  if (!solver->stable || !solver->mab)
    return;

  /* Keep small history and EMAs across calls. */
  enum { N = 10 };
  const double ALPHA_FAST = 0.30;
  const double ALPHA_SLOW = 0.05;
  const double PENALTY_FACTOR = 0.50;

  static double glue_history[N];
  static unsigned hist_pos;
  static bool initialized;

  static double fast_ema;
  static double slow_ema;
  static uint64_t last_learned;
  static unsigned stagnation_counter;

  if (!initialized) {
    for (unsigned i = 0; i < N; i++)
      glue_history[i] = 0.0;
    hist_pos = 0;
    fast_ema = slow_ema = 0.0;
    last_learned = solver->statistics.clauses_learned;
    stagnation_counter = 0;
    initialized = true;
  }

  /* Use existing glue averages as a proxy for "quality" (LBD-like). */
  const double fast_glue = AVERAGE (fast_glue);
  const double slow_glue = AVERAGE (slow_glue);

  /* Update circular buffer with current observed glue proxy. */
  glue_history[hist_pos] = fast_glue;
  hist_pos = (hist_pos + 1u) % N;

  double sum = 0.0;
  for (unsigned i = 0; i < N; i++)
    sum += glue_history[i];

  const double avg = sum / (double) N;

  /* Reward: higher when glue is low.  Clamp denominator to avoid div-by-zero. */
  const double denom = fmax (1e-9, avg);
  const double R = 1.0 / denom;

  fast_ema = ALPHA_FAST * R + (1.0 - ALPHA_FAST) * fast_ema;
  slow_ema = ALPHA_SLOW * R + (1.0 - ALPHA_SLOW) * slow_ema;

  /* Stagnation: no new learned clauses since last call. */
  const uint64_t learned = solver->statistics.clauses_learned;
  if (learned == last_learned)
    stagnation_counter++;
  else
    stagnation_counter = 0;
  last_learned = learned;

  /* Priority: emphasize recent improvements. */
  double P = slow_ema + 1.5 * (fast_ema - slow_ema);

  if (stagnation_counter > 5)
    P *= PENALTY_FACTOR;

  /* "Bandit" action: pick between two heuristics (0/1) based on P.
     We do not know the full heuristic space here, but solver->heuristic is
     used elsewhere, so keep it binary and stable.
  */
  const unsigned old = solver->heuristic;
  const unsigned alt = old ^ 1u;

  /* If fast glue is worse than slow glue, exploration is encouraged. */
  const bool worsening = (fast_glue > slow_glue);

  /* Simple decision rule derived from P (bounded, deterministic). */
  const double threshold = worsening ? 0.10 : 0.25;

  if (P < threshold)
    solver->heuristic = alt;
  else
    solver->heuristic = old;

  kissat_extremely_verbose (
      solver,
      "restart_mab: fast_glue=%g slow_glue=%g avg=%g R=%g fast_ema=%g slow_ema=%g "
      "P=%g stagnation=%u heuristic %u->%u",
      fast_glue, slow_glue, avg, R, fast_ema, slow_ema, P, stagnation_counter,
      old, solver->heuristic);
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
