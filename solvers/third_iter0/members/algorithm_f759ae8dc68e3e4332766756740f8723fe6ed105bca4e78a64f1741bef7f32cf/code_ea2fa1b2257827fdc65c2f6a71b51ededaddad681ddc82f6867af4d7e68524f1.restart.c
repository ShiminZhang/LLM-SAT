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
  /* Multi-armed bandit restart heuristic selector.
     This function is called from 'kissat_restart' in stable mode when
     'solver->mab' is enabled.  It updates 'solver->heuristic' to pick
     the next restart heuristic.

     NOTE: We must only use fields/macros that exist in this file/includes.
     In particular, there is no 'statistics.clauses_reduced' in this code base,
     so we use 'statistics.reduced' (number of reduced/deleted clauses) which
     is the common Kissat statistic name. */

#define BANDIT_WINDOW 1024u
#define BANDIT_DECAY_WINDOW 8192u
#define BANDIT_PI 3.14159265358979323846

  /* Persistent bandit state (process-wide, as in the injected code). */
  static double S[3] = { 1.0, 1.0, 1.0 };
  static double F[3] = { 1.0, 1.0, 1.0 };
  static uint64_t last_switch = 0;
  static uint64_t last_decay = 0;
  static unsigned arm = 0;

  static double sum_ge = 0;
  static uint64_t n_ge = 0;
  static double gma = 0;
  static double max_r = 1e-6;

  static double rewards[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
  static unsigned r_idx = 0;

  static uint64_t last_c = 0;
  static uint64_t last_red = 0;
  static unsigned peak_lvl = 0;

  /* Guard: only meaningful if we are in a search state. */
  if (!solver)
    return;

  /* Track peak decision level between conflicts and compute a simple
     "geometric efficiency" proxy on each new conflict. */
  if (solver->level > peak_lvl)
    peak_lvl = solver->level;

  if (CONFLICTS > last_c) {
    const double ge =
        (peak_lvl > solver->level)
            ? ((double) (peak_lvl - solver->level) / (double) (peak_lvl + 1u))
            : 0.0;
    sum_ge += ge;
    n_ge++;
    last_c = CONFLICTS;
    peak_lvl = solver->level;
  }

  /* Evaluate and potentially switch arm every BANDIT_WINDOW conflicts. */
  if (CONFLICTS >= last_switch + BANDIT_WINDOW) {
    const double avg_ge = n_ge ? (sum_ge / (double) n_ge) : 0.0;

    double lbd = AVERAGE (slow_glue);
    if (lbd < 1.0)
      lbd = 1.0;

    /* Use existing statistic 'reduced' instead of non-existent
       'clauses_reduced'. */
    const uint64_t reduced_now = solver->statistics.reduced;
    const uint64_t reduced_delta = reduced_now - last_red;
    const double red_rate = (double) reduced_delta / (double) BANDIT_WINDOW;

    /* Reward: (avg_ge * log2(1 + red_rate)) / avg_lbd */
    const double log2_1p = log (1.0 + red_rate) / log (2.0);
    const double R = (avg_ge * log2_1p) / lbd;

    if (R > max_r)
      max_r = R;

    if (R > gma)
      S[arm] += 1.0 + (R / max_r);
    else
      F[arm] += 1.0;

    gma = (gma == 0.0) ? R : (0.9 * gma + 0.1 * R);

    rewards[r_idx] = R;
    r_idx = (r_idx + 1u) & 7u;

    /* Volatility-based decay. */
    double avg_r_window = 0.0;
    for (unsigned i = 0; i < 8u; i++)
      avg_r_window += rewards[i];
    avg_r_window /= 8.0;

    double var_r = 0.0;
    for (unsigned i = 0; i < 8u; i++) {
      const double diff = rewards[i] - avg_r_window;
      var_r += diff * diff;
    }
    const double sigma_r = sqrt (var_r / 8.0);

    if (CONFLICTS >= last_decay + BANDIT_DECAY_WINDOW || R < 0.5 * gma) {
      double lambda = 0.95 - sigma_r;
      if (lambda < 0.5)
        lambda = 0.5;
      for (unsigned i = 0; i < 3u; i++) {
        S[i] *= lambda;
        F[i] *= lambda;
        if (S[i] < 1.0)
          S[i] = 1.0;
        if (F[i] < 1.0)
          F[i] = 1.0;
      }
      last_decay = CONFLICTS;
    }

    /* Thompson sampling via normal approximation to Beta(S,F). */
    double max_theta = -1.0;
    unsigned best_arm = arm;
    for (unsigned i = 0; i < 3u; i++) {
      const double denom = S[i] + F[i];
      const double mu = S[i] / denom;
      const double var =
          (S[i] * F[i]) / ((denom * denom) * (denom + 1.0));

      const double u1 = kissat_pick_double (&solver->random);
      const double u2 = kissat_pick_double (&solver->random);

      const double z =
          sqrt (-2.0 * log (u1 + 1e-15)) * cos (2.0 * BANDIT_PI * u2);

      const double theta = mu + z * sqrt (var);

      if (theta > max_theta) {
        max_theta = theta;
        best_arm = i;
      }
    }
    arm = best_arm;

    last_switch = CONFLICTS;
    last_red = reduced_now;
    sum_ge = 0.0;
    n_ge = 0;
  }

  /* Map arm -> heuristic choice.
     We only *select* the heuristic here; the actual restart decision is
     still made by the normal restart code paths. */
  solver->heuristic = arm;
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
