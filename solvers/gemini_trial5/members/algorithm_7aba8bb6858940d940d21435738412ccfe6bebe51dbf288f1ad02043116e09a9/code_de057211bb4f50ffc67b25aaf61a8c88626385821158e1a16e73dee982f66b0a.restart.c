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
  // Static state for fatigue tracking
  static unsigned last_heuristic = 0;
  static int consecutive_modes = 0;

  // Step 1: Compute metrics
  // L: Phase average LBD (using fast glue EMA)
  double L = AVERAGE (fast_glue);
  if (L < 1.0)
    L = 1.0;

  // A: Average assigned variables (trail EMA)
  double A = AVERAGE (trail);

  // N_active: Currently non-fixed variables
  double N_active = (double) solver->vars;
  if (N_active < 1.0)
    N_active = 1.0;

  // Search space penetration ratio
  double d_ratio = A / N_active;

  // Step 2: Base reward signal
  double R = 1.0 / L;

  // Step 3: Apply Asymmetric Incentives
  if (solver->stable)
    R *= (1.0 + (1.5 * d_ratio));

  // Step 4: Apply Fatigue Penalty
  if (solver->heuristic == last_heuristic)
    consecutive_modes++;
  else {
    consecutive_modes = 1;
    last_heuristic = solver->heuristic;
  }

  if (consecutive_modes > 4)
    R *= 0.6;

  // Normalize R into [0,1]
  double R_scaled = R / 3.0;
  if (R_scaled > 1.0)
    R_scaled = 1.0;
  if (R_scaled < 0.0)
    R_scaled = 0.0;

  // Step 5: Update Multi-Armed Bandit statistics
  unsigned current_arm = solver->heuristic;
  solver->mab_reward[current_arm] += R_scaled;
  solver->mab_select[current_arm] += 1;

  // Step 6: Select next mode using Thompson Sampling
  //
  // The original code called 'pick_beta_dist', which does not exist in this
  // compilation unit (and is not provided by the includes), causing an
  // undefined reference.  We implement Thompson sampling by drawing from a
  // Beta(alpha,beta) distribution using the relationship:
  //   If X~Gamma(alpha,1), Y~Gamma(beta,1) independent, then X/(X+Y)~Beta(alpha,beta).
  //
  // We approximate Gamma(k,1) for k>=1 by summing exponentials:
  //   Gamma(k,1) = sum_{j=1..m} Exp(1) + Gamma(frac,1)
  // and approximate the fractional remainder by one more Exp(1) scaled.
  // This keeps the implementation self-contained and avoids extra dependencies.
  unsigned best_arm = current_arm;
  double max_sample = -1.0;

  for (unsigned i = 0; i < solver->mab_heuristics; i++) {
    double s = solver->mab_reward[i];
    double n = (double) solver->mab_select[i];

    if (s > n)
      s = n;

    const double alpha = s + 1.0;
    const double beta = (n - s) + 1.0;

    // Draw Gamma(shape,1) with a simple exponential-sum approximation.
    // Uses solver RNG to stay deterministic w.r.t. solver seed.
    double x = 0.0, y = 0.0;

    // Helper macro: add one Exp(1) sample to accumulator.
#define ADD_EXP1(acc)                                                          \
  do {                                                                         \
    /* kissat_random(solver) is available via internal.h */                     \
    uint64_t r__ = kissat_random (solver);                                      \
    /* map to (0,1] to avoid log(0) */                                          \
    double u__ = ((double) (r__ + 1)) / ((double) UINT64_MAX + 1.0);            \
    (acc) += -log (u__);                                                       \
  } while (0)

    // Integer parts
    int ia = (int) floor (alpha);
    int ib = (int) floor (beta);
    for (int k = 0; k < ia; k++)
      ADD_EXP1 (x);
    for (int k = 0; k < ib; k++)
      ADD_EXP1 (y);

    // Fractional remainders (in [0,1))
    double fa = alpha - (double) ia;
    double fb = beta - (double) ib;
    if (fa > 0.0) {
      double tmp = 0.0;
      ADD_EXP1 (tmp);
      x += fa * tmp;
    }
    if (fb > 0.0) {
      double tmp = 0.0;
      ADD_EXP1 (tmp);
      y += fb * tmp;
    }

#undef ADD_EXP1

    double sample;
    const double sum = x + y;
    if (sum > 0.0)
      sample = x / sum;
    else
      sample = 0.5;

    if (sample > max_sample) {
      max_sample = sample;
      best_arm = i;
    }
  }

  solver->heuristic = best_arm;
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
