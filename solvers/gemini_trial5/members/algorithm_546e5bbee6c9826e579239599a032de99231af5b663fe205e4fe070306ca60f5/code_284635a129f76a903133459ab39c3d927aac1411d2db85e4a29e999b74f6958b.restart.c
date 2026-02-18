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
  // Step 1: Compute metrics
  // L: Average LBD of the phase
  double L = AVERAGE (slow_glue);
  if (L < 1.0)
    L = 1.0;

  // D: Average decision level
  double D = AVERAGE (level);

  // N: Number of variables
  double N = (double) solver->vars;
  if (N < 1.0)
    N = 1.0;

  // d_ratio: Normalized depth (clamp to [0,1] for stability)
  double d_ratio = D / N;
  if (d_ratio < 0.0)
    d_ratio = 0.0;
  if (d_ratio > 1.0)
    d_ratio = 1.0;

  // Step 2: Base reward
  double R = 1.0 / L;

  // Step 3: Dual Asymmetric Incentives
  if (solver->stable) {
    // STABLE: Aggressively reward deep exploration
    R *= (1.0 + (2.0 * d_ratio));
  } else {
    // FOCUSED: Reward keeping search shallow and local
    R *= (1.0 + (0.5 * (1.0 - d_ratio)));
  }

  // Step 4: Apply Fatigue Penalty
  static int consecutive_modes = 0;
  static bool last_mode_init = false;
  static bool last_mode = false;

  if (!last_mode_init) {
    last_mode = solver->stable;
    consecutive_modes = 1;
    last_mode_init = true;
  } else if (solver->stable == last_mode) {
    consecutive_modes++;
  } else {
    consecutive_modes = 1;
    last_mode = solver->stable;
  }

  if (consecutive_modes > 4)
    R *= 0.6;

  // Step 5: Update MAB statistics
  // Map modes to arms: 0 = Focused, 1 = Stable
  const unsigned arm = solver->stable ? 1u : 0u;

  // Clamp reward to [0,1] so that "successes" never exceed "trials"
  if (R < 0.0)
    R = 0.0;
  if (R > 1.0)
    R = 1.0;

  solver->mab_reward[arm] += R;
  solver->mab_select[arm]++;

  // Step 6: Select next mode using a lightweight Thompson-like sampling.
  //
  // The original code called 'sample_beta', which does not exist in this
  // compilation unit (and caused the link error).  We approximate Thompson
  // sampling by drawing from a normal approximation of Beta(alpha,beta):
  //   mean = alpha/(alpha+beta)
  //   var  = alpha*beta/((alpha+beta)^2*(alpha+beta+1))
  // and then clamping to [0,1].
  //
  // Randomness source: use libc 'rand()' (no extra project dependencies).
  // This keeps the overall logic (stochastic arm selection) without requiring
  // missing symbols.
  const double u01 = ((double) rand () + 1.0) / ((double) RAND_MAX + 2.0);
  const double u02 = ((double) rand () + 1.0) / ((double) RAND_MAX + 2.0);

  // Box-Muller transform for standard normal.
  const double z0 = sqrt (-2.0 * log (u01)) * cos (2.0 * M_PI * u02);

  // Arm 0 (Focused)
  const double alpha0 = solver->mab_reward[0] + 1.0;
  const double beta0 =
      (double) solver->mab_select[0] - solver->mab_reward[0] + 1.0;
  const double sum0 = alpha0 + beta0;
  double mean0 = alpha0 / sum0;
  double var0 = (alpha0 * beta0) / (sum0 * sum0 * (sum0 + 1.0));
  if (var0 < 0.0)
    var0 = 0.0;
  double sample0 = mean0 + sqrt (var0) * z0;
  if (sample0 < 0.0)
    sample0 = 0.0;
  if (sample0 > 1.0)
    sample0 = 1.0;

  // Arm 1 (Stable) - use an independent normal draw
  const double u11 = ((double) rand () + 1.0) / ((double) RAND_MAX + 2.0);
  const double u12 = ((double) rand () + 1.0) / ((double) RAND_MAX + 2.0);
  const double z1 = sqrt (-2.0 * log (u11)) * cos (2.0 * M_PI * u12);

  const double alpha1 = solver->mab_reward[1] + 1.0;
  const double beta1 =
      (double) solver->mab_select[1] - solver->mab_reward[1] + 1.0;
  const double sum1 = alpha1 + beta1;
  double mean1 = alpha1 / sum1;
  double var1 = (alpha1 * beta1) / (sum1 * sum1 * (sum1 + 1.0));
  if (var1 < 0.0)
    var1 = 0.0;
  double sample1 = mean1 + sqrt (var1) * z1;
  if (sample1 < 0.0)
    sample1 = 0.0;
  if (sample1 > 1.0)
    sample1 = 1.0;

  // Select the arm with higher sampled value
  solver->stable = (sample1 > sample0);
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
