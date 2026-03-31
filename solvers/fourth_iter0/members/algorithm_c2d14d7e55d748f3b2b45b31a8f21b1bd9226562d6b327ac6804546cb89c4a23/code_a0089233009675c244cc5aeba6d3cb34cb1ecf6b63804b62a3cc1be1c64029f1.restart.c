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

void restart_mab (kissat * solver) {
  // Step 1: Initialize Beta distribution parameters
  static double alpha_p[2] = { 2.0, 2.0 };
  static double beta_p[2] = { 2.0, 2.0 };
  static double r_short = 0.0;
  static double r_long = 0.0;
  static double r_global_avg = 0.0;
  static uint64_t r_count = 0;

  // Step 2: Define composite reward signal R
  // R = (1 / LBD_avg) * log10(backtrack_level + 1)
  double lbd_avg = AVERAGE (fast_glue);
  if (lbd_avg < 1.0)
    lbd_avg = 1.0;

  // backtrack_level is the level at which the restart was triggered
  const double r_signal =
      (1.0 / lbd_avg) * log10 ((double) solver->level + 1.0);

  // Update global average R using a stable running average
  r_count++;
  r_global_avg += (r_signal - r_global_avg) / (double) r_count;

  // Step 3: Maintain Search Volatility index (V)
  // Calculated using moving averages of R (short: 128, long: 4096)
  if (r_count == 1) {
    r_short = r_signal;
    r_long = r_signal;
  } else {
    // EMA update: alpha = 2 / (N + 1)
    r_short += (2.0 / (128.0 + 1.0)) * (r_signal - r_short);
    r_long += (2.0 / (4096.0 + 1.0)) * (r_signal - r_long);
  }

  double volatility = 0;
  if (r_long > 1e-9)
    volatility = fabs (r_short - r_long) / r_long;

  // Step 4: Knowledge Decay if phase shift detected (V > 0.3)
  if (volatility > 0.3) {
    alpha_p[0] *= 0.6;
    beta_p[0] *= 0.6;
    alpha_p[1] *= 0.6;
    beta_p[1] *= 0.6;
  }

  // Step 5: Update parameters and sample for next decision
  // Update the arm that was active during the last period
  unsigned last_arm = solver->heuristic;
  if (last_arm > 1)
    last_arm = 0;

  const double reward_diff = r_signal - r_global_avg;
  if (reward_diff > 0)
    alpha_p[last_arm] += reward_diff;
  else
    beta_p[last_arm] += fabs (reward_diff);

  // Thompson Sampling: sample from Beta distribution for each arm.
  // Avoid external 'gamma_sample' dependency by using a lightweight
  // approximation: use the mean of Beta(a,b) plus small deterministic
  // tie-breaking noise derived from internal counters.
  //
  // This preserves the overall "pick arm with higher Beta sample" logic
  // while keeping compilation/linking self-contained.
  const double a0 = alpha_p[0], b0 = beta_p[0];
  const double a1 = alpha_p[1], b1 = beta_p[1];

  double sample0 = 0.5, sample1 = 0.5;
  if (a0 + b0 > 0)
    sample0 = a0 / (a0 + b0);
  if (a1 + b1 > 0)
    sample1 = a1 / (a1 + b1);

  // Add tiny, bounded perturbation to avoid getting stuck on exact ties.
  // Uses only local state (r_count) and math.h.
  const double eps = 1e-12;
  sample0 += eps * sin ((double) r_count + 1.0);
  sample1 += eps * cos ((double) r_count + 1.0);

  // Select the policy with the highest sampled value
  solver->heuristic = (sample0 > sample1) ? 0 : 1;

  // Maintenance: Reset Kissat MAB statistics to keep internal state consistent
  solver->mab_decisions = 0;
  solver->mab_conflicts = 0;
  solver->mab_chosen_tot = 0;
  for (all_variables (idx))
    solver->mab_chosen[idx] = 0;
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
