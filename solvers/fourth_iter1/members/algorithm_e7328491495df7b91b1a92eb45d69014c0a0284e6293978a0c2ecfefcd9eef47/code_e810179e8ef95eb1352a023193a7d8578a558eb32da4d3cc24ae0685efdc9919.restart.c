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
  // Step 1: Initialize arm parameters (Beta distribution alpha/beta)
  // and reward moving averages.
  static double alpha[2] = {2.0, 2.0};
  static double beta[2] = {2.0, 2.0};
  static double r_ema_128 = 0.0;
  static double r_ema_4096 = 0.0;
  static bool initialized = false;

  if (!initialized) {
    alpha[0] = 2.0; beta[0] = 2.0;
    alpha[1] = 2.0; beta[1] = 2.0;
    initialized = true;
  }

  // Step 2: Calculate reward signal R.
  // R = (1 / LBD_avg) * log10(backtrack_level + 1)
  // We use AVERAGE(fast_glue) as the LBD average over the recent window.
  const double lbd_avg = AVERAGE (fast_glue);
  const double backtrack_level = (double) solver->level;
  const double r = (lbd_avg > 1e-6) ? (1.0 / lbd_avg) * log10 (backtrack_level + 1.0) : 0.0;

  // Step 3: Maintain Search Volatility index (V) via moving averages of R.
  // Alpha for EMA is 2/(N+1). For 128: ~0.0155. For 4096: ~0.000488.
  r_ema_128 = (r_ema_128 == 0.0) ? r : r_ema_128 + (2.0 / 129.0) * (r - r_ema_128);
  r_ema_4096 = (r_ema_4096 == 0.0) ? r : r_ema_4096 + (2.0 / 4097.0) * (r - r_ema_4096);

  const double v = (r_ema_4096 > 1e-9) ? fabs (r_ema_128 - r_ema_4096) / r_ema_4096 : 0.0;

  // Step 5 (part 1): Update the chosen arm's parameters based on the reward R.
  // Update is proportional to relative performance compared to the long-term average.
  const unsigned chosen = solver->heuristic % 2;
  if (r_ema_4096 > 1e-9 && r > 1e-9) {
    if (r > r_ema_4096) {
      alpha[chosen] += (r / r_ema_4096);
    } else {
      beta[chosen] += (r_ema_4096 / r);
    }
  } else {
    // Fallback update if reward or average is near zero
    alpha[chosen] += 1.0;
  }

  // Step 4: Handle high volatility.
  if (v > 0.3) {
    // Apply proportional decay
    const double v_capped = (v > 0.8) ? 0.8 : v;
    const double decay = 1.0 - v_capped;
    alpha[0] *= decay; beta[0] *= decay;
    alpha[1] *= decay; beta[1] *= decay;

    // Force exploration step: select arm with lower current mean
    const double mean0 = alpha[0] / (alpha[0] + beta[0]);
    const double mean1 = alpha[1] / (alpha[1] + beta[1]);
    solver->heuristic = (mean0 < mean1) ? 0 : 1;
  } else {
    // Step 5 (part 2): Thompson Sampling for arm selection.
    // Sample from Beta distribution for each arm using Normal approximation.
    double samples[2];
    for (unsigned i = 0; i < 2; i++) {
      // Box-Muller transform for Normal distribution sampling
      const double u1 = 1.0 - kissat_pick_double (&solver->random);
      const double u2 = 1.0 - kissat_pick_double (&solver->random);
      const double n = sqrt (-2.0 * log (u1)) * cos (2.0 * 3.14159265358979323846 * u2);
      
      const double mean = alpha[i] / (alpha[i] + beta[i]);
      const double sum = alpha[i] + beta[i];
      const double variance = (alpha[i] * beta[i]) / (sum * sum * (sum + 1.0));
      samples[i] = mean + n * sqrt (variance);
    }
    // Select the policy with the highest sampled value
    solver->heuristic = (samples[0] > samples[1]) ? 0 : 1;
  }

  // Housekeeping: Reset MAB tracking fields for the next search period
  solver->mab_decisions = 0;
  solver->mab_conflicts = 0;
  for (all_variables (idx)) {
    solver->mab_chosen[idx] = 0;
  }
  solver->mab_chosen_tot = 0;
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
