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
  // Persistent state for Thompson Sampling and Reward tracking
  static double alpha[2] = {1.0, 1.0};
  static double beta[2] = {1.0, 1.0};
  static double reward_ema = 1.0;
  static bool ema_init = false;
  static uint64_t last_scale_conflicts = 0;
  static uint64_t history[8] = {0};
  static int history_idx = 0;
  static int history_count = 0;

  // Step 1: Calculate Trail Signature
  // XOR-sum of decision levels of the first 32 variables assigned
  uint64_t signature = 0;
  const unsigned trail_size = SIZE_STACK (solver->trail);
  const unsigned limit = (trail_size < 32) ? trail_size : 32;
  unsigned current_level = 0;
  for (unsigned i = 0; i < limit; i++) {
    while (current_level < solver->level &&
           i >= FRAME (current_level + 1).trail) {
      current_level++;
    }
    signature ^= (uint64_t) current_level;
  }

  // Step 2: Calculate Overlap (Jaccard Similarity)
  bool in_history = false;
  for (int i = 0; i < history_count; i++) {
    if (history[i] == signature) {
      in_history = true;
      break;
    }
  }

  double overlap = 0.0;
  if (in_history) {
    uint64_t combined[9];
    combined[0] = signature;
    for (int i = 0; i < history_count; i++)
      combined[i + 1] = history[i];

    int unique_total = 0;
    for (int i = 0; i < history_count + 1; i++) {
      bool seen = false;
      for (int j = 0; j < i; j++) {
        if (combined[i] == combined[j]) {
          seen = true;
          break;
        }
      }
      if (!seen)
        unique_total++;
    }
    overlap = 1.0 / (double) unique_total;
  }

  // Reward R = (LBD_global / LBD_current) * (1 - Overlap)
  const double global_lbd = AVERAGE (slow_glue);
  const double current_lbd = AVERAGE (fast_glue);
  const double ratio = (current_lbd > 0) ? (global_lbd / current_lbd) : 1.0;
  const double reward = ratio * (1.0 - overlap);

  // Step 3: Thompson Sampling update
  if (reward > 1.0)
    alpha[solver->heuristic] += 1.0;
  else
    beta[solver->heuristic] += 1.0;

  // EMA update for volatility detection
  if (!ema_init) {
    reward_ema = reward;
    ema_init = true;
  } else {
    reward_ema = 0.9 * reward_ema + 0.1 * reward;
  }

  // Step 4: Volatility-Triggered Scaling
  if (reward < 0.5 * reward_ema || (CONFLICTS - last_scale_conflicts) >= 2048) {
    for (int i = 0; i < 2; i++) {
      alpha[i] *= 0.5;
      beta[i] *= 0.5;
      if (alpha[i] < 1.0)
        alpha[i] = 1.0;
      if (beta[i] < 1.0)
        beta[i] = 1.0;
    }
    last_scale_conflicts = CONFLICTS;
  }

  // Update history buffer with current signature
  history[history_idx] = signature;
  history_idx = (history_idx + 1) % 8;
  if (history_count < 8)
    history_count++;

  // Thompson Sampling: sample from Beta(alpha,beta) via Gamma draws.
  // C does not support C++ lambdas/auto, so use local static helpers.

  // Standard normal via Box-Muller.
  static double sample_normal (kissat *s) {
    const double u1 = kissat_pick_double (&s->random);
    const double u2 = kissat_pick_double (&s->random);
    const double r = sqrt (-2.0 * log (u1 + 1e-9));
    const double theta = 2.0 * 3.14159265358979323846 * u2;
    return r * cos (theta);
  }

  // Marsaglia-Tsang gamma sampler for shape a >= 1.
  static double sample_gamma_ge1 (kissat *s, double a) {
    const double d = a - 1.0 / 3.0;
    const double c = 1.0 / sqrt (9.0 * d);
    for (;;) {
      const double x = sample_normal (s);
      double v = 1.0 + c * x;
      if (v <= 0.0)
        continue;
      v = v * v * v;
      const double u = kissat_pick_double (&s->random);
      if (u < 1.0 - 0.0331 * x * x * x * x)
        return d * v;
      if (log (u) < 0.5 * x * x + d * (1.0 - v + log (v)))
        return d * v;
    }
  }

  static double sample_gamma (kissat *s, double a) {
    if (a < 1.0) {
      // Boosting method: Gamma(a) = Gamma(a+1) * U^(1/a)
      const double u = kissat_pick_double (&s->random);
      return pow (u, 1.0 / a) * sample_gamma_ge1 (s, a + 1.0);
    }
    return sample_gamma_ge1 (s, a);
  }

  double samples[2];
  for (int i = 0; i < 2; i++) {
    const double ga = sample_gamma (solver, alpha[i]);
    const double gb = sample_gamma (solver, beta[i]);
    const double denom = ga + gb;
    samples[i] = denom > 0.0 ? (ga / denom) : 0.5;
  }

  // Select Arm (Luby=0 vs Glucose=1)
  solver->heuristic = (samples[1] > samples[0]) ? 1 : 0;

  // Reset Kissat MAB tracking fields
  solver->mab_reward[0] = alpha[0];
  solver->mab_reward[1] = alpha[1];
  solver->mab_select[0] = (unsigned) (alpha[0] + beta[0]);
  solver->mab_select[1] = (unsigned) (alpha[1] + beta[1]);

  for (all_variables (idx))
    solver->mab_chosen[idx] = 0;

  solver->mab_chosen_tot = 0;
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
