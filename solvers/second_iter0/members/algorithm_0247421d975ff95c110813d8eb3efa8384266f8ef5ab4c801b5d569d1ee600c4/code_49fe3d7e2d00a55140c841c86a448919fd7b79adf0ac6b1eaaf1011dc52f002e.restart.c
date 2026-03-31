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

static double normal_sample (kissat * solver) {
  double u1 = kissat_pick_double (&solver->random);
  double u2 = kissat_pick_double (&solver->random);
  if (u1 < 1e-9) u1 = 1e-9;
  return sqrt (-2.0 * log (u1)) * cos (6.283185307 * u2);
}

static double calculate_activity_entropy (kissat * solver) {
  double sum = 0;
  const heap *const scores = kissat_get_scores (solver);
  for (unsigned i = 0; i < solver->vars; i++) {
    double act = solver->stable ? (scores ? kissat_get_heap_score (scores, i) : 0) : (double) solver->links[i].stamp;
    if (act > 0) sum += act;
  }
  if (sum <= 0) return 0;
  double h = 0;
  for (unsigned i = 0; i < solver->vars; i++) {
    double act = solver->stable ? (scores ? kissat_get_heap_score (scores, i) : 0) : (double) solver->links[i].stamp;
    if (act > 0) {
      double p = act / sum;
      h -= p * (log (p) / 0.69314718);
    }
  }
  return h;
}

static double calculate_h_conf (unsigned *window) {
  double h = 0;
  bool skip[256];
  for (int i = 0; i < 256; i++) skip[i] = false;
  for (int i = 0; i < 256; i++) {
    if (skip[i]) continue;
    int c = 1;
    for (int j = i + 1; j < 256; j++) {
      if (window[i] == window[j]) {
        c++;
        skip[j] = true;
      }
    }
    double p = c / 256.0;
    h -= p * (log (p) / 0.69314718);
  }
  return h;
}

bool kissat_restarting (kissat * solver) {
  assert (solver->unassigned);
  
  // Step 1: Static State and Maintenance
  static double alphas[3] = {1.0, 1.0, 1.0};
  static double betas[3] = {1.0, 1.0, 1.0};
  static unsigned window[256] = {0};
  static uint64_t w_ptr = 0;
  static uint64_t last_conf = 0;
  static uint64_t last_decay_conf = 0;
  static uint64_t last_rest = 0;
  static int current_arm = 0;
  static int force_stochastic = 0;
  static double last_h_act = -1.0;

  // Track conflict levels in sliding window
  while (last_conf < CONFLICTS) {
    window[w_ptr % 256] = solver->level;
    w_ptr++;
    last_conf++;
  }
  
  double h_conf = calculate_h_conf (window);

  // Step 5: Dynamic Decay every 500 conflicts
  if (CONFLICTS >= last_decay_conf + 500) {
    double gamma = 0.9 + 0.09 * (h_conf / 8.0);
    if (gamma > 0.99) gamma = 0.99;
    for (int i = 0; i < 3; i++) {
      alphas[i] *= gamma;
      betas[i] *= gamma;
      if (alphas[i] < 1.0) alphas[i] = 1.0;
      if (betas[i] < 1.0) betas[i] = 1.0;
    }
    last_decay_conf = CONFLICTS;
  }

  // Step 3 & 5: Reward Update and Thompson Sampling Selection
  if (solver->statistics.restarts > last_rest) {
    double h_act = calculate_activity_entropy (solver);
    if (last_h_act < 0) last_h_act = h_act;
    double h_delta = h_act - last_h_act;
    double lbd = AVERAGE (slow_glue);
    
    // Reward R = (w1 / LBD_ema) + (w2 * H_delta)
    double reward = (0.5 / (lbd + 1.0)) + (0.5 * h_delta);
    if (reward < 0) reward = 0;
    if (reward > 1) reward = 1;

    // Update Beta distribution for the arm that triggered the last restart
    alphas[current_arm] += reward;
    betas[current_arm] += (1.0 - reward);

    // Thompson Sampling: Sample from Beta(alpha, beta) using Normal approximation
    double best_sample = -1.0;
    for (int i = 0; i < 3; i++) {
      double mu = alphas[i] / (alphas[i] + betas[i]);
      double var = (alphas[i] * betas[i]) / (pow(alphas[i] + betas[i], 2) * (alphas[i] + betas[i] + 1.0));
      double sample = mu + sqrt (var) * normal_sample (solver);
      if (sample > best_sample) {
        best_sample = sample;
        current_arm = i;
      }
    }

    // Step 4: Contextual Shift Trigger
    if (h_conf < 1.5) force_stochastic = 3;
    if (force_stochastic > 0) {
      current_arm = 2; // Arm 2: Stochastic
      force_stochastic--;
    }

    last_h_act = h_act;
    last_rest = solver->statistics.restarts;
  }

  // Baseline Kissat Checks
  if (!GET_OPTION (restart)) return false;
  if (!solver->level) return false;
  if (CONFLICTS < solver->limits.restart.conflicts) return false;

  // Step 2: Execute Arm Strategy
  if (current_arm == 0) {
    // Arm 0: Luby / Reluctant
    if (solver->stable) return kissat_reluctant_triggered (&solver->reluctant);
    else {
      const double fast = AVERAGE (fast_glue);
      const double slow = AVERAGE (slow_glue);
      const double margin = (100.0 + GET_OPTION (restartmargin)) / 100.0;
      return (margin * slow <= fast);
    }
  } else if (current_arm == 1) {
    // Arm 1: Glucose-style LBD
    const double fast = AVERAGE (fast_glue);
    const double slow = AVERAGE (slow_glue);
    const double margin = (100.0 + GET_OPTION (restartmargin)) / 100.0;
    return (margin * slow <= fast);
  } else {
    // Arm 2: Stochastic p=0.01
    return (kissat_pick_double (&solver->random) < 0.01);
  }
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
