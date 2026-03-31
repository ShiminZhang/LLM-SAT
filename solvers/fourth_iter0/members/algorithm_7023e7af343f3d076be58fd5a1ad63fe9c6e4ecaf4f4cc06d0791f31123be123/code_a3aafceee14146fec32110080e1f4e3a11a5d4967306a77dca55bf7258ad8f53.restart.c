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
  // Step 1: Trail Signature
  // 64-bit XOR-sum of the decision levels of the first 32 variables assigned.
  uint64_t signature = 0;
  const unsigned trail_size = SIZE_STACK (solver->trail);
  const unsigned count = (trail_size < 32) ? trail_size : 32;
  unsigned frame_idx = 0;
  for (unsigned i = 0; i < count; i++) {
    // Find the decision level for the literal at trail position i
    while (frame_idx < solver->level && i >= FRAME (frame_idx + 1).trail) {
      frame_idx++;
    }
    signature ^= (uint64_t) frame_idx;
  }

  // Step 2: Reward Calculation
  // R = (LBD_global / LBD_current) * (1 - Overlap)
  // Overlap is the Jaccard similarity between current Trail Signature and a circular buffer of last 8.
  static uint64_t history[8];
  static unsigned h_idx = 0;
  static unsigned h_cnt = 0;
  double max_overlap = 0;

  for (unsigned i = 0; i < h_cnt; i++) {
    const uint64_t h = history[i];
    const unsigned inter = __builtin_popcountll (signature & h);
    const unsigned uni = __builtin_popcountll (signature | h);
    const double overlap = (uni > 0) ? (double) inter / uni : 1.0;
    if (overlap > max_overlap)
      max_overlap = overlap;
  }

  // Update circular buffer
  history[h_idx] = signature;
  h_idx = (h_idx + 1) % 8;
  if (h_cnt < 8)
    h_cnt++;

  const double lbd_global = AVERAGE (slow_glue);
  double lbd_current = AVERAGE (fast_glue);
  if (lbd_current < 1e-6)
    lbd_current = 1e-6; // Avoid division by zero
  const double reward = (lbd_global / lbd_current) * (1.0 - max_overlap);

  // Step 3: Thompson Sampling Integration
  // Maintain Beta distribution (alpha, beta) for each arm.
  // Increment alpha by R and beta by 1/R for the active arm.
  static double alpha[2] = { 1.0, 1.0 };
  static double beta[2] = { 1.0, 1.0 };

  const unsigned active = solver->heuristic;
  alpha[active] += reward;
  beta[active] += (reward > 1e-6) ? (1.0 / reward) : 1000.0;

  // Step 4: Scaling to favor recent search topology
  // Every 2^10 conflicts, scale alpha and beta by 0.5.
  static uint64_t last_scale_conflicts = 0;
  if (CONFLICTS >= last_scale_conflicts + 1024) {
    for (unsigned i = 0; i < 2; i++) {
      alpha[i] *= 0.5;
      beta[i] *= 0.5;
    }
    last_scale_conflicts = CONFLICTS;
  }

  // Select Arm using Thompson Sampling (Gaussian approximation for Beta)
  double best_sample = -1.0;
  unsigned selected_arm = solver->heuristic;

  for (unsigned i = 0; i < 2; i++) {
    const double sum = alpha[i] + beta[i];
    const double mu = alpha[i] / sum;
    const double var = (alpha[i] * beta[i]) / (sum * sum * (sum + 1.0));
    const double sigma = sqrt (var);

    // Box-Muller transform for Gaussian sampling
    const double u1 = kissat_pick_double (&solver->random);
    const double u2 = kissat_pick_double (&solver->random);
    const double z =
      sqrt (-2.0 * log (u1 + 1e-9)) * cos (2.0 * 3.14159265358979 * u2);
    const double sample = mu + z * sigma;

    if (sample > best_sample) {
      best_sample = sample;
      selected_arm = i;
    }
  }
  solver->heuristic = selected_arm;

  // Cleanup internal MAB tracking fields to prevent interference
  for (all_variables (idx)) {
    solver->mab_chosen[idx] = 0;
  }
  solver->mab_chosen_tot = 0;
  solver->mab_decisions = 0;
  solver->mab_conflicts = 0;

  // Sync solver struct reward fields for statistical visibility
  solver->mab_reward[0] = alpha[0];
  solver->mab_reward[1] = alpha[1];
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
