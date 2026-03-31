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
    // Step 1: Calculate Trail Signature (XOR-sum of levels of first 32 variables assigned)
    uint64_t signature = 0;
    const unsigned *trail = (const unsigned *)BEGIN_STACK(solver->trail);
    const unsigned trail_size = SIZE_STACK(solver->trail);
    const unsigned n_signature = (trail_size < 32) ? trail_size : 32;

    for (unsigned i = 0; i < n_signature; i++) {
        unsigned level = 0;
        // Find decision level of the i-th assigned literal
        for (unsigned l = 1; l <= solver->level; l++) {
            if (i < FRAME(l).trail) {
                level = l - 1;
                break;
            }
            level = l;
        }
        signature ^= (uint64_t)level;
    }

    // Step 2: Calculate Overlap (Average Jaccard similarity with circular buffer of last 8)
    double overlap = 0;
    // Use solver->mab_chosen as storage for the circular buffer (8 uint64_t = 16 unsigned)
    // Ensure we have enough space in the allocated array
    if (solver->vars >= 16) {
        unsigned count = 0;
        unsigned buffer_limit = 8;
        unsigned current_total = solver->mab_chosen_tot;
        
        if (current_total > 0) {
            unsigned limit = (current_total < buffer_limit) ? current_total : buffer_limit;
            double sum_jaccard = 0;
            for (unsigned i = 0; i < limit; i++) {
                uint64_t b = ((uint64_t)solver->mab_chosen[i * 2 + 1] << 32) | (uint64_t)solver->mab_chosen[i * 2];
                uint64_t and_bits = signature & b;
                uint64_t or_bits = signature | b;
                if (or_bits == 0) {
                    sum_jaccard += 1.0;
                } else {
                    sum_jaccard += (double)__builtin_popcountll(and_bits) / (double)__builtin_popcountll(or_bits);
                }
            }
            overlap = sum_jaccard / (double)limit;
            count = limit;
        }

        // Update circular buffer
        unsigned store_idx = (current_total % buffer_limit) * 2;
        solver->mab_chosen[store_idx] = (unsigned)(signature & 0xFFFFFFFF);
        solver->mab_chosen[store_idx + 1] = (unsigned)(signature >> 32);
        solver->mab_chosen_tot++;
    }

    // Reward R = (LBD_global / LBD_current) * (1 - Overlap)
    double fast_lbd = AVERAGE(fast_glue);
    double slow_lbd = AVERAGE(slow_glue);
    double lbd_ratio = (fast_lbd > 0) ? (slow_lbd / fast_lbd) : 1.0;
    double reward = lbd_ratio * (1.0 - overlap);
    if (reward < 1e-6) reward = 1e-6; // Prevent division by zero or negative

    // Step 3: Thompson Sampling Update
    // alpha = mab_reward[0], beta = mab_reward[1]
    if (solver->mab_reward[0] <= 0) solver->mab_reward[0] = 1.0;
    if (solver->mab_reward[1] <= 0) solver->mab_reward[1] = 1.0;

    solver->mab_reward[0] += reward;        // alpha <- alpha + R
    solver->mab_reward[1] += (1.0 / reward); // beta <- beta + 1/R

    // Sample theta ~ Beta(alpha, beta) using Normal approximation for Thompson Sampling
    double alpha = solver->mab_reward[0];
    double beta = solver->mab_reward[1];
    double mu = alpha / (alpha + beta);
    double variance = (alpha * beta) / ((alpha + beta) * (alpha + beta) * (alpha + beta + 1.0));
    double sigma = sqrt(variance);

    // Box-Muller transform for Gaussian sample
    double u1 = kissat_pick_double(&solver->random);
    double u2 = kissat_pick_double(&solver->random);
    double z = sqrt(-2.0 * log(u1 + 1e-12)) * cos(2.0 * 3.14159265358979323846 * u2);
    double theta = mu + sigma * z;

    // Select arm: 1 = Glucose, 0 = Luby
    solver->heuristic = (theta > 0.5) ? 1 : 0;

    // Step 4: Periodic Scaling (Every 2^10 conflicts)
    if (solver->mab_conflicts >= 1024) {
        solver->mab_reward[0] *= 0.5;
        solver->mab_reward[1] *= 0.5;
        solver->mab_conflicts = 0;
    }

    // Reset standard MAB tracking fields
    solver->mab_decisions = 0;
    // solver->mab_conflicts is handled in the scaling step
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
