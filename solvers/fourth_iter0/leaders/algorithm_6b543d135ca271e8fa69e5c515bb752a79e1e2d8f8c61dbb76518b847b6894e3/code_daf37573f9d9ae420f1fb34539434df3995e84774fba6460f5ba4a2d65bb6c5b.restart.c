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
    // Step 1: Calculate Trail Signature
    // 64-bit XOR-sum of the decision levels of the first 32 variables assigned.
    uint64_t signature = 0;
    const unsigned trail_size = SIZE_STACK(solver->trail);
    const unsigned count = (trail_size < 32) ? trail_size : 32;
    
    for (unsigned i = 0; i < count; i++) {
        unsigned level = 0;
        // In Kissat, variables at trail[i] have level 'l' if FRAME(l).trail <= i.
        // We iterate through frames to find the highest level 'l' such that its trail start is <= i.
        for (unsigned l = 1; l <= solver->level; l++) {
            if (FRAME(l).trail > i) break;
            level = l;
        }
        signature ^= (uint64_t)level;
    }

    // Step 2: Define Reward R = (LBD_global / LBD_current) * (1 - Overlap)
    // Overlap is the Jaccard similarity between the current Trail Signature and a circular buffer of the last 8.
    static uint64_t buffer[8];
    static unsigned b_idx = 0;
    static unsigned b_count = 0;
    
    double overlap = 0.0;
    if (b_count > 0) {
        bool found = false;
        uint64_t unique_elements[8];
        unsigned num_unique = 0;
        
        // Identify if current signature is in the buffer and count unique signatures in the buffer
        for (unsigned i = 0; i < b_count; i++) {
            if (buffer[i] == signature) found = true;
            bool already_added = false;
            for (unsigned j = 0; j < num_unique; j++) {
                if (unique_elements[j] == buffer[i]) {
                    already_added = true;
                    break;
                }
            }
            if (!already_added) {
                unique_elements[num_unique++] = buffer[i];
            }
        }
        
        // Jaccard similarity between {signature} and {buffer contents}
        // J(A, B) = |A ∩ B| / |A ∪ B|
        unsigned intersection_size = found ? 1 : 0;
        unsigned union_size = num_unique + (found ? 0 : 1);
        overlap = (double)intersection_size / (double)union_size;
    }

    // Update the circular buffer with the current signature
    buffer[b_idx] = signature;
    b_idx = (b_idx + 1) % 8;
    if (b_count < 8) b_count++;

    // LBD Global is the slow EMA, LBD Current is the fast EMA
    const double lbd_global = AVERAGE(slow_glue);
    const double lbd_current = AVERAGE(fast_glue);
    const double lbd_ratio = (lbd_current > 0) ? (lbd_global / lbd_current) : 1.0;
    const double reward = lbd_ratio * (1.0 - overlap);

    // Step 3: Thompson Sampling Integration
    // Arms: 0 (VSIDS/Luby) vs 1 (CHB/Glucose). Maintain Beta distributions (alpha, beta).
    static double alpha[2] = {1.0, 1.0};
    static double beta_param[2] = {1.0, 1.0};
    
    // Update Beta parameters based on reward
    if (reward > 1.0) {
        alpha[solver->heuristic] += 1.0;
    } else {
        beta_param[solver->heuristic] += 1.0;
    }

    // Step 4: Scaling every 2^10 (1024) conflicts
    static uint64_t last_scale_conflicts = 0;
    if (CONFLICTS >= last_scale_conflicts + 1024) {
        for (unsigned i = 0; i < 2; i++) {
            alpha[i] *= 0.5;
            beta_param[i] *= 0.5;
            // Ensure parameters stay >= 1.0 for valid Beta distribution sampling
            if (alpha[i] < 1.0) alpha[i] = 1.0;
            if (beta_param[i] < 1.0) beta_param[i] = 1.0;
        }
        last_scale_conflicts = CONFLICTS;
    }

    // Thompson Sampling: Sample from Beta(alpha, beta) for each arm and pick the maximum
    double max_sample = -1.0;
    unsigned best_arm = 0;
    for (unsigned i = 0; i < 2; i++) {
        double u = kissat_pick_double(&solver->random);
        double v = kissat_pick_double(&solver->random);
        // Numerical stability for pow(0, ...)
        if (u < 1e-11) u = 1e-11;
        if (v < 1e-11) v = 1e-11;
        
        // Beta sample approximation: X = U^(1/alpha) / (U^(1/alpha) + V^(1/beta))
        double sample_u = pow(u, 1.0 / alpha[i]);
        double sample_v = pow(v, 1.0 / beta_param[i]);
        double sample = sample_u / (sample_u + sample_v);
        
        if (sample > max_sample) {
            max_sample = sample;
            best_arm = i;
        }
    }
    
    // Update solver heuristic to the chosen arm
    solver->heuristic = best_arm;

    // Reset Kissat's internal MAB tracking to prevent interference
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
