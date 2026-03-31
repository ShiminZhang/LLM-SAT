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
    // Persistent state for MAB: Circular buffer of signatures and Beta distributions
    static uint64_t signatures[8];
    static int sig_count = 0;
    static int sig_idx = 0;
    static double alphas[2] = {1.0, 1.0};
    static double betas[2] = {1.0, 1.0};
    static uint64_t last_decay_conflicts = 0;

    // Step 1: Calculate Trail Signature (XOR-sum of decision levels of first 32 assigned variables)
    uint64_t current_sig = 0;
    unsigned count = 0;
    unsigned trail_size = SIZE_STACK(solver->trail);
    for (unsigned l = 0; l <= solver->level && count < 32; l++) {
        // Frame(l).trail is the start index of variables assigned at level l
        // Frame(l+1).trail is the start index of variables assigned at level l+1
        unsigned end = (l < solver->level) ? FRAME(l + 1).trail : trail_size;
        while (count < end && count < 32) {
            current_sig ^= (uint64_t)l;
            count++;
        }
    }

    // Step 2: Calculate Overlap (Jaccard similarity with last 8 signatures)
    double overlap = 0;
    if (sig_count > 0) {
        for (int i = 0; i < sig_count; i++) {
            uint64_t s1 = current_sig;
            uint64_t s2 = signatures[i];
            uint64_t combined = s1 | s2;
            if (combined == 0) {
                overlap += 1.0;
            } else {
                // Manual popcount for portability (Brian Kernighan's way)
                int intersection_bits = 0;
                for (uint64_t n = s1 & s2; n; n &= n - 1) intersection_bits++;
                int union_bits = 0;
                for (uint64_t n = combined; n; n &= n - 1) union_bits++;
                overlap += (double)intersection_bits / (double)union_bits;
            }
        }
        overlap /= sig_count;
    }

    // Step 2 (cont): Calculate Reward R
    double lbd_global = AVERAGE(slow_glue);
    double lbd_current = AVERAGE(fast_glue);
    double reward = 0;
    if (lbd_current > 0) {
        reward = (lbd_global / lbd_current) * (1.0 - overlap);
    }

    // Step 3: Update Beta distribution for the arm that was just active
    // solver->heuristic maps to Arm 0 (Luby/VSIDS) or Arm 1 (Glucose/CHB)
    if (reward > 1.0) {
        alphas[solver->heuristic] += 1.0;
    } else {
        betas[solver->heuristic] += 1.0;
    }

    // Update circular buffer with the current signature
    signatures[sig_idx] = current_sig;
    sig_idx = (sig_idx + 1) % 8;
    if (sig_count < 8) sig_count++;

    // Step 4: Polarization-Aware Adaptive Bandit Decay (Every 2^10 conflicts)
    if (CONFLICTS >= last_decay_conflicts + 1024) {
        for (int i = 0; i < 2; i++) {
            double a = alphas[i];
            double b = betas[i];
            double p = fabs(a - b) / (a + b);
            // Dynamic decay factor gamma = 0.5 + 0.4 * (1 - P)
            double gamma = 0.5 + (0.4 * (1.0 - p));
            alphas[i] *= gamma;
            betas[i] *= gamma;
        }
        last_decay_conflicts = CONFLICTS;
    }

    // Step 3 (cont): Thompson Sampling to select the heuristic for the next period
    double samples[2];
    for (int i = 0; i < 2; i++) {
        double a = alphas[i];
        double b = betas[i];
        // For Thompson Sampling from Beta(a,b), we use a Normal approximation
        // where mu = a/(a+b) and sigma^2 = (ab)/((a+b)^2(a+b+1))
        double mu = a / (a + b);
        double var = (a * b) / ((a + b) * (a + b) * (a + b + 1.0));
        
        // Box-Muller transform for sampling from Normal distribution
        double u1 = kissat_pick_double(&solver->random);
        double u2 = kissat_pick_double(&solver->random);
        if (u1 < 1e-9) u1 = 1e-9; // Avoid log(0)
        double z = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979 * u2);
        samples[i] = mu + z * sqrt(var);
    }
    
    // Select the arm with the highest sample
    solver->heuristic = (samples[0] > samples[1]) ? 0 : 1;

    // Maintain solver state consistency (Kissat MAB bookkeeping)
    unsigned idx;
    for (all_variables(idx)) {
        solver->mab_chosen[idx] = 0;
    }
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
