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

bool kissat_restarting(kissat *solver) {
  assert(solver->unassigned);

  if (!GET_OPTION(restart))
    return false;
  if (!solver->level)
    return false;
  if (CONFLICTS < solver->limits.restart.conflicts)
    return false;

  // Persistent State for Adaptive Gradient-Triggered Stochastic Diversification
  static unsigned window[256];
  static unsigned w_ptr = 0;
  static unsigned w_count = 0;
  static uint64_t last_conflict_count = 0;
  static double h_avg = 0;
  static int consec_drops = 0;
  static double alpha[3] = {1.0, 1.0, 1.0};
  static double beta[3] = {1.0, 1.0, 1.0};
  static int current_arm = 0;
  static double last_h_act = 0;
  static bool prev_restart_decided = true;
  static uint64_t last_decay_conflicts = 0;
  static int div_restarts_left = 0;
  static bool initialized = false;

  // Step 1: Maintain a sliding window of the last 256 conflict decision levels
  if (CONFLICTS > last_conflict_count) {
    window[w_ptr] = solver->level;
    w_ptr = (w_ptr + 1) % 256;
    if (w_count < 256)
      w_count++;
    last_conflict_count = CONFLICTS;

    // Step 4: Dynamic Entropy-Gradient Trigger (Calculated every 256 conflicts)
    if (w_count == 256 && (CONFLICTS % 256 == 0)) {
      double h_new = 0;
      for (int i = 0; i < 256; i++) {
        int count = 0;
        for (int j = 0; j < 256; j++)
          if (window[i] == window[j])
            count++;
        double p = (double)count / 256.0;
        h_new -= (p * log2(p)) / (double)count;
      }

      if (!initialized) {
        h_avg = h_new;
        initialized = true;
      }

      double rel_change = (h_new - h_avg) / (h_avg + 1e-9);
      if (rel_change < -0.15) {
        consec_drops++;
      } else {
        consec_drops = 0;
      }

      // Trigger Diversification Phase
      if (consec_drops >= 2) {
        div_restarts_left = 5;
        consec_drops = 0;
        // Flip target phases of top 10% most active variables
        unsigned flip_limit = solver->vars / 10;
        if (solver->stable) {
          for (unsigned i = 0; i < flip_limit && i < solver->heap.vars; i++) {
            unsigned v_idx = solver->heap.nodes[i];
            solver->phases[v_idx].target = !solver->phases[v_idx].target;
          }
        } else {
          for (unsigned i = 0; i < flip_limit; i++) {
            solver->phases[i].target = !solver->phases[i].target;
          }
        }
      }
      h_avg = 0.9 * h_avg + 0.1 * h_new;
    }
  }

  // Step 3 & 5: MAB Reward and Thompson Sampling Update
  if (prev_restart_decided) {
    // Reward calculation for the previous interval
    double current_h_act = 0;
    double sum_act = 0;
    const double *scores = kissat_get_scores(solver);
    for (unsigned i = 0; i < solver->vars; i++)
      sum_act += scores[i];
    if (sum_act > 1e-9) {
      for (unsigned i = 0; i < solver->vars; i++) {
        double p = scores[i] / sum_act;
        if (p > 1e-9)
          current_h_act -= p * log2(p);
      }
    }

    double h_delta = current_h_act - last_h_act;
    last_h_act = current_h_act;

    // Normalize reward components to [0, 1]
    double r_h = (atan(h_delta * 10.0) / 1.5708 + 1.0) / 2.0;
    double r_lbd = 1.0 / (1.0 + AVERAGE(slow_glue));
    double reward = 0.5 * r_h + 0.5 * r_lbd;

    // Update weights for the arm that just finished
    alpha[current_arm] += reward;
    beta[current_arm] += (1.0 - reward);

    // Thompson Sampling for the next arm
    if (div_restarts_left > 0) {
      current_arm = 2; // Force Arm 2 during Diversification Phase
    } else {
      double max_sample = -1.0;
      for (int i = 0; i < 3; i++) {
        double mu = alpha[i] / (alpha[i] + beta[i]);
        double var = (alpha[i] * beta[i]) / (pow(alpha[i] + beta[i], 2) * (alpha[i] + beta[i] + 1.0) + 1e-9);
        // Box-Muller or simplified normal approximation for Beta sampling
        double noise = (kissat_pick_double(&solver->random) - 0.5) * 3.4641;
        double sample = mu + sqrt(var) * noise;
        if (sample > max_sample) {
          max_sample = sample;
          current_arm = i;
        }
      }
    }
    prev_restart_decided = false;
  }

  // Step 5: Thompson Sampling weight decay
  if (CONFLICTS >= last_decay_conflicts + 1000) {
    for (int i = 0; i < 3; i++) {
      alpha[i] *= 0.99;
      beta[i] *= 0.99;
    }
    last_decay_conflicts = CONFLICTS;
  }

  // Step 2: Arm Logic Execution
  bool restart_result = false;
  if (current_arm == 0) {
    // Arm 0: Luby sequence
    restart_result = kissat_reluctant_triggered(&solver->reluctant);
  } else if (current_arm == 1) {
    // Arm 1: Glucose-style LBD-based
    const double fast = AVERAGE(fast_glue);
    const double slow = AVERAGE(slow_glue);
    const double margin = (100.0 + GET_OPTION(restartmargin)) / 100.0;
    restart_result = (margin * slow <= fast);
  } else {
    // Arm 2: Stochastic restart with p=0.01
    restart_result = (kissat_pick_double(&solver->random) < 0.01);
  }

  if (restart_result) {
    prev_restart_decided = true;
    if (div_restarts_left > 0)
      div_restarts_left--;
  }

  return restart_result;
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
