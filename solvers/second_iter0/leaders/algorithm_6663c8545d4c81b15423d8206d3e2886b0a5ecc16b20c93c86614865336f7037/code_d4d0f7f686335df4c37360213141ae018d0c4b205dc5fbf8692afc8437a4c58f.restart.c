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

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <inttypes.h>

#define WINDOW_SIZE 256
#define DECAY_INTERVAL 1000
#define STOCHASTIC_PROB 0.01
#define ENTROPY_THRESHOLD 1.5
#define LOG2_VAL 0.6931471805599453

bool kissat_restarting (kissat *solver) {
  assert (solver->unassigned);
  if (!GET_OPTION (restart))
    return false;
  if (!solver->level)
    return false;
  if (CONFLICTS < solver->limits.restart.conflicts)
    return false;

  // Persistent MAB state
  static double alphas[3] = {1.0, 1.0, 1.0};
  static double betas[3] = {1.0, 1.0, 1.0};
  static unsigned arm = 0;
  static uint64_t last_restarts = 0;
  static uint64_t last_c = 0;
  static uint64_t last_decay = 0;
  static double last_h_act = 0;
  static unsigned window[WINDOW_SIZE] = {0};
  static unsigned w_head = 0;
  static int force_count = 0;
  static bool initialized = false;

  // Step 1: Maintain a sliding window of the last 256 conflict decision levels
  if (CONFLICTS > last_c) {
    window[w_head] = solver->level;
    w_head = (w_head + 1) % WINDOW_SIZE;
    last_c = CONFLICTS;
  }

  // Step 5: Thompson Sampling Decay (0.99 per 1000 conflicts)
  if (CONFLICTS >= last_decay + DECAY_INTERVAL) {
    for (int i = 0; i < 3; i++) {
      alphas[i] = (alphas[i] > 1.0) ? (alphas[i] * 0.99) : 1.0;
      betas[i] = (betas[i] > 1.0) ? (betas[i] * 0.99) : 1.0;
    }
    last_decay = CONFLICTS;
  }

  // If a restart occurred since the last call, update the reward for the previous arm and pick a new one
  if (solver->statistics.restarts > last_restarts || !initialized) {
    
    // Calculate Variable Activity Entropy (H_act)
    double h_act = 0;
    double sum_act = 0;
    heap *scores = kissat_get_scores (solver);
    for (unsigned i = 0; i < solver->vars; i++) {
      sum_act += kissat_get_heap_score (scores, i);
    }
    if (sum_act > 0) {
      for (unsigned i = 0; i < solver->vars; i++) {
        double s = kissat_get_heap_score (scores, i);
        if (s > 0) {
          double p = s / sum_act;
          h_act -= p * (log (p) / LOG2_VAL);
        }
      }
    }

    if (initialized) {
      // Step 3: Reward Function R = (w1 / LBD_ema) + (w2 * H_delta)
      double lbd_ema = AVERAGE (slow_glue);
      double h_delta = h_act - last_h_act;
      double reward = (1.0 / (lbd_ema + 1.0)) + h_delta;
      if (reward < 0) reward = 0;
      
      // Update Thompson Sampling weights
      alphas[arm] += reward;
      betas[arm] += 1.0;
    }

    last_h_act = h_act;
    last_restarts = solver->statistics.restarts;
    initialized = true;

    // Step 1: Calculate Shannon Entropy (H) of the conflict decision level distribution
    double H_levels = 0;
    bool processed[WINDOW_SIZE] = { false };
    for (int i = 0; i < WINDOW_SIZE; i++) {
      if (processed[i]) continue;
      unsigned count = 1;
      for (int j = i + 1; j < WINDOW_SIZE; j++) {
        if (window[i] == window[j]) {
          count++;
          processed[j] = true;
        }
      }
      double p = (double)count / (double)WINDOW_SIZE;
      H_levels -= p * (log (p) / LOG2_VAL);
    }

    // Step 4: Contextual Shift Trigger
    if (force_count > 0) {
      arm = 2;
      force_count--;
    } else if (H_levels < ENTROPY_THRESHOLD) {
      arm = 2;
      force_count = 2; // Current + next 2 = 3 total
    } else {
      // Step 5: Thompson Sampling Arm Selection
      double best_sample = -1.0;
      for (int i = 0; i < 3; i++) {
        double a = alphas[i];
        double b = betas[i];
        double mu = a / (a + b);
        double var = (a * b) / (pow (a + b, 2) * (a + b + 1));
        // Approximate Beta sample using Normal distribution
        double sample = mu + (kissat_pick_double (&solver->random) + kissat_pick_double (&solver->random) - 1.0) * sqrt (var);
        if (sample > best_sample) {
          best_sample = sample;
          arm = i;
        }
      }
    }
  }

  // Step 2: Define and Execute Arm Triggers
  if (arm == 0) {
    // Arm 0: Luby sequence (Reluctant)
    return kissat_reluctant_triggered (&solver->reluctant);
  } else if (arm == 1) {
    // Arm 1: Glucose-style LBD-based
    const double fast = AVERAGE (fast_glue);
    const double slow = AVERAGE (slow_glue);
    const double margin = (100.0 + GET_OPTION (restartmargin)) / 100.0;
    const double limit = margin * slow;
    return (limit <= fast);
  } else {
    // Arm 2: Stochastic restart with probability p=0.01
    return (kissat_pick_double (&solver->random) < STOCHASTIC_PROB);
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
