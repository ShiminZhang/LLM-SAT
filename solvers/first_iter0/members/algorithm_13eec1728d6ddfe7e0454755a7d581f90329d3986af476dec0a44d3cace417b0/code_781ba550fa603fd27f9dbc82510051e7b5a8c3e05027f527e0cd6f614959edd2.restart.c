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

bool kissat_restarting (kissat *solver) {
  assert (solver->unassigned);
  if (!GET_OPTION (restart))
    return false;
  if (!solver->level)
    return false;
  if (CONFLICTS < solver->limits.restart.conflicts)
    return false;

  // VICTR with Logistic Soft-Bernoulli Updates Implementation
  #define VICTR_WINDOW_SIZE 512
  #define VICTR_DECAY_INTERVAL 10000
  #define VICTR_DECAY_FACTOR 0.95

  static double alphas[3][3], betas[3][3];
  static uint16_t lbd_window[VICTR_WINDOW_SIZE];
  static uint64_t w_ptr = 0;
  static uint64_t last_c_update = 0;
  static uint64_t last_decay = 0;
  static uint64_t c_at_last_restart = 0;
  static uint64_t t_at_last_restart = 0;
  static int current_arm = -1;
  static int current_ctx = -1;
  static bool victr_init = false;

  // Initialization
  if (!victr_init) {
    double init_lbd = AVERAGE (slow_glue);
    if (init_lbd < 1.0) init_lbd = 2.0;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        alphas[i][j] = 1.0;
        betas[i][j] = 1.0;
      }
    }
    for (int i = 0; i < VICTR_WINDOW_SIZE; i++) {
      lbd_window[i] = (uint16_t)init_lbd;
    }
    victr_init = true;
  }

  // Step 7: Decay factor applied every 10,000 conflicts
  if (CONFLICTS >= last_decay + VICTR_DECAY_INTERVAL) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        alphas[i][j] *= VICTR_DECAY_FACTOR;
        betas[i][j] *= VICTR_DECAY_FACTOR;
      }
    }
    last_decay = CONFLICTS;
  }

  // Step 1: Maintain sliding window of last 512 learned clause LBDs
  if (CONFLICTS > last_c_update) {
    double cur_lbd = AVERAGE (fast_glue);
    lbd_window[w_ptr] = (cur_lbd > 1.0) ? (uint16_t)cur_lbd : 1;
    w_ptr = (w_ptr + 1) % VICTR_WINDOW_SIZE;
    last_c_update = CONFLICTS;
  }

  // Identify context if we are starting a new restart decision cycle
  if (current_arm == -1) {
    // Step 2: Calculate Coefficient of Variation (CV)
    double sum = 0, sq_sum = 0;
    for (int i = 0; i < VICTR_WINDOW_SIZE; i++) {
      sum += lbd_window[i];
      sq_sum += (double)lbd_window[i] * lbd_window[i];
    }
    double mean = sum / VICTR_WINDOW_SIZE;
    double var = (sq_sum / VICTR_WINDOW_SIZE) - (mean * mean);
    double std_dev = sqrt (var > 0 ? var : 0);
    double cv = (mean > 0) ? (std_dev / mean) : 0;

    // Step 3: Discretize search state into contexts
    if (cv < 0.6) current_ctx = 0;      // Stagnant
    else if (cv <= 1.4) current_ctx = 1; // Steady
    else current_ctx = 2;              // Volatile

    // Step 4: Thompson Sampling from Beta distribution
    double samples[3];
    for (int i = 0; i < 3; i++) {
      double a = alphas[current_ctx][i];
      double b = betas[current_ctx][i];
      // Normal approximation for Beta sampling (Mean + Noise * StdDev)
      double m_arm = a / (a + b);
      double s_arm = sqrt ((a * b) / (pow (a + b, 2) * (a + b + 1.0)));
      double noise = 0;
      for (int n = 0; n < 4; n++) noise += kissat_pick_double (&solver->random);
      samples[i] = m_arm + (noise - 2.0) * 1.73205 * s_arm;
    }
    // Select arm with highest sample
    current_arm = (samples[0] > samples[1]) ? (samples[0] > samples[2] ? 0 : 2) : (samples[1] > samples[2] ? 1 : 2);
  }

  // Evaluate the selected Arm's restart condition
  bool triggered = false;
  if (current_arm == 0) {
    // Arm 0: Luby-style restarts
    triggered = kissat_reluctant_triggered (&solver->reluctant);
  } else if (current_arm == 1) {
    // Arm 1: Aggressive EMA-based restarts (margin 1.0)
    triggered = (AVERAGE (fast_glue) >= AVERAGE (slow_glue));
  } else {
    // Arm 2: Lazy EMA-based restarts (margin from options, typically > 1.0)
    const double margin = (100.0 + GET_OPTION (restartmargin)) / 100.0;
    triggered = (AVERAGE (fast_glue) >= margin * AVERAGE (slow_glue));
  }

  if (triggered) {
    // Step 5: Reward R
    double global_lbd = AVERAGE (slow_glue);
    double sum_window = 0;
    for (int i = 0; i < VICTR_WINDOW_SIZE; i++) sum_window += lbd_window[i];
    double window_lbd = sum_window / VICTR_WINDOW_SIZE;
    
    double cur_ticks = (double)solver->ticks - t_at_last_restart;
    double cur_confs = (double)CONFLICTS - c_at_last_restart;
    double current_cps = (cur_ticks > 0) ? cur_confs / cur_ticks : 0;
    double global_cps = (solver->ticks > 0) ? (double)CONFLICTS / (double)solver->ticks : 0;

    double reward = 1.0;
    if (window_lbd > 0.1 && global_cps > 0.000001) {
        reward = (global_lbd / window_lbd) * (current_cps / global_cps);
    }

    // Step 6: Logistic Soft-Bernoulli Update
    double p = 1.0 / (1.0 + exp (-2.0 * (reward - 1.0)));
    alphas[current_ctx][current_arm] += p;
    betas[current_ctx][current_arm] += (1.0 - p);

    // Reset cycle tracking
    c_at_last_restart = CONFLICTS;
    t_at_last_restart = solver->ticks;
    current_arm = -1; 
    return true;
  }

  return false;
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
