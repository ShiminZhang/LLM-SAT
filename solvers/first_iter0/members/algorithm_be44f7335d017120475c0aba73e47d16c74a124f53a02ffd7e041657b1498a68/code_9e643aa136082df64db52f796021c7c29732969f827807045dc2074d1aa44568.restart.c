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
  // Static variables to maintain state across calls to kissat_restarting
  static unsigned S[3] = {1, 1, 1};
  static unsigned F[3] = {1, 1, 1};
  static uint8_t arm_hist[128];
  static bool success_hist[128];
  static int win_ptr = 0;
  static int win_size = 0;
  static int active_arm = 0;
  static double global_R_ema = 0.5;
  static double current_R_sum = 0.0;
  static uint64_t current_R_cnt = 0;
  static uint64_t last_fixed_restart = 0;

  #define ARM_LUBY 0
  #define ARM_EMA 1
  #define ARM_FIXED 2

  // Basic guards as found in the baseline
  assert (solver->unassigned);
  if (!GET_OPTION (restart))
    return false;
  if (!solver->level)
    return false;

  // Step 2: Reward R for each conflict
  // Calculate efficiency using current LBD and a proxy for jump distance
  double current_lbd = (double) AVERAGE (fast_glue);
  if (current_lbd < 1.0) current_lbd = 1.0;
  // Use current level as a proxy for potential search efficiency/jump distance
  double jump_dist = (double) solver->level;
  double R = log2 (jump_dist + 1.0) / current_lbd;
  current_R_sum += R;
  current_R_cnt++;

  // Step 4: Search Stagnation Coefficient G
  double active_vars = (double) (solver->vars - solver->unassigned);
  double total_vars = (double) (solver->vars > 0 ? solver->vars : 1);
  double curr_level = (double) solver->level;
  double max_level = (double) (solver->max_level > 0 ? solver->max_level : 1);
  double G = (active_vars / total_vars) * (curr_level / max_level);
  if (G > 1.0) G = 1.0;

  // Step 5: Thompson Sampling to select next arm
  double samples[3];
  for (int i = 0; i < 3; i++) {
    // Sample from Beta(S_i + 1, F_i + 1) using Gamma distributions
    double x = 0.0, y = 0.0;
    for (unsigned j = 0; j < S[i] + 1; j++)
      x -= log (kissat_pick_double (&solver->random) + 1e-10);
    for (unsigned j = 0; j < F[i] + 1; j++)
      y -= log (kissat_pick_double (&solver->random) + 1e-10);
    samples[i] = x / (x + y);
  }

  // Apply Phase Bias
  if (solver->stable)
    samples[ARM_LUBY] *= (1.0 + G);
  else
    samples[ARM_EMA] *= (1.0 + G);

  // Step 6: Identify arm with highest modified sample (i_max)
  int i_max = 0;
  if (samples[1] > samples[0]) i_max = 1;
  if (samples[2] > samples[i_max]) i_max = 2;

  // Hysteresis switch logic
  if (i_max != active_arm) {
    double margin = 0.05 * (1.0 - G);
    if (samples[i_max] > samples[active_arm] + margin) {
      active_arm = i_max;
    }
  }

  // Execute corresponding restart trigger logic for the active arm
  bool triggered = false;
  if (active_arm == ARM_LUBY) {
    triggered = kissat_reluctant_triggered (&solver->reluctant);
  } else if (active_arm == ARM_EMA) {
    const double fast = AVERAGE (fast_glue);
    const double slow = AVERAGE (slow_glue);
    const double margin = (100.0 + GET_OPTION (restartmargin)) / 100.0;
    triggered = (margin * slow <= fast);
  } else {
    // Arm 2: Aggressive Fixed-Interval (e.g., every 64 conflicts)
    triggered = (CONFLICTS - last_fixed_restart >= 64);
  }

  // Step 3: Update success/failure counts at the decision point
  if (triggered) {
    double avg_R = (current_R_cnt > 0) ? (current_R_sum / current_R_cnt) : 0.0;
    bool success = (avg_R > global_R_ema);

    // Maintain sliding window of the last 128 restarts
    if (win_size == 128) {
      unsigned old_arm = (unsigned) arm_hist[win_ptr];
      bool old_success = success_hist[win_ptr];
      if (old_success) {
        if (S[old_arm] > 0) S[old_arm]--;
      } else {
        if (F[old_arm] > 0) F[old_arm]--;
      }
    } else {
      win_size++;
    }

    // Store new result in window
    arm_hist[win_ptr] = (uint8_t) active_arm;
    success_hist[win_ptr] = success;
    if (success) S[active_arm]++;
    else F[active_arm]++;
    win_ptr = (win_ptr + 1) % 128;

    // Update global moving average of reward R
    global_R_ema = 0.95 * global_R_ema + 0.05 * avg_R;

    // Reset cycle data
    current_R_sum = 0.0;
    current_R_cnt = 0;
    last_fixed_restart = CONFLICTS;
  }

  return triggered;
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
