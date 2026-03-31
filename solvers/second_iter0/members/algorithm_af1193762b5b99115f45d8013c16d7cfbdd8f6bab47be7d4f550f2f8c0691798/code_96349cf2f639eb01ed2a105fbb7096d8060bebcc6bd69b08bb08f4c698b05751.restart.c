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
  // Static variables to maintain bandit state without modifying the kissat struct
  static double rewards[2] = {1.0, 1.0};
  static double counts[2] = {1.0, 1.0};
  static double d_window[VTAB_WINDOW_SIZE];
  static double l_window[VTAB_WINDOW_SIZE];
  static double v_window[VTAB_WINDOW_SIZE];
  static unsigned window_ptr = 0;
  static unsigned window_filled = 0;
  static uint64_t last_conflict_count = 0;
  static uint64_t last_decay_conflict = 0;
  static unsigned last_decision_level = 0;
  static double long_term_velocity_sum = 0.0;
  static uint64_t long_term_velocity_count = 0;
  static unsigned low_velocity_consecutive = 0;
  static int active_arm = 0;
  static int forced_arm = -1;
  static unsigned forced_remaining = 0;

  // Basic Kissat restart guards
  if (!GET_OPTION (restart))
    return false;
  if (!solver->level)
    return false;
  if (CONFLICTS < solver->limits.restart.conflicts)
    return false;

  // Initialize state on first call
  if (last_conflict_count == 0) {
    last_conflict_count = CONFLICTS;
    last_decision_level = solver->level;
    last_decay_conflict = CONFLICTS;
    return false;
  }

  // Step 5 Calculation (Pre-calculation of Sigma for UCB and Phase Shift)
  double mean_v = 0, sigma_v = 0;
  if (window_filled > 0) {
    for (unsigned i = 0; i < window_filled; i++)
      mean_v += v_window[i];
    mean_v /= window_filled;
    double variance_v = 0;
    for (unsigned i = 0; i < window_filled; i++)
      variance_v += (v_window[i] - mean_v) * (v_window[i] - mean_v);
    sigma_v = sqrt (variance_v / window_filled);
  }
  double exploration_constant = 0.8 * sigma_v;

  // Step 2 & 3: Update sliding window and calculate search metrics
  if (CONFLICTS > last_conflict_count) {
    uint64_t delta_conflicts = CONFLICTS - last_conflict_count;
    // D is the backjump distance (levels cleared since last check)
    double current_D = (last_decision_level > solver->level) ? (double) (last_decision_level - solver->level) : 0.0;
    // L is the current conflict level
    double current_L = (double) solver->level;
    
    // Update window for all conflicts that passed since last call
    for (uint64_t i = 0; i < delta_conflicts; i++) {
      d_window[window_ptr] = current_D;
      l_window[window_ptr] = current_L;
      v_window[window_ptr] = (current_L > 0) ? (current_D / current_L) : 0.0;
      window_ptr = (window_ptr + 1) % VTAB_WINDOW_SIZE;
      if (window_filled < VTAB_WINDOW_SIZE)
        window_filled++;
    }

    // Step 3: Calculate Search Velocity (V)
    double avg_D = 0, avg_L = 0;
    for (unsigned i = 0; i < window_filled; i++) {
      avg_D += d_window[i];
      avg_L += l_window[i];
    }
    double current_velocity = (avg_L > 0) ? (avg_D / avg_L) : 0.0;

    // Step 4: Reward Update for the arm that was active during these conflicts
    double lbd_avg = AVERAGE (fast_glue);
    double lbd_influence = 1.0 - (lbd_avg / VTAB_LBD_CAP);
    if (lbd_influence < 0) lbd_influence = 0;
    double reward = current_velocity * lbd_influence;
    
    rewards[active_arm] += reward;
    counts[active_arm] += 1.0;

    // Step 6: Structural Phase Shift Detection
    double lt_velocity_avg = (long_term_velocity_count > 0) ? (long_term_velocity_sum / long_term_velocity_count) : 0.0;
    if (long_term_velocity_count > 0 && current_velocity < (0.2 * lt_velocity_avg)) {
      low_velocity_consecutive += (unsigned) delta_conflicts;
    } else {
      low_velocity_consecutive = 0;
    }

    if (forced_remaining == 0 && low_velocity_consecutive >= VTAB_PHASE_SHIFT_THRESHOLD) {
      low_velocity_consecutive = 0;
      // Determine non-dominant arm using current UCB
      double t = counts[0] + counts[1];
      double ucb0 = (rewards[0] / counts[0]) + exploration_constant * sqrt (log (t) / counts[0]);
      double ucb1 = (rewards[1] / counts[1]) + exploration_constant * sqrt (log (t) / counts[1]);
      forced_arm = (ucb1 > ucb0) ? 0 : 1;
      forced_remaining = VTAB_PHASE_SHIFT_DURATION;
    } else if (forced_remaining > 0) {
      if (delta_conflicts >= forced_remaining) forced_remaining = 0;
      else forced_remaining -= (unsigned) delta_conflicts;
    }

    // Update long-term stats
    long_term_velocity_sum += current_velocity * (double) delta_conflicts;
    long_term_velocity_count += delta_conflicts;
    last_conflict_count = CONFLICTS;
  }
  last_decision_level = solver->level;

  // Step 7: Temporal Decay
  if (CONFLICTS >= last_decay_conflict + VTAB_DECAY_INTERVAL) {
    rewards[0] *= VTAB_DECAY_FACTOR;
    rewards[1] *= VTAB_DECAY_FACTOR;
    counts[0] *= VTAB_DECAY_FACTOR;
    counts[1] *= VTAB_DECAY_FACTOR;
    if (counts[0] < 1.0) counts[0] = 1.0;
    if (counts[1] < 1.0) counts[1] = 1.0;
    last_decay_conflict = CONFLICTS;
  }

  // Step 5: Arm Selection (UCB1)
  double total_n = counts[0] + counts[1];
  double current_C = (forced_remaining > 0) ? (2.0 * exploration_constant) : exploration_constant;

  if (forced_remaining > 0) {
    active_arm = forced_arm;
  } else {
    double ucb0 = (rewards[0] / counts[0]) + current_C * sqrt (log (total_n) / counts[0]);
    double ucb1 = (rewards[1] / counts[1]) + current_C * sqrt (log (total_n) / counts[1]);
    active_arm = (ucb1 > ucb0) ? 1 : 0;
  }

  // Step 1: Execute selected Arm Strategy
  if (active_arm == 0) {
    // Arm 0: Aggressive Luby (Focused Search Logic)
    return kissat_reluctant_triggered (&solver->reluctant);
  } else {
    // Arm 1: Geometric/Exponential (Stable Search Logic)
    // Using the conflict limit threshold as the geometric trigger
    return (CONFLICTS >= solver->limits.restart.conflicts);
  }
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
    // Reset MAB tracking variables
    unsigned stable_restarts = 0;
    solver->mab_reward[solver->heuristic] += log2(solver->mab_decisions) / log2(solver->mab_conflicts);
    
    // Clear per-variable MAB data
    for (all_variables(idx)) {
        solver->mab_chosen[idx] = 0;
    }
    solver->mab_chosen_tot = 0;
    solver->mab_decisions = 0;
    solver->mab_conflicts = 0;
    
    // Count stable restarts across all heuristics
    for (unsigned i = 0; i < solver->mab_heuristics; i++) {
        stable_restarts += solver->mab_select[i];
    }

    // Track recent gains with momentum
    static double recent_gains[10] = {0};
    static int gain_index = 0;
    static double momentum = 1.0;

    double current_gain = solver->mab_reward[solver->heuristic] / solver->mab_select[solver->heuristic];
    recent_gains[gain_index] = current_gain;
    gain_index = (gain_index + 1) % 10;

    // Compute average gain over recent window
    double avg_gain = 0;
    for (int i = 0; i < 10; i++) {
        avg_gain += recent_gains[i];
    }
    avg_gain /= 10;

    // Update momentum based on performance
    if (current_gain > avg_gain) {
        momentum *= 1.1;
    } else {
        momentum *= 0.9;
    }

    // Compute adaptive exploration parameter
    double adaptive_c = solver->mabc / (momentum * (stable_restarts + 1));

    // Select next heuristic
    if (stable_restarts < solver->mab_heuristics) {
        // Exploration phase: alternate between first two heuristics
        solver->heuristic = solver->heuristic == 0 ? 1 : 0;
    } else {
        // UCB-based selection
        double ucb[2];
        solver->heuristic = 0;
        for (unsigned i = 0; i < solver->mab_heuristics; i++) {
            ucb[i] = solver->mab_reward[i] / solver->mab_select[i] 
                   + sqrt(adaptive_c * log(stable_restarts + 1) / solver->mab_select[i]);
            if (i != 0 && ucb[i] > ucb[solver->heuristic]) {
                solver->heuristic = i;
            }
        }
    }
    
    // Update selection count for chosen heuristic
    solver->mab_select[solver->heuristic]++;
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
