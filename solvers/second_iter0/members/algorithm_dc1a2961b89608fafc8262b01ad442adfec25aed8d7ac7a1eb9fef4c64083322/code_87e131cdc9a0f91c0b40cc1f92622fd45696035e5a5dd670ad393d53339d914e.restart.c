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
  #define RSVB_WIN 128
  #define RSVB_LOW_V 0.02
  #define RSVB_LOW_V_LIMIT 512
  #define RSVB_RESET_PERIOD 100

  // Static bandit state persistent across calls
  static double d_win[RSVB_WIN];
  static double l_win[RSVB_WIN];
  static double v_win[RSVB_WIN];
  static unsigned w_ptr = 0;
  static bool w_full = false;

  static uint64_t l_conf = 0;
  static uint64_t l_red = 0;
  static unsigned peak_lev = 0;

  static double rew[2] = {0, 0};
  static uint64_t pulls[2] = {0, 0};
  static unsigned active_arm = 0;
  static uint64_t low_v_cnt = 0;
  static int reset_timer = 0;

  // Track the highest decision level reached to estimate conflict level (L)
  if (solver->level > peak_lev)
    peak_lev = solver->level;

  // Step 2 & 3: Maintain sliding window and calculate Search Velocity (V)
  if (CONFLICTS > l_conf) {
    double D = (peak_lev > solver->level) ? (double) (peak_lev - solver->level) : 0.0;
    double L = (peak_lev > 0) ? (double) peak_lev : 1.0;

    d_win[w_ptr] = D;
    l_win[w_ptr] = L;
    v_win[w_ptr] = D / L;
    w_ptr = (w_ptr + 1) % RSVB_WIN;
    if (w_ptr == 0)
      w_full = true;

    unsigned count = w_full ? RSVB_WIN : w_ptr;
    double sum_d = 0, sum_l = 0, sum_v = 0, sum_v2 = 0;
    for (unsigned i = 0; i < count; i++) {
      sum_d += d_win[i];
      sum_l += l_win[i];
      sum_v += v_win[i];
      sum_v2 += v_win[i] * v_win[i];
    }

    double v_avg = (sum_l > 0) ? (sum_d / sum_l) : 0;
    double avg_v = (count > 0) ? (sum_v / count) : 0;
    double var_v = (count > 0) ? (sum_v2 / count - avg_v * avg_v) : 0;
    double sigma_v = sqrt (var_v > 0 ? var_v : 0);

    // Step 4: Calculate Reward (R) for the active arm
    const double slow_lbd = AVERAGE (slow_glue);
    const double capped_lbd = (slow_lbd > 64.0) ? 64.0 : slow_lbd;
    double reward = v_avg * (1.0 - capped_lbd / 64.0);
    rew[active_arm] += reward;
    pulls[active_arm]++;

    // Step 6: Epsilon-Greedy Reset logic
    if (v_avg < RSVB_LOW_V)
      low_v_cnt++;
    else
      low_v_cnt = 0;

    if (low_v_cnt > RSVB_LOW_V_LIMIT) {
      reset_timer = RSVB_RESET_PERIOD;
      rew[0] = rew[1] = 0;
      pulls[0] = pulls[1] = 0;
      low_v_cnt = 0;
    }

    // Step 7: Synchronize reward decay with clause reduction
    if (solver->statistics.clauses_reduced > l_red) {
      double gamma = 0.95 - (0.1 * sigma_v);
      if (gamma < 0)
        gamma = 0;
      rew[0] *= gamma;
      rew[1] *= gamma;
      l_red = solver->statistics.clauses_reduced;
    }

    // Step 5: Update Arm Selection using UCB1
    if (reset_timer > 0) {
      // Force exploration: alternate arms during reset period
      active_arm = (unsigned) ((pulls[0] + pulls[1]) % 2);
    } else {
      if (pulls[0] == 0)
        active_arm = 0;
      else if (pulls[1] == 0)
        active_arm = 1;
      else {
        double total_pulls = (double) (pulls[0] + pulls[1]);
        double c_t = 0.8 * sigma_v;
        double ucb0 = (rew[0] / pulls[0]) + c_t * sqrt (log (total_pulls) / pulls[0]);
        double ucb1 = (rew[1] / pulls[1]) + c_t * sqrt (log (total_pulls) / pulls[1]);
        active_arm = (ucb1 > ucb0) ? 1 : 0;
      }
    }

    l_conf = CONFLICTS;
    peak_lev = solver->level;
  }

  // Baseline Kissat logic checks
  assert (solver->unassigned);
  if (!GET_OPTION (restart))
    return false;
  if (!solver->level)
    return false;

  // Step 1: Implement Arm logic
  bool trigger = false;
  if (active_arm == 0) {
    // Arm 0: Aggressive Focused Search (Glue-based)
    if (CONFLICTS < solver->limits.restart.conflicts)
      return false;
    const double fast = AVERAGE (fast_glue);
    const double slow = AVERAGE (slow_glue);
    const double margin = (100.0 + GET_OPTION (restartmargin)) / 100.0;
    trigger = (margin * slow <= fast);
  } else {
    // Arm 1: Stable Search (Geometric/Exponential Reluctant)
    trigger = kissat_reluctant_triggered (&solver->reluctant);
  }

  if (trigger && reset_timer > 0)
    reset_timer--;

  return trigger;
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
