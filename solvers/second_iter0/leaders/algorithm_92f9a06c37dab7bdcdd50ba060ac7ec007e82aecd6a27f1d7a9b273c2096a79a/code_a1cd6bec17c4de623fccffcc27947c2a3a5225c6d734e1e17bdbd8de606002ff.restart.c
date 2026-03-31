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

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <inttypes.h>

#define VAMB_WINDOW 128
#define VAMB_THRESHOLD 0.02
#define VAMB_CONSECUTIVE 512
#define VAMB_EPSILON_RESTARTS 100
#define VAMB_DECAY_INTERVAL 32768
#define VAMB_DECAY_FACTOR 0.95
#define VAMB_LBD_CAP 64.0

void restart_mab(kissat *solver) {
    // Persistent state for the Multi-Armed Bandit
    static double rewards[2] = {0.0, 0.0};
    static uint64_t pulls[2] = {0, 0};
    static double D_win[VAMB_WINDOW];
    static double L_win[VAMB_WINDOW];
    static unsigned win_ptr = 0;
    static uint64_t last_conflicts = 0;
    static uint64_t last_decay_conf = 0;
    static uint64_t low_v_consecutive = 0;
    static int epsilon_timer = 0;
    static unsigned last_arm = 0;
    static bool init = false;

    // Current search statistics
    const uint64_t current_conflicts = CONFLICTS;
    const double cur_L = AVERAGE(level);
    const double cur_LBD = AVERAGE(fast_glue);
    // Approximate backjump distance D as (Level - LBD)
    const double cur_D = (cur_L > cur_LBD) ? (cur_L - cur_LBD) : 0.0;
    const double cur_V = (cur_L > 0.001) ? (cur_D / cur_L) : 0.0;

    // Initialization of window and counters
    if (!init) {
        last_conflicts = current_conflicts;
        last_decay_conf = current_conflicts;
        last_arm = solver->stable ? 1 : 0;
        for (int i = 0; i < VAMB_WINDOW; i++) {
            D_win[i] = cur_D;
            L_win[i] = cur_L;
        }
        init = true;
    }

    // Step 2: Update sliding window with current conflict metrics
    D_win[win_ptr] = cur_D;
    L_win[win_ptr] = cur_L;
    win_ptr = (win_ptr + 1) % VAMB_WINDOW;

    // Step 4: Calculate Reward for the arm that was active during the last flight
    // R = V * (1 - LBD_avg / 64)
    double lbd_influence = 1.0 - ((cur_LBD > VAMB_LBD_CAP ? VAMB_LBD_CAP : cur_LBD) / VAMB_LBD_CAP);
    double reward = cur_V * lbd_influence;
    rewards[last_arm] += reward;
    pulls[last_arm]++;

    // Step 3 & 5: Calculate Search Velocity (V) and its Standard Deviation (sigma)
    double sum_V = 0.0;
    double sum_V2 = 0.0;
    for (int i = 0; i < VAMB_WINDOW; i++) {
        double v_i = (L_win[i] > 0.001) ? (D_win[i] / L_win[i]) : 0.0;
        sum_V += v_i;
        sum_V2 += v_i * v_i;
    }
    double avg_V = sum_V / (double)VAMB_WINDOW;
    double var_V = (sum_V2 / (double)VAMB_WINDOW) - (avg_V * avg_V);
    if (var_V < 0) var_V = 0;
    double sigma_V = sqrt(var_V);
    
    // Non-static exploration constant C_t
    double C_t = 0.8 * sigma_V;

    // Step 6: Epsilon-Greedy Reset Trigger
    uint64_t delta_conf = (current_conflicts >= last_conflicts) ? (current_conflicts - last_conflicts) : 0;
    if (cur_V < VAMB_THRESHOLD) {
        low_v_consecutive += delta_conf;
    } else {
        low_v_consecutive = 0;
    }
    last_conflicts = current_conflicts;

    if (low_v_consecutive > VAMB_CONSECUTIVE) {
        epsilon_timer = VAMB_EPSILON_RESTARTS;
        rewards[0] = 0;
        rewards[1] = 0;
        pulls[0] = 0;
        pulls[1] = 0;
        low_v_consecutive = 0;
    }

    // Step 7: Temporal Decay (every 2^15 conflicts)
    if (current_conflicts - last_decay_conf >= VAMB_DECAY_INTERVAL) {
        rewards[0] *= VAMB_DECAY_FACTOR;
        rewards[1] *= VAMB_DECAY_FACTOR;
        last_decay_conf = current_conflicts;
    }

    // Arm Selection Logic
    unsigned next_arm;
    if (epsilon_timer > 0) {
        // Epsilon-Greedy exploration phase
        epsilon_timer--;
        next_arm = kissat_pick_bool(&solver->random) ? 1 : 0;
    } else if (pulls[0] == 0) {
        next_arm = 0;
    } else if (pulls[1] == 0) {
        next_arm = 1;
    } else {
        // UCB1 selection formula
        double total_pulls = (double)(pulls[0] + pulls[1]);
        double ucb0 = (rewards[0] / (double)pulls[0]) + C_t * sqrt(log(total_pulls) / (double)pulls[0]);
        double ucb1 = (rewards[1] / (double)pulls[1]) + C_t * sqrt(log(total_pulls) / (double)pulls[1]);
        next_arm = (ucb1 > ucb0) ? 1 : 0;
    }

    // Step 1: Apply selected strategy
    // Arm 0: Focused Search (Aggressive Luby)
    // Arm 1: Stable Search (Geometric/Exponential)
    last_arm = next_arm;
    solver->heuristic = next_arm;
    solver->stable = (next_arm == 1);
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
