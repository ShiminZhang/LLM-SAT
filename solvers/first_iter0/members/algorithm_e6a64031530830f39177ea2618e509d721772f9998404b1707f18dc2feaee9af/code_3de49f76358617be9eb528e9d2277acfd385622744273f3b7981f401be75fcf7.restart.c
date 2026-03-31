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
    // Basic Kissat gates for restart logic
    if (!GET_OPTION(restart))
        return false;
    if (!solver->level)
        return false;
    if (CONFLICTS < solver->limits.restart.conflicts)
        return false;

    // Define constants for Thompson Sampling Arms
    #define ARM_LUBY 0
    #define ARM_EMA 1
    #define ARM_FIXED 2
    #define WINDOW_LIMIT 128

    // Static variables to maintain state across calls for Stagnation-Aware MAB
    static unsigned S[3] = {0, 0, 0};
    static unsigned F[3] = {0, 0, 0};
    static uint8_t window[3][WINDOW_LIMIT];
    static unsigned win_ptr[3] = {0, 0, 0};
    static unsigned win_size[3] = {0, 0, 0};
    static int active_arm = -1;
    static double global_avg_R = 0.5;
    static uint64_t last_restart_conflicts = 0;
    static double max_level = 1.0;
    static bool initialized = false;

    // Initialize conflict tracker on first call
    if (!initialized) {
        last_restart_conflicts = CONFLICTS;
        initialized = true;
    }

    // Step 4: Calculate 'Search Stagnation' coefficient G
    if ((double)solver->level > max_level)
        max_level = (double)solver->level;
    
    double active_vars = (double)(solver->vars - solver->unassigned);
    double total_vars = (double)(solver->vars > 0 ? solver->vars : 1);
    double current_level = (double)solver->level;
    double G = (active_vars / total_vars) * (current_level / max_level);

    // Step 5 & 6: Thompson Sampling to select next arm if none is active
    if (active_arm == -1) {
        double theta[3];
        for (int i = 0; i < 3; i++) {
            // Draw sample theta_i from Beta(S_i + 1, F_i + 1)
            // Using the property that Gamma(a,1)/(Gamma(a,1)+Gamma(b,1)) ~ Beta(a,b)
            // and Gamma(n,1) is the sum of n Exponential(1) variables
            double alpha = (double)S[i] + 1.0;
            double beta_v = (double)F[i] + 1.0;
            double x = 0, y = 0;
            for (unsigned j = 0; j < (unsigned)alpha; j++)
                x -= log(kissat_pick_double(&solver->random) + 1e-9);
            for (unsigned j = 0; j < (unsigned)beta_v; j++)
                y -= log(kissat_pick_double(&solver->random) + 1e-9);
            theta[i] = x / (x + y);
        }

        // Apply non-linear Phase-Stagnation boosts
        double theta_prime[3];
        theta_prime[0] = theta[0];
        theta_prime[1] = theta[1];
        theta_prime[2] = theta[2];

        if (solver->stable)
            theta_prime[ARM_LUBY] *= exp(G);
        else
            theta_prime[ARM_EMA] *= exp(G);

        if (G > 0.5)
            theta_prime[ARM_FIXED] *= (1.0 + G * G);

        // Select the arm with the highest modified sample
        if (theta_prime[0] >= theta_prime[1] && theta_prime[0] >= theta_prime[2])
            active_arm = ARM_LUBY;
        else if (theta_prime[1] >= theta_prime[0] && theta_prime[1] >= theta_prime[2])
            active_arm = ARM_EMA;
        else
            active_arm = ARM_FIXED;
    }

    // Execute corresponding trigger logic for the selected arm
    bool triggered = false;
    if (active_arm == ARM_LUBY) {
        triggered = kissat_reluctant_triggered(&solver->reluctant);
    } else if (active_arm == ARM_EMA) {
        const double fast = AVERAGE(fast_glue);
        const double slow = AVERAGE(slow_glue);
        const double margin = (100.0 + (double)GET_OPTION(restartmargin)) / 100.0;
        const double limit = margin * slow;
        triggered = (limit <= fast);
    } else if (active_arm == ARM_FIXED) {
        // Aggressive fixed-interval trigger (50 conflicts)
        triggered = (CONFLICTS >= last_restart_conflicts + 50);
    }

    // Step 3: Update success/failure counts at the decision point
    if (triggered) {
        // Step 2: Define 'Search Efficiency' reward R
        // R = (log2(Backjump_Distance + 1) / Current_LBD)
        // Using solver->level as a proxy for backjump potential and fast_glue for LBD
        double current_R = (log(current_level + 1.0) * 1.44269504089) / (AVERAGE(fast_glue) + 1.0);
        bool success = (current_R > global_avg_R);

        // Update S, F using a sliding window of 128
        unsigned arm = (unsigned)active_arm;
        if (win_size[arm] == WINDOW_LIMIT) {
            if (window[arm][win_ptr[arm]]) S[arm]--;
            else F[arm]--;
        } else {
            win_size[arm]++;
        }

        window[arm][win_ptr[arm]] = (uint8_t)success;
        if (success) S[arm]++;
        else F[arm]++;
        win_ptr[arm] = (win_ptr[arm] + 1) % WINDOW_LIMIT;

        // Update global moving average of reward R
        global_avg_R = 0.99 * global_avg_R + 0.01 * current_R;

        // Reset arm and conflict tracker for next cycle
        active_arm = -1;
        last_restart_conflicts = CONFLICTS;
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
