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

bool kissat_restarting(kissat *solver) {
    assert(solver->unassigned);

    // 0. Basic Option Checks
    if (!GET_OPTION(restart))
        return false;
    if (!solver->level)
        return false;

    // Stable mode delegation (Reluctant Doubling)
    if (solver->stable)
        return kissat_reluctant_triggered(&solver->reluctant);

    // --- Throttled Turbulent Restart Strategy Implementation ---

    // Static State Maintenance
    // Note: These preserve state across calls to implement the EMA and counters.
    static double lbd_var_ema = 0.0;
    static double last_fast_glue = 0.0;
    static uint64_t last_check_conflicts = 0;
    static uint64_t conflict_at_last_restart = 0;
    static uint64_t stored_restarts = 0;
    static int laminar_skips = 0;

    // Detect if a restart happened since the last call (to sync state)
    if (solver->statistics.restarts > stored_restarts) {
        stored_restarts = solver->statistics.restarts;
        conflict_at_last_restart = CONFLICTS; 
        laminar_skips = 0;
        // Re-initialize tracking to prevent spikes from restart boundaries
        last_fast_glue = AVERAGE(fast_glue);
        last_check_conflicts = CONFLICTS;
    }

    const double fast = AVERAGE(fast_glue);
    const double slow = AVERAGE(slow_glue);

    // Step 1: Update LBD Variance EMA
    // We approximate the "current LBD" by reconstructing the average LBD 
    // of the batch of conflicts since the last check, using the change in fast_glue.
    if (CONFLICTS > last_check_conflicts) {
        uint64_t n = CONFLICTS - last_check_conflicts;
        double alpha = EMA(fast_glue).alpha;
        
        // Safety for alpha
        if (alpha <= 0.0) alpha = 1.0 / (double)GET_OPTION(emafast);

        // Reconstruct batch LBD: fast_new = fast_old + n * alpha * (lbd_batch - fast_old)
        // Implies: lbd_batch = fast_old + (fast_new - fast_old) / (n * alpha)
        double diff_fast = fast - last_fast_glue;
        double estimated_batch_lbd = last_fast_glue + diff_fast / (n * alpha);
        
        // Algorithm: diff = current_lbd - fast_lbd_ema
        double diff = estimated_batch_lbd - fast;
        
        // Update variance EMA: 0.95 * old + 0.05 * diff^2
        lbd_var_ema = 0.95 * lbd_var_ema + 0.05 * (diff * diff);
        
        // Update history
        last_fast_glue = fast;
        last_check_conflicts = CONFLICTS;
    } else if (last_fast_glue == 0.0) {
        // First run initialization
        last_fast_glue = fast;
        last_check_conflicts = CONFLICTS;
    }

    // Step 2: Define 'Laminar Search'
    bool laminar = (fast < 0.75 * slow) && (lbd_var_ema < 1.0);

    // Step 3: Define 'Turbulent Search'
    bool turbulent = (fast > 1.1 * slow) && (lbd_var_ema > 3.5);

    // Determine Standard Restart Condition
    // Standard Kissat logic: Check conflict limit, then check glue limit.
    bool standard_restart = false;
    if (CONFLICTS >= solver->limits.restart.conflicts) {
        const double margin = (100.0 + GET_OPTION(restartmargin)) / 100.0;
        const double limit = margin * slow;
        
        kissat_extremely_verbose(solver,
            "restart glue limit %g = %.02f * %g (slow) %c %g (fast)",
            limit, margin, slow,
            (limit > fast ? '>' : limit == fast ? '=' : '<'), fast);
            
        if (limit <= fast) {
            standard_restart = true;
        }
    }

    // Step 4: Laminar Override (Throttle)
    // If standard condition says YES, but we are Laminar, skip restart.
    if (standard_restart) {
        if (laminar) {
            if (laminar_skips < 5) {
                laminar_skips++;
                kissat_extremely_verbose(solver, "Laminar override: skipping restart (%d/5)", laminar_skips);
                return false;
            }
        }
        // Reset skips if we proceed to restart (or if not laminar anymore)
        laminar_skips = 0;
        return true;
    }

    // Step 5: Turbulent Override (Escape)
    // If standard condition says NO, but we are Turbulent, force restart (with safeguard).
    if (!standard_restart && turbulent) {
        // Safeguard: Ensure conflicts since last restart >= 50% of current interval.
        // The current interval target is solver->limits.restart.conflicts.
        uint64_t target = solver->limits.restart.conflicts;
        uint64_t start = conflict_at_last_restart;
        
        // Calculate interval duration (fallback to restartint if wrap/init issues)
        uint64_t interval = (target > start) ? (target - start) : GET_OPTION(restartint);
        uint64_t elapsed = CONFLICTS - start;

        if (elapsed >= 0.5 * interval) {
            kissat_extremely_verbose(solver, "Turbulent override: forcing restart");
            return true;
        }
    }

    return false;
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
