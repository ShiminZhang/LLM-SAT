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
    // Parameters defined by the algorithm
    const double alpha = 0.15;      // Learning rate for Q
    const double beta = 2.0;        // Trend gain
    const double decay = 0.95;      // Fatigue decay factor
    const double r_mix = 0.3;       // Baseline update mix (0.7 old, 0.3 new)

    // Static state to maintain history across restarts.
    // Note: We use static variables because we cannot modify the solver struct.
    // This assumes a single solver instance per process or sequential execution.
    static double R_avg[2] = {0.0, 0.0};
    static double Q[2] = {0.0, 0.0};
    static int consecutive_count = 0;
    static uint64_t last_ticks = 0;
    static bool initialized = false;

    // --- Step 1: Compute Raw Reward R ---
    
    unsigned conflicts = solver->mab_conflicts;
    uint64_t current_ticks = solver->ticks;
    uint64_t elapsed_ticks = 0;

    if (initialized) {
        if (current_ticks >= last_ticks) {
            elapsed_ticks = current_ticks - last_ticks;
        }
    } else {
        // First execution: assume minimal elapsed time to avoid skewing or 0
        elapsed_ticks = 0;
    }

    // R = log2(conflicts_generated + 1) / (elapsed_ticks + 1)
    double R = log2((double)conflicts + 1.0) / ((double)elapsed_ticks + 1.0);

    unsigned current_mode = solver->heuristic;
    // Safety clamp for static array access (Kissat typically has 2 heuristics)
    if (current_mode > 1) current_mode = 0;

    // Initialize baseline on first execution
    if (!initialized) {
        R_avg[current_mode] = R;
        // Note: Q implies starting at 0 or near R. Leaving at 0 allows the update rule to take over.
        last_ticks = current_ticks;
        initialized = true;
    }

    // --- Step 2: Calculate Performance Trend T ---

    // T = R - R_avg (using stored average for this mode)
    double T = R - R_avg[current_mode];

    // Immediately update the baseline: R_avg = 0.7 * R_avg + 0.3 * R
    R_avg[current_mode] = (1.0 - r_mix) * R_avg[current_mode] + r_mix * R;

    // --- Step 3: Update Q-value ---

    // Q_new = 0.85 * Q_old + 0.15 * (R + 2.0 * T)
    // Note: 0.85 is (1.0 - alpha) where alpha is 0.15
    double Q_old = Q[current_mode];
    double Q_new = (1.0 - alpha) * Q_old + alpha * (R + beta * T);
    Q[current_mode] = Q_new;

    // --- Step 4: Calculate Selection Score S ---

    unsigned best_mode = current_mode;
    double best_S = -1e100; // Start with a very low number

    // Iterate through available heuristics (typically 2: VSIDS and CHB)
    unsigned n_heuristics = solver->mab_heuristics;
    if (n_heuristics < 1) n_heuristics = 1;
    if (n_heuristics > 2) n_heuristics = 2; // Clamp to static array size

    for (unsigned i = 0; i < n_heuristics; i++) {
        double S;
        if (i == current_mode) {
            // For the mode just finished: S = Q_new * (0.95 ^ consecutive_count)
            S = Q[i] * pow(decay, (double)consecutive_count);
        } else {
            // For the inactive mode: S = Q_current
            S = Q[i];
        }

        if (S > best_S) {
            best_S = S;
            best_mode = i;
        }
    }

    // --- Step 5: Select Mode ---

    if (best_mode != current_mode) {
        consecutive_count = 0;
    } else {
        consecutive_count++;
    }

    solver->heuristic = best_mode;

    // --- Housekeeping (Standard Kissat MAB Reset) ---

    // Update global tick counter for next delta calculation
    last_ticks = current_ticks;

    // Reset per-variable chosen counts and MAB stats
    for (all_variables(idx)) {
        solver->mab_chosen[idx] = 0;
    }
    solver->mab_chosen_tot = 0;
    solver->mab_decisions = 0;
    solver->mab_conflicts = 0;

    // Increment selection count for the chosen heuristic
    if (best_mode < 2) {
        solver->mab_select[best_mode]++;
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
