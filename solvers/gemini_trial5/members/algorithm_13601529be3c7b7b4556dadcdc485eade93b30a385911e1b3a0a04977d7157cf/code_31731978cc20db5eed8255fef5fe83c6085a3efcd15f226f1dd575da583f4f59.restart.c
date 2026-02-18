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
  // Step 1: Calculate reward R
  // R = (conflicts / max(1, sum_of_LBDs))
  // We need to track the sum of LBDs of learned clauses during the last interval.
  // We use the stable mode glue statistics (index 1) as MAB runs in stable mode.
  // We use static variables to track the previous values of the cumulative statistics.

  static uint64_t prev_glue_counts[128] = {0};
  uint64_t sum_lbd = 0;
  uint64_t glue_2_count = 0; // Clauses with LBD <= 2

  // Iterate over glue values (0 to 127)
  for (int i = 0; i < 128; i++) {
    uint64_t current = solver->statistics.used[1].glue[i];
    uint64_t delta = current - prev_glue_counts[i];
    prev_glue_counts[i] = current;

    // Sum of LBDs = sum(count * lbd_value)
    sum_lbd += delta * i;

    // Check for LBD <= 2 (indices 0, 1, 2)
    if (i <= 2) {
      glue_2_count += delta;
    }
  }

  double conflicts = (double) solver->mab_conflicts;
  double denominator = (sum_lbd < 1) ? 1.0 : (double) sum_lbd;
  double R = conflicts / denominator;

  // Step 2: Update EMAs for the currently active mode
  static double fast_ema[2] = {0};
  static double slow_ema[2] = {0};

  unsigned current_mode = solver->heuristic;

  // Ensure we don't buffer overflow if heuristic index is weird (typically 0 or 1)
  if (current_mode < 2) {
    // Fast_EMA (alpha=0.3)
    fast_ema[current_mode] = 0.3 * R + 0.7 * fast_ema[current_mode];
    // Slow_EMA (alpha=0.05)
    slow_ema[current_mode] = 0.05 * R + 0.95 * slow_ema[current_mode];
  }

  // Step 3: Update Stagnation Counter
  static int stagnation_counter = 0;
  if (glue_2_count == 0) {
    stagnation_counter++;
  } else {
    stagnation_counter--;
    if (stagnation_counter < 0)
      stagnation_counter = 0;
  }

  // Step 4: Calculate Priority P for each available mode
  double priority[2] = {0};
  unsigned num_heuristics = solver->mab_heuristics;
  if (num_heuristics > 2)
    num_heuristics = 2; // Clamp to static array size

  for (unsigned i = 0; i < num_heuristics; i++) {
    double P = slow_ema[i] + 1.5 * (fast_ema[i] - slow_ema[i]);

    // Step 5: If Stagnation_Counter > 6 and the mode is currently active, apply a penalty
    if (stagnation_counter > 6 && i == current_mode) {
      P = P * 0.5;
    }

    priority[i] = P;
  }

  // Step 6: Select the next restart mode corresponding to the highest Priority P
  unsigned best_heuristic = 0;
  double max_p = priority[0];

  for (unsigned i = 1; i < num_heuristics; i++) {
    if (priority[i] > max_p) {
      max_p = priority[i];
      best_heuristic = i;
    }
  }

  solver->heuristic = best_heuristic;

  // Update MAB stats (housekeeping compatible with Kissat infrastructure)
  solver->mab_select[solver->heuristic]++;

  // Reset interval counters
  for (all_variables (idx)) {
    solver->mab_chosen[idx] = 0;
  }
  solver->mab_chosen_tot = 0;
  solver->mab_decisions = 0;
  solver->mab_conflicts = 0;
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
