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

  // Step 1: Initialize restart strategy arms and sliding window state
  static unsigned S[3] = {1, 1, 1}; // Success counts for Arms 0, 1, 2
  static unsigned F[3] = {1, 1, 1}; // Failure counts for Arms 0, 1, 2
  static unsigned arms_hist[128];   // Sliding window: Arm used
  static bool success_hist[128];    // Sliding window: Outcome
  static unsigned win_ptr = 0;      // Current window index
  static uint64_t last_restarts = 0;
  static uint64_t last_restart_conflicts = 0;
  static unsigned active_arm = 0;   // Currently selected Arm (0=Luby, 1=EMA, 2=Fixed)
  
  static double global_R_avg = 0.5;
  static uint64_t global_R_count = 0;
  static double interval_R_sum = 0;
  static uint64_t interval_R_count = 0;

  // Step 2: Define and accumulate Search Efficiency reward R for each conflict
  // kissat_restarting is called after each conflict in the search loop.
  double cur_lbd = AVERAGE (fast_glue);
  if (cur_lbd < 1.0) cur_lbd = 1.0;
  // Reward R = (log2(Backjump_Distance + 1) / Current_LBD)
  // We use current level as a proxy for the potential backjump distance.
  double R = (log ((double)solver->level + 1.0) * 1.44269504089) / cur_lbd;
  interval_R_sum += R;
  interval_R_count++;

  // Step 3 & 5: Update MAB at the decision point for a restart
  // We detect a new restart interval by checking the solver's restart counter.
  if (solver->statistics.restarts > last_restarts) {
    double avg_R = (interval_R_count > 0) ? (interval_R_sum / interval_R_count) : 0;
    bool success = (avg_R > global_R_avg);

    // Update Success/Failure counts using sliding window of 128
    if (global_R_count >= 128) {
      unsigned prev_arm = arms_hist[win_ptr];
      bool prev_success = success_hist[win_ptr];
      if (prev_success) { if (S[prev_arm] > 1) S[prev_arm]--; }
      else { if (F[prev_arm] > 1) F[prev_arm]--; }
    }

    arms_hist[win_ptr] = active_arm;
    success_hist[win_ptr] = success;
    if (success) S[active_arm]++; else F[active_arm]++;
    win_ptr = (win_ptr + 1) % 128;

    // Update global moving average of R (simple EMA)
    if (global_R_count == 0) global_R_avg = avg_R;
    else global_R_avg = (0.9 * global_R_avg) + (0.1 * avg_R);
    global_R_count++;

    // Step 4: Calculate Search Stagnation coefficient G
    double G = 0;
    if (solver->vars > 0 && solver->statistics.max_level > 0) {
      G = ((double)solver->active / (double)solver->vars) * 
          ((double)solver->level / (double)solver->statistics.max_level);
    }

    // Step 5: Thompson Sampling from Beta(S+1, F+1)
    double samples[3];
    for (unsigned i = 0; i < 3; i++) {
      double x = 0, y = 0;
      // Sampling Gamma(S+1, 1) and Gamma(F+1, 1) to get Beta sample
      for (unsigned j = 0; j < S[i] + 1; j++)
        x -= log (1.0 - kissat_pick_double (&solver->random));
      for (unsigned j = 0; j < F[i] + 1; j++)
        y -= log (1.0 - kissat_pick_double (&solver->random));
      samples[i] = (x + y > 0) ? (x / (x + y)) : 0;
    }

    // Apply Phase Bias
    if (solver->stable) samples[0] *= (1.0 + G); // Favor Luby in Stable
    else samples[1] *= (1.0 + G);               // Favor Glucose in Focused

    // Step 6: Select arm with highest modified sample
    if (samples[0] >= samples[1] && samples[0] >= samples[2]) active_arm = 0;
    else if (samples[1] >= samples[0] && samples[1] >= samples[2]) active_arm = 1;
    else active_arm = 2;

    // Reset interval tracking
    interval_R_sum = 0;
    interval_R_count = 0;
    last_restarts = solver->statistics.restarts;
    last_restart_conflicts = CONFLICTS;
  }

  // Step 6: Execute corresponding restart trigger logic
  if (active_arm == 0) {
    // Arm 0: Luby sequence
    return kissat_reluctant_triggered (&solver->reluctant);
  } else if (active_arm == 1) {
    // Arm 1: Glucose-style EMA
    const double fast = AVERAGE (fast_glue);
    const double slow = AVERAGE (slow_glue);
    const double margin = (100.0 + GET_OPTION (restartmargin)) / 100.0;
    const double limit = margin * slow;
    return (limit <= fast);
  } else {
    // Arm 2: Aggressive Fixed-Interval (e.g., every 50 conflicts)
    return (CONFLICTS >= last_restart_conflicts + 50);
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
