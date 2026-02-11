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

  // Stable mode: Use Reluctant Doubling Strategy
  // The provided algorithm ("Globally Anchored Glue Throttling") relies on glue (LBD)
  // trends which are primarily relevant for the 'focused' search mode.
  // In stable mode, Kissat relies on reluctant doubling of conflict intervals.
  if (solver->stable)
    return kissat_reluctant_triggered (&solver->reluctant);

  // Focused mode: Globally Anchored Glue Throttling implementation

  // Step 1: Retrieve averages
  const double fast_glue = AVERAGE (fast_glue);
  const double slow_glue = AVERAGE (slow_glue);

  // Calculate global arithmetic mean of LBDs ('global_glue')
  // We compute this by iterating the glue distribution histogram.
  // The histogram tracks LBDs from 0 to 127 for both modes.
  uint64_t total_glue_sum = 0;
  uint64_t total_count = 0;

  for (int mode = 0; mode < 2; mode++) {
    for (int g = 0; g < 128; g++) {
      uint64_t count = solver->statistics.used[mode].glue[g];
      total_glue_sum += count * g;
      total_count += count;
    }
  }

  double global_glue = slow_glue; // Fallback to slow_glue if no stats yet
  if (total_count > 0)
    global_glue = (double) total_glue_sum / total_count;

  // Compute 'hybrid_slow' baseline
  const double hybrid_slow = (slow_glue + global_glue) / 2.0;

  // Step 2: Calculate the trend ratio R
  double R = 1.0;
  if (hybrid_slow > 1e-9) // Prevent division by zero
    R = fast_glue / hybrid_slow;

  // Step 3: Define a scaling factor S
  double S = 1.0;
  if (R > 1.10) {
    // Recent search quality degrading (fast glue > hybrid baseline)
    // Encourage earlier restart
    S = 0.75;
  } else if (R < 0.90) {
    // Search finding high-quality conflicts (fast glue < hybrid baseline)
    // Delay restart to exploit current branch
    S = 1.35;
  }

  // Step 4: Retrieve the current scheduled restart limit
  // This limit is an absolute conflict count target.
  const uint64_t limit = solver->limits.restart.conflicts;

  // Step 5: Return true if the current conflict count exceeds (limit * S)
  // Dynamically stretching or shrinking the restart interval based on live quality.
  const uint64_t threshold = (uint64_t) (limit * S);

  kissat_extremely_verbose (solver,
                            "restart check: fast=%.2f slow=%.2f global=%.2f "
                            "hybrid=%.2f R=%.2f S=%.2f limit=%" PRIu64
                            " threshold=%" PRIu64 " conflicts=%" PRIu64,
                            fast_glue, slow_glue, global_glue, hybrid_slow,
                            R, S, limit, threshold, CONFLICTS);

  return (CONFLICTS > threshold);
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
