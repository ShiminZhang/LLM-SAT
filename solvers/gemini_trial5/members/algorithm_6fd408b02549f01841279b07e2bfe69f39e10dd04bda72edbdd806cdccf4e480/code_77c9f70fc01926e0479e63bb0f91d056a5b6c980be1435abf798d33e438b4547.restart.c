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
    // Static state for Hysteresis-Damped Momentum Switching
    // We assume max 2 heuristics as per solver->mab_reward[2] definition
    static double fast_ema[2] = {0};
    static double slow_ema[2] = {0};
    static bool ema_initialized[2] = {false, false};
    static uint64_t last_ticks = 0;
    static unsigned tenure = 0;

    // Safety check
    if (solver->mab_heuristics < 2) return;

    // 1. Identify Incumbent (current) and Challenger (other)
    unsigned incumbent = solver->heuristic;
    unsigned challenger = (incumbent + 1) % solver->mab_heuristics;

    // 2. Calculate Yield 'Y' (log of conflicts per second)
    // We use solver->ticks as a proxy for time
    uint64_t current_ticks = solver->ticks;
    uint64_t delta_ticks = current_ticks - last_ticks;
    last_ticks = current_ticks;

    double conflicts = (double)solver->mab_conflicts;
    double duration = (double)delta_ticks;
    double yield = 0.0;

    // Avoid division by zero and handle low activity
    if (duration < 1.0) duration = 1.0;
    
    if (conflicts <= 0.0) {
        yield = -20.0; // Low yield penalty for no conflicts
    } else {
        yield = log(conflicts / duration); // Natural log
    }

    // 3. Update EMAs for the mode that just finished (Incumbent)
    if (!ema_initialized[incumbent]) {
        fast_ema[incumbent] = yield;
        slow_ema[incumbent] = yield;
        ema_initialized[incumbent] = true;
    } else {
        // Fast EMA (alpha=0.20)
        fast_ema[incumbent] = 0.20 * yield + 0.80 * fast_ema[incumbent];
        // Slow EMA (alpha=0.05)
        slow_ema[incumbent] = 0.05 * yield + 0.95 * slow_ema[incumbent];
    }

    // 4. Calculate Momentum and Projected Scores for both modes
    double momentum[2];
    double projected[2];

    for (unsigned i = 0; i < solver->mab_heuristics; i++) {
        // Momentum M = Fast - Slow
        momentum[i] = fast_ema[i] - slow_ema[i];
        // Projected Score P = Fast + 1.5 * M
        projected[i] = fast_ema[i] + (1.5 * momentum[i]);
    }

    // 5. Switching Logic
    bool switch_heuristic = false;
    double buffer = 0.05;

    // If Incumbent has negative Momentum, remove buffer
    if (momentum[incumbent] < 0.0) {
        buffer = 0.0;
    }

    // Enforce Minimum Tenure (3 consecutive intervals)
    if (tenure < 3) {
        tenure++;
    } else {
        // Switch if Challenger beats Incumbent + Buffer
        if (projected[challenger] > projected[incumbent] + buffer) {
            switch_heuristic = true;
        } else {
            tenure++;
        }
    }

    // 6. Standard Kissat MAB Housekeeping
    // Update cumulative rewards for the finished phase (Incumbent)
    if (solver->mab_conflicts > 0) {
        double reward = 0.0;
        if (solver->mab_decisions > 0.0) {
            reward = log2(solver->mab_decisions) / log2((double)solver->mab_conflicts);
        }
        solver->mab_reward[incumbent] += reward;
    }
    solver->mab_select[incumbent]++;

    // Apply Switch
    if (switch_heuristic) {
        solver->heuristic = challenger;
        tenure = 0;
    }

    // 7. Reset MAB Counters for next phase
    solver->mab_decisions = 0;
    solver->mab_conflicts = 0;
    solver->mab_chosen_tot = 0;
    for (all_variables(idx)) {
        solver->mab_chosen[idx] = 0;
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
