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
    assert(solver->unassigned);
    if (!GET_OPTION(restart) || !solver->level) return false;
    if (CONFLICTS < solver->limits.restart.conflicts) return false;

    // Step 1: Maintain EMAs
    static double fast_ema_stable = 0.0;
    static double slow_ema_stable = 0.0;
    static double fast_ema_focused = 0.0;
    static double slow_ema_focused = 0.0;

    double alpha_fast = kissat_pick_double(&solver->random); // Dynamic alpha for fast EMA
    double alpha_slow = 0.05;

    // Step 2: Calculate reward R
    double R = (solver->statistics.conflicts) / (double)max(1, solver->sum_of_LBDs);

    // Step 3: Update EMAs
    if (solver->stable) {
        fast_ema_stable = alpha_fast * R + (1 - alpha_fast) * fast_ema_stable;
        slow_ema_stable = alpha_slow * R + (1 - alpha_slow) * slow_ema_stable;
    } else {
        fast_ema_focused = alpha_fast * R + (1 - alpha_fast) * fast_ema_focused;
        slow_ema_focused = alpha_slow * R + (1 - alpha_slow) * slow_ema_focused;
    }

    // Step 4: Track global Stagnation_Counter
    static unsigned stagnation_counter = 0;
    if (solver->new_glue_clauses_learned == 0) {
        stagnation_counter++;
    } else {
        stagnation_counter = 0;
    }

    // Step 5: Calculate Priority P
    double P_stable = slow_ema_stable + 1.5 * (fast_ema_stable - slow_ema_stable);
    double P_focused = slow_ema_focused + 1.5 * (fast_ema_focused - slow_ema_focused);

    // Step 6: Apply penalty if stagnation_counter exceeds a threshold
    unsigned dynamic_threshold = 5; // Example threshold
    if (stagnation_counter > dynamic_threshold) {
        P_stable *= 0.5;
        P_focused *= 0.5;
    }

    // Step 7: Select next restart mode based on highest Priority P
    if (P_stable > P_focused) {
        solver->stable = true;
    } else {
        solver->stable = false;
    }

    // Step 8: Switch mode if negative momentum detected
    static unsigned negative_momentum_counter = 0;
    if (solver->stable && negative_momentum_counter > 5) {
        solver->stable = false;
        negative_momentum_counter = 0; // Reset counter
    } else if (!solver->stable && negative_momentum_counter > 5) {
        solver->stable = true;
        negative_momentum_counter = 0; // Reset counter
    }

    return true; // Restart triggered
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
