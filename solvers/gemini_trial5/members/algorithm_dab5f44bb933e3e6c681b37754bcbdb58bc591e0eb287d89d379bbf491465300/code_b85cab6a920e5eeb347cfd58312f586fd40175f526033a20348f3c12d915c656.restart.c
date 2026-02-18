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

     // Housekeeping: Reset MAB interval counters for the next period
     // This ensures that the solver tracks decisions/conflicts correctly for the next interval
     for (all_variables(idx)) {
         solver->mab_chosen[idx] = 0;
     }
     solver->mab_chosen_tot = 0;
     solver->mab_decisions = 0;
     solver->mab_conflicts = 0;

     // Algorithm: Hysteretic Derivative-Driven MAB

     // Step 1: Calculate the raw performance metric 'P'
     // P = 10000 / (average_LBD * sqrt(average_trail_level) + 1)
     // Using fast_glue for average_LBD as it represents recent performance
     double average_LBD = AVERAGE(fast_glue);
     double average_trail_level = AVERAGE(level);
     
     // Ensure argument for sqrt is non-negative
     if (average_trail_level < 0) average_trail_level = 0;

     double P = 10000.0 / (average_LBD * sqrt(average_trail_level) + 1.0);

     // Step 2: Update the exponential moving average Score 'S'
     // S_new = 0.8 * S_old + 0.2 * P
     unsigned active_mode = solver->heuristic;
     double S_old = solver->mab_reward[active_mode];
     double S_new = 0.8 * S_old + 0.2 * P;
     
     solver->mab_reward[active_mode] = S_new;

     // Step 3: Calculate the discrete derivative 'Delta'
     double Delta = S_new - S_old;

     // Step 4: Execute Mode Selection using Hysteretic Logic
     // Maintain a persistent counter 'decline_streak' for the active mode
     // Using static array to persist state between calls (0=VSIDS, 1=CHB)
     static int decline_streak[2] = {0, 0};

     if (Delta >= 0) {
         // Reset 'decline_streak' to 0 and deterministically keep the current mode
         decline_streak[active_mode] = 0;
         // Keep current mode (implicit)
     } else {
         // Delta < 0: Increment 'decline_streak'
         decline_streak[active_mode]++;
         
         // Switch modes ONLY IF ('decline_streak' >= 2) OR (Delta < -0.15 * S_new)
         // Note: Delta is negative here, so we check if it is below the negative threshold
         bool massive_drop = (Delta < (-0.15 * S_new));
         
         if (decline_streak[active_mode] >= 2 || massive_drop) {
             // Switch heuristic (0 -> 1 or 1 -> 0)
             solver->heuristic = 1 - active_mode;
         }
     }

     // Update selection count for the chosen heuristic
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
