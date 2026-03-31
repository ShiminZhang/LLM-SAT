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
    // Static variables to maintain state across calls without hallucinating solver fields
    static double alpha[2] = {2.0, 2.0};
    static double beta[2] = {2.0, 2.0};
    static double r_short = 0.0;
    static double r_long = 0.0;
    static double global_avg_r = 0.0;
    static uint64_t count = 0;

    // Step 2: Define reward signal R = (1 / LBD_avg) * log10(backtrack_level + 1)
    // LBD_avg is taken from the fast EMA (window size usually ~33-128)
    double lbd_avg = AVERAGE(fast_glue);
    if (lbd_avg < 1.0) lbd_avg = 1.0;
    double backtrack_level = (double)solver->level;
    double reward = (1.0 / lbd_avg) * log10(backtrack_level + 1.0);

    // Update parameters for the arm (policy) that was active during the last search interval
    // Arms: 0 = Luby, 1 = Glucose
    unsigned last_arm = solver->heuristic;
    if (count > 0) {
        // Update chosen arm's alpha if R > global_average_R, otherwise updating beta
        if (reward > global_avg_r)
            alpha[last_arm] += 1.0;
        else
            beta[last_arm] += 1.0;
    }

    // Step 3: Maintain 'Search Volatility' index (V)
    // Calculated as the relative difference between short-term (128) and long-term (4096) EMA
    // alpha = 2 / (N + 1)
    if (count == 0) {
        r_short = reward;
        r_long = reward;
        global_avg_r = reward;
    } else {
        r_short = (0.0155 * reward) + ((1.0 - 0.0155) * r_short);   // N=128
        r_long = (0.000488 * reward) + ((1.0 - 0.000488) * r_long); // N=4096
        // Update global average reward
        global_avg_r = (global_avg_r * (double)count + reward) / (double)(count + 1);
    }
    count++;

    double volatility = (r_long > 1e-9) ? fabs(r_short - r_long) / r_long : 0.0;

    // Step 4: Knowledge Decay if V > 0.3 (indicates phase shift)
    if (volatility > 0.3) {
        alpha[0] *= 0.6;
        beta[0] *= 0.6;
        alpha[1] *= 0.6;
        beta[1] *= 0.6;
    }

    // Step 5: Thompson Sampling for the next restart policy
    // Sample from Beta distribution for each arm using approximation: X = U^(1/a) / (U^(1/a) + V^(1/b))
    double s[2];
    for (unsigned i = 0; i < 2; i++) {
        double u = kissat_pick_double(&solver->random);
        double v = kissat_pick_double(&solver->random);
        // Ensure values are non-zero for pow()
        if (u < 1e-12) u = 1e-12;
        if (v < 1e-12) v = 1e-12;
        
        double x = pow(u, 1.0 / alpha[i]);
        double y = pow(v, 1.0 / beta[i]);
        s[i] = x / (x + y);
    }

    // Select policy with the highest sampled value
    solver->heuristic = (s[1] > s[0]) ? 1 : 0;

    // Reset Kissat's internal MAB tracking to keep statistics consistent
    solver->mab_decisions = 0;
    solver->mab_conflicts = 0;
    for (all_variables(idx)) {
        solver->mab_chosen[idx] = 0;
    }
    solver->mab_chosen_tot = 0;
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
