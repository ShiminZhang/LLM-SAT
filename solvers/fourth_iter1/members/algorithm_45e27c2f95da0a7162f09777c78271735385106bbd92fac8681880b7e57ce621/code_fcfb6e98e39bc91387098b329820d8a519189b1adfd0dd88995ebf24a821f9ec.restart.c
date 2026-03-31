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

void restart_mab (kissat *solver) {
  // Persistence of Beta parameters and search history
  static double alpha[2] = {2.0, 2.0};
  static double beta[2] = {2.0, 2.0};
  static double r_history[4096] = {0};
  static uint64_t r_count = 0;
  static double g_reward_sum = 0;
  static uint64_t g_reward_count = 0;
  static int probing_window = 0;

  // Step 2: Calculate reward R for the arm that was active during the last search segment
  // R = (1 / LBD_avg) * log10(backtrack_level + 1)
  double cur_lbd = AVERAGE (fast_glue);
  if (cur_lbd < 1.0) cur_lbd = 1.0;
  
  // Backtrack level is retrieved from the solver state at the point of restart
  double level_val = (double) solver->level;
  double r = (1.0 / cur_lbd) * (log (level_val + 1.0) / 2.302585092994); // log10 approximation

  // Identify the arm that was active (Arm 0 = Luby/reluctant, Arm 1 = Glucose/EMA)
  unsigned last_arm = (solver->heuristic == 1) ? 1 : 0;

  // Update global average reward for parameter reinforcement
  g_reward_sum += r;
  g_reward_count++;
  double global_avg = g_reward_sum / g_reward_count;

  // Step 5 Update logic: Update chosen arm's alpha if R > global_average_R, else beta
  if (r > global_avg)
    alpha[last_arm] += 1.0;
  else
    beta[last_arm] += 1.0;

  // Step 3: Maintain reward history and calculate Search Volatility index (V)
  r_history[r_count % 4096] = r;
  r_count++;

  // Calculate long-term average (up to 4096 restarts)
  uint64_t n4096 = (r_count > 4096) ? 4096 : r_count;
  double sum4096 = 0;
  for (uint64_t i = 0; i < n4096; i++) {
    sum4096 += r_history[i];
  }
  double avg4096 = sum4096 / n4096;

  // Calculate short-term average (up to 128 restarts)
  uint64_t n128 = (r_count > 128) ? 128 : r_count;
  double sum128 = 0;
  for (uint64_t i = 0; i < n128; i++) {
    uint64_t idx = (r_count > 0) ? (r_count - 1 - i) % 4096 : 0;
    sum128 += r_history[idx];
  }
  double avg128 = sum128 / n128;

  // V = relative difference between short-term and long-term reward averages
  double v = (avg4096 > 1e-9) ? fabs (avg128 - avg4096) / avg4096 : 0;

  // Step 4: Volatility trigger
  if (v > 0.25) {
    // Non-linear decay to discount stale history
    double decay_factor = exp (-2.0 * v);
    alpha[0] *= decay_factor;
    beta[0] *= decay_factor;
    alpha[1] *= decay_factor;
    beta[1] *= decay_factor;

    // Initiate probing window
    probing_window = (int) (10.0 * v);
    if (probing_window < 1) probing_window = 1;
  }

  // Step 5: Arm selection for the next search segment
  unsigned chosen;
  if (probing_window > 0) {
    probing_window--;
    // Epsilon-greedy selection (epsilon=0.5)
    if (kissat_pick_double (&solver->random) < 0.5) {
      chosen = kissat_pick_random (&solver->random, 0, 2);
    } else {
      // Pick best current arm based on mean of Beta distribution
      double m0 = alpha[0] / (alpha[0] + beta[0]);
      double m1 = alpha[1] / (alpha[1] + beta[1]);
      chosen = (m0 > m1) ? 0 : 1;
    }
  } else {
    // Thompson Sampling using Normal approximation for the Beta distributions
    double samples[2];
    for (unsigned i = 0; i < 2; i++) {
      double u1 = kissat_pick_double (&solver->random);
      double u2 = kissat_pick_double (&solver->random);
      // Box-Muller transform for N(0,1)
      double z = sqrt (-2.0 * log (u1 + 1e-9)) * cos (2.0 * 3.141592653589 * u2);
      
      double a = alpha[i];
      double b = beta[i];
      double mu = a / (a + b);
      // Variance of Beta distribution: (a*b) / ((a+b)^2 * (a+b+1))
      double var = (a * b) / ((a + b) * (a + b) * (a + b + 1.0));
      samples[i] = mu + z * sqrt (var);
    }
    chosen = (samples[0] > samples[1]) ? 0 : 1;
  }

  // Assign the selected policy arm to the solver's heuristic field
  solver->heuristic = chosen;
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
