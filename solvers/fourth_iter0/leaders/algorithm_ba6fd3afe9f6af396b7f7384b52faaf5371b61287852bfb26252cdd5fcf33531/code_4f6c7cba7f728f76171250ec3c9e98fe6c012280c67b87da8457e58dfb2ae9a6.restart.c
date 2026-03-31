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
    // Persistent state for Beta distributions and Reward averages
    static double alphas[2] = {2.0, 2.0};
    static double betas[2] = {2.0, 2.0};
    static double r_short = 0.0;
    static double r_long = 0.0;
    static double r_global = 0.0;
    static uint64_t total_restarts = 0;

    // Step 2: Calculate reward R for the search segment just finished.
    // LBD_avg is accessed via the fast_glue EMA macro.
    double lbd_avg = AVERAGE(fast_glue);
    if (lbd_avg < 1.0) lbd_avg = 1.0;
    
    // Reward R = (1 / LBD_avg) * log10(backtrack_level + 1)
    // solver->level is the decision level at the point of restart.
    double R = (1.0 / lbd_avg) * log10((double)solver->level + 1.0);

    // Step 3: Update Moving Averages and Arm Parameters
    if (total_restarts == 0) {
        r_short = r_long = r_global = R;
    } else {
        // EMAs for 128 and 4096 conflict windows
        double a_s = 2.0 / (128.0 + 1.0);
        double a_l = 2.0 / (4096.0 + 1.0);
        r_short = (1.0 - a_s) * r_short + a_s * R;
        r_long = (1.0 - a_l) * r_long + a_l * R;
        
        // Global average update
        r_global = (r_global * (double)total_restarts + R) / (double)(total_restarts + 1);

        // Update the arm used during the previous segment (Step 5 update rule)
        unsigned prev_arm = solver->heuristic;
        if (prev_arm < 2) {
            if (R > r_global) {
                alphas[prev_arm] += 1.0;
            } else {
                betas[prev_arm] += 1.0;
            }
        }
    }

    // Step 4: Search Volatility Index (V)
    double V = (r_long > 1e-9) ? (fabs(r_short - r_long) / r_long) : 0.0;
    bool forced_exploration = false;

    if (V > 0.3) {
        // Proportional decay factor
        double decay_val = (V > 0.8) ? 0.8 : V;
        double scale = 1.0 - decay_val;

        alphas[0] *= scale;
        betas[0] *= scale;
        alphas[1] *= scale;
        betas[1] *= scale;

        // Force exploration: pick arm with the lower current mean
        double mean0 = alphas[0] / (alphas[0] + betas[0]);
        double mean1 = alphas[1] / (alphas[1] + betas[1]);
        solver->heuristic = (mean0 < mean1) ? 0 : 1;
        forced_exploration = true;
    }

    // Step 5: Thompson Sampling Selection
    if (!forced_exploration) {
        double samples[2];
        for (unsigned i = 0; i < 2; i++) {
            // Normal approximation for Beta(a, b) sampling
            // mu = a/(a+b), sigma^2 = (ab)/((a+b)^2 * (a+b+1))
            double a = alphas[i];
            double b = betas[i];
            double mu = a / (a + b);
            double var = (a * b) / ((a + b) * (a + b) * (a + b + 1.0));
            double sigma = sqrt(var);

            // Box-Muller transform to sample from Normal(mu, sigma)
            double u1 = kissat_pick_double(&solver->random);
            double u2 = kissat_pick_double(&solver->random);
            if (u1 < 1e-10) u1 = 1e-10; // Avoid log(0)
            
            double z = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.1415926535897932 * u2);
            samples[i] = mu + z * sigma;
        }
        // Select the policy with the highest sampled value
        solver->heuristic = (samples[0] > samples[1]) ? 0 : 1;
    }

    total_restarts++;
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
