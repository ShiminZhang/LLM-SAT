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

    // Step 1: Compute metrics for the completed phase
    // L = Average LBD (fast glue EMA)
    double L = AVERAGE(fast_glue);
    // D = Average Decision Level (level EMA)
    double D = AVERAGE(level);
    unsigned N = solver->vars;
    
    // d_ratio = D / N (Normalize D by number of variables)
    double d_ratio = (N > 0) ? (D / (double)N) : 0.0;

    // Step 2: Base reward R = 1 / L
    // Protect against L near zero
    double R = (L > 1e-6) ? (1.0 / L) : 0.0;

    // Step 3: Apply Asymmetric Incentives
    // "If the completed phase was STABLE". This function is called within stable mode context.
    // Modify R -> R * (1.0 + (1.5 * d_ratio))
    if (solver->stable) {
        R = R * (1.0 + (1.5 * d_ratio));
    }

    // Step 4: Apply Fatigue Penalty
    // Track consecutive selections of the same mode
    static unsigned last_mode = 99999; // Initialize to invalid mode
    static unsigned consecutive_modes = 0;
    
    unsigned current_mode = solver->heuristic;
    
    if (current_mode == last_mode) {
        consecutive_modes++;
    } else {
        consecutive_modes = 1;
        last_mode = current_mode;
    }

    // If the solver has chosen the current mode > 4 times in a row, decay R
    if (consecutive_modes > 4) {
        R *= 0.6;
    }

    // Step 5: Non-stationary update to MAB statistics
    // Discount factor gamma = 0.95
    double gamma = 0.95;
    
    // Update the statistics for the current (just completed) arm
    // Reward: Discounted accumulated reward + current modified R
    solver->mab_reward[current_mode] = (solver->mab_reward[current_mode] * gamma) + R;
    
    // Count: Discounted visit count + 1
    // mab_select is unsigned, so we cast to double for calculation then back
    double discounted_count = ((double)solver->mab_select[current_mode] * gamma) + 1.0;
    solver->mab_select[current_mode] = (unsigned)discounted_count;

    // Step 6: Select the next mode using Thompson Sampling
    // We select the arm with the higher sampled value from updated Beta distributions.
    // Mapping: Alpha = Reward, Beta = Count - Reward.
    // We use a Normal Approximation for Beta sampling for efficiency.
    
    unsigned best_arm = 0;
    double best_sample = -1e100; // Start with a very low number

    for (unsigned i = 0; i < solver->mab_heuristics; i++) {
        double alpha = solver->mab_reward[i];
        double count = (double)solver->mab_select[i];
        
        // Ensure valid parameters for Beta/Normal approximation
        if (alpha < 1e-4) alpha = 1e-4;
        
        // Beta = Count - Alpha
        // If Reward > Count (possible due to bonus scaling), we clamp Beta to small positive
        double beta = count - alpha;
        if (beta < 1e-4) beta = 1e-4;
        
        // Normal Approximation of Beta(alpha, beta)
        // Mean = alpha / (alpha + beta)
        // Variance = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))
        
        double sum_ab = alpha + beta;
        double mean = alpha / sum_ab;
        double var = (alpha * beta) / (sum_ab * sum_ab * (sum_ab + 1.0));
        double std_dev = sqrt(var);

        // Box-Muller Transform for standard normal sample
        double u1 = kissat_pick_double(&solver->random);
        double u2 = kissat_pick_double(&solver->random);
        
        // Avoid log(0)
        if (u1 < 1e-10) u1 = 1e-10;
        
        // z ~ N(0, 1)
        double z = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265359 * u2);
        
        // sample ~ N(mean, std_dev) which approximates Beta(alpha, beta)
        double sample = mean + (z * std_dev);

        if (sample > best_sample) {
            best_sample = sample;
            best_arm = i;
        }
    }

    // Set the next heuristic
    solver->heuristic = best_arm;

    // Reset MAB tracking variables for the next phase (Standard Kissat housekeeping)
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
