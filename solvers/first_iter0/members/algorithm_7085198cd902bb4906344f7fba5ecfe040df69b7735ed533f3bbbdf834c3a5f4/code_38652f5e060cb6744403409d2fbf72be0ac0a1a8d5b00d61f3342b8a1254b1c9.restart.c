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
#include <stdbool.h>
#include <stdint.h>

#define VICTR_WINDOW_SIZE 512
#define VICTR_PI 3.14159265358979323846

static double sample_beta_victr(double a, double b, kissat *solver) {
    double total = a + b;
    if (total <= 0) return 0.5;
    double mu = a / total;
    double var = (a * b) / (total * total * (total + 1.0));
    double sd = sqrt(var > 0 ? var : 1e-10);
    double u1 = kissat_pick_double(&solver->random);
    double u2 = kissat_pick_double(&solver->random);
    if (u1 < 1e-9) u1 = 1e-9;
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * VICTR_PI * u2);
    return mu + z * sd;
}

bool kissat_restarting (kissat *solver) {
    // Basic Kissat guards
    assert (solver->unassigned);
    if (!GET_OPTION (restart)) return false;
    if (!solver->level) return false;
    if (solver->statistics.conflicts < solver->limits.restart.conflicts) return false;

    // Algorithm Statics
    static double alphas[3][3] = {{1,1,1}, {1,1,1}, {1,1,1}};
    static double betas[3][3] = {{1,1,1}, {1,1,1}, {1,1,1}};
    static double window[VICTR_WINDOW_SIZE] = {0};
    static unsigned window_ptr = 0;
    static bool window_full = false;
    static int chosen_arm = -1;
    static int last_context = -1;
    static uint64_t last_decay_conflict = 0;
    static uint64_t conflicts_at_last_restart = 0;
    static uint64_t ticks_at_last_restart = 0;
    static uint64_t last_conflict_count = 0;

    // Step 1: Maintain sliding window of learned clause LBDs
    if (solver->statistics.conflicts > last_conflict_count) {
        window[window_ptr] = AVERAGE(fast_glue);
        window_ptr = (window_ptr + 1) % VICTR_WINDOW_SIZE;
        if (window_ptr == 0) window_full = true;
        last_conflict_count = solver->statistics.conflicts;
    }

    // Step 2: Calculate CV and Discretize search state
    unsigned n = window_full ? VICTR_WINDOW_SIZE : window_ptr;
    if (n < 2) return false;

    double sum = 0, sq_sum = 0;
    for (unsigned i = 0; i < n; i++) {
        sum += window[i];
        sq_sum += window[i] * window[i];
    }
    double mean = sum / n;
    double var = (sq_sum / n) - (mean * mean);
    double stddev = sqrt(var > 0 ? var : 0);
    double cv = (mean > 0) ? (stddev / mean) : 0;

    int current_context = (cv < 0.6) ? 0 : (cv <= 1.4 ? 1 : 2);

    // Step 7: Periodic and Transition Decay
    bool periodic_decay = (solver->statistics.conflicts >= last_decay_conflict + 10000);
    bool transition_decay = (last_context != -1 && current_context != last_context);

    if (periodic_decay || transition_decay) {
        if (periodic_decay) {
            double lambda = (current_context == 2) ? 0.85 : (current_context == 1 ? 0.95 : 0.98);
            for (int c = 0; c < 3; c++) {
                for (int a = 0; a < 3; a++) {
                    alphas[c][a] *= lambda;
                    betas[c][a] *= lambda;
                }
            }
            last_decay_conflict = solver->statistics.conflicts;
        }
        if (transition_decay) {
            for (int c = 0; c < 3; c++) {
                for (int a = 0; a < 3; a++) {
                    alphas[c][a] *= 0.90;
                    betas[c][a] *= 0.90;
                }
            }
        }
    }
    last_context = current_context;

    // Step 4: Thompson Sampling (pick arm if not already picked for this cycle)
    if (chosen_arm == -1) {
        double s0 = sample_beta_victr(alphas[current_context][0], betas[current_context][0], solver);
        double s1 = sample_beta_victr(alphas[current_context][1], betas[current_context][1], solver);
        double s2 = sample_beta_victr(alphas[current_context][2], betas[current_context][2], solver);
        
        if (s0 >= s1 && s0 >= s2) chosen_arm = 0;
        else if (s1 >= s0 && s1 >= s2) chosen_arm = 1;
        else chosen_arm = 2;
    }

    // Step 3: Identify conditions for arms
    bool trigger = false;
    const double fast = AVERAGE(fast_glue);
    const double slow = AVERAGE(slow_glue);
    const double margin = (100.0 + GET_OPTION(restartmargin)) / 100.0;

    if (chosen_arm == 0) {
        trigger = kissat_reluctant_triggered(&solver->reluctant);
    } else if (chosen_arm == 1) {
        trigger = (fast > slow * margin);
    } else {
        trigger = (fast > slow * margin * 1.15); // Lazy EMA
    }

    // Step 5 & 6: Reward Update on Restart
    if (trigger) {
        double global_lbd_avg = slow;
        double window_lbd_avg = mean;
        
        double current_cps = (double)(solver->statistics.conflicts - conflicts_at_last_restart) / 
                             (double)(solver->statistics.ticks - ticks_at_last_restart + 1);
        double global_cps = (double)solver->statistics.conflicts / (double)(solver->statistics.ticks + 1);
        
        double R = 1.0;
        if (window_lbd_avg > 0 && global_cps > 0) {
            R = (global_lbd_avg / window_lbd_avg) * (current_cps / global_cps);
        }
        if (R > 5.0) R = 5.0; // Cap reward to prevent explosion

        if (R > 1.0) alphas[current_context][chosen_arm] += R;
        else betas[current_context][chosen_arm] += (1.0 / (R > 0.01 ? R : 0.01));

        // Reset for next cycle
        conflicts_at_last_restart = solver->statistics.conflicts;
        ticks_at_last_restart = solver->statistics.ticks;
        chosen_arm = -1; 
        return true;
    }

    return false;
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
