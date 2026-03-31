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
    if (!victr_initialized) {
        for (int c = 0; c < 3; c++) {
            for (int a = 0; a < 3; a++) {
                victr_alpha[c][a] = VICTR_ALPHA_INITIAL;
                victr_beta[c][a] = VICTR_BETA_INITIAL;
            }
        }
        victr_initialized = true;
        victr_interval_start_conflicts = solver->statistics.conflicts;
        victr_interval_start_ticks = solver->ticks;
    }

    double sum_lbd = 0;
    double sq_sum_lbd = 0;
    for (unsigned i = 0; i < victr_window_filled; i++) {
        sum_lbd += victr_lbd_window[i];
        sq_sum_lbd += victr_lbd_window[i] * victr_lbd_window[i];
    }

    double window_mean = (victr_window_filled > 0) ? (sum_lbd / victr_window_filled) : 1.0;
    double window_var = (victr_window_filled > 0) ? ((sq_sum_lbd / victr_window_filled) - (window_mean * window_mean)) : 0;
    if (window_var < 0) window_var = 0;
    double cv = (window_mean > 0) ? (sqrt(window_var) / window_mean) : 0;

    if (victr_window_filled > 0) {
        double global_avg_lbd = AVERAGE(slow_glue);
        if (global_avg_lbd < 1.0) global_avg_lbd = 1.0;

        uint64_t delta_conflicts = solver->statistics.conflicts - victr_interval_start_conflicts;
        uint64_t delta_ticks = solver->ticks - victr_interval_start_ticks;
        
        double global_cps = (double)solver->statistics.conflicts / (double)(solver->ticks + 1);
        double current_cps = (double)delta_conflicts / (double)(delta_ticks + 1);
        
        double reward = (global_avg_lbd / window_mean) * (current_cps / (global_cps + 1e-6));

        if (reward > 1.0) {
            victr_alpha[victr_current_context][victr_selected_arm] += reward;
        } else {
            victr_beta[victr_current_context][victr_selected_arm] += (1.0 / (reward > 1e-3 ? reward : 1e-3));
        }
    }

    if (cv < 0.6) victr_current_context = 0;
    else if (cv <= 1.4) victr_current_context = 1;
    else victr_current_context = 2;

    double max_sample = -1.0;
    for (unsigned arm = 0; arm < 3; arm++) {
        double sample = victr_sample_beta(solver, victr_alpha[victr_current_context][arm], victr_beta[victr_current_context][arm]);
        if (sample > max_sample) {
            max_sample = sample;
            victr_selected_arm = arm;
        }
    }

    victr_interval_start_conflicts = solver->statistics.conflicts;
    victr_interval_start_ticks = solver->ticks;

    if (solver->statistics.conflicts - victr_last_decay_conflicts >= VICTR_DECAY_INTERVAL) {
        for (int c = 0; c < 3; c++) {
            for (int a = 0; a < 3; a++) {
                victr_alpha[c][a] *= 0.95;
                victr_beta[c][a] *= 0.95;
            }
        }
        victr_last_decay_conflicts = solver->statistics.conflicts;
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
