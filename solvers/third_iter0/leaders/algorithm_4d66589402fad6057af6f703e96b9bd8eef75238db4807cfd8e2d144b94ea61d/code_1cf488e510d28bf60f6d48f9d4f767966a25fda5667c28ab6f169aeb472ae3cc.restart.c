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
  // Keep this function self-contained and only use symbols available via
  // the current includes of 'restart.c'.

  // Basic guards: only meaningful in stable mode with restart enabled.
  if (!GET_OPTION (restart))
    return;
  if (!solver->stable)
    return;

  // ---- Luby helper (iterative, 1-indexed) ----
  // Returns: 1,1,2,1,1,2,4,1,1,2,1,1,2,4,8,...
  auto uint64_t luby (uint64_t i) -> uint64_t {
    uint64_t k = 1;
    while ((UINT64_C (1) << k) - 1 < i)
      k++;
    while (i != (UINT64_C (1) << k) - 1) {
      i = i - ((UINT64_C (1) << (k - 1)) - 1);
      k--;
      while ((UINT64_C (1) << k) - 1 < i)
        k++;
    }
    return UINT64_C (1) << (k - 1);
  }

  // ---- Bandit state (file-static inside function) ----
  static double n_i[4] = {0, 0, 0, 0};   // discounted counts
  static double r_sum[4] = {0, 0, 0, 0}; // discounted reward sums
  static double h_hist[50];              // entropy window
  static int h_idx = 0, h_cnt = 0;

  static uint64_t luby_i[2] = {1, 1};    // indices for two Luby arms
  static uint64_t last_conf = 0;         // conflicts at last restart
  static int active_arm = 0;             // currently selected arm
  static bool initialized = false;

  if (!initialized) {
    for (int i = 0; i < 50; i++)
      h_hist[i] = 0.0;
    last_conf = CONFLICTS;
    initialized = true;
  }

  // ---- Arm trigger check (whether we *would* restart now) ----
  const double fast = AVERAGE (fast_glue);
  const double slow = AVERAGE (slow_glue);
  const uint64_t delta_conf = CONFLICTS - last_conf;

  bool triggered = false;
  if (active_arm == 0) { // Luby base 64
    const uint64_t limit = luby (luby_i[0]) * UINT64_C (64);
    triggered = (delta_conf >= limit);
  } else if (active_arm == 1) { // Luby base 512
    const uint64_t limit = luby (luby_i[1]) * UINT64_C (512);
    triggered = (delta_conf >= limit);
  } else if (active_arm == 2) { // Glucose fast
    triggered = (fast > slow * 1.05);
  } else { // active_arm == 3, Glucose slow
    triggered = (fast > slow * 1.25);
  }

  // If no restart is triggered, do not change heuristic.
  if (!triggered)
    return;

  // ---- Compute entropy of trail literal polarities ----
  // Note: solver->trail stores unsigned literals in this code base.
  unsigned pos = 0, neg = 0;
  for (all_stack (unsigned, lit, solver->trail)) {
    if (lit & 1u)
      neg++;
    else
      pos++;
  }

  const double total = (double) pos + (double) neg;
  const double p_pos = total > 0.0 ? (double) pos / total : 0.5;

  double h_curr = 0.0;
  if (p_pos > 1e-12 && p_pos < (1.0 - 1e-12)) {
    // Use log()/log(2) to avoid relying on log2() availability.
    const double inv_log2 = 1.0 / log (2.0);
    const double p_neg = 1.0 - p_pos;
    h_curr = -(p_pos * log (p_pos) + p_neg * log (p_neg)) * inv_log2;
  }

  // ---- Reward proxy ----
  const double lbd_proxy = fast + 1.0;
  const double level = (double) solver->level;
  const double inv_log2 = 1.0 / log (2.0);
  const double denom = lbd_proxy * (log (level + 2.0) * inv_log2);
  const double reward = denom > 0.0 ? (level * 0.5) / denom : 0.0;

  // ---- Discounted UCB update ----
  const double gamma = 0.92;
  for (int i = 0; i < 4; i++) {
    n_i[i] *= gamma;
    r_sum[i] *= gamma;
  }
  n_i[active_arm] += 1.0;
  r_sum[active_arm] += reward;

  // ---- Context shift: reset on large entropy deviation ----
  if (h_cnt == 50) {
    double sum = 0.0, sumsq = 0.0;
    for (int i = 0; i < 50; i++) {
      sum += h_hist[i];
      sumsq += h_hist[i] * h_hist[i];
    }
    const double mean = sum / 50.0;
    double var = sumsq / 50.0 - mean * mean;
    if (var < 0.0)
      var = 0.0;
    const double stddev = sqrt (var);

    if (stddev > 0.0 && fabs (h_curr - mean) > 2.0 * stddev) {
      for (int i = 0; i < 4; i++) {
        n_i[i] = 0.0;
        r_sum[i] = 0.0;
      }
    }
  }

  h_hist[h_idx] = h_curr;
  h_idx = (h_idx + 1) % 50;
  if (h_cnt < 50)
    h_cnt++;

  // ---- Select next arm (D-UCB) ----
  double total_n = 0.0;
  for (int i = 0; i < 4; i++)
    total_n += n_i[i];

  int arm_used = active_arm;
  double best = -1e300;
  int best_arm = active_arm;

  for (int i = 0; i < 4; i++) {
    double ucb;
    if (n_i[i] < 0.01) {
      // Force exploration of untried arms.
      ucb = 1e9 + kissat_pick_double (&solver->random);
    } else {
      const double mean = r_sum[i] / n_i[i];
      const double bonus = 2.0 * sqrt (log (total_n + 1.0) / n_i[i]);
      ucb = mean + bonus + kissat_pick_double (&solver->random) * 1e-6;
    }
    if (ucb > best) {
      best = ucb;
      best_arm = i;
    }
  }

  active_arm = best_arm;

  // ---- Advance state for the arm that triggered this restart ----
  if (arm_used == 0)
    luby_i[0]++;
  else if (arm_used == 1)
    luby_i[1]++;

  last_conf = CONFLICTS;

  // ---- Map arm choice to solver heuristic ----
  // We only change the heuristic here; the actual restart/backtrack is done
  // by the caller ('kissat_restart').
  //
  // Heuristic mapping:
  //  - Arms 0/1: keep current heuristic (no change).
  //  - Arm 2: prefer "focused" style heuristic if available.
  //  - Arm 3: prefer "stable" style heuristic if available.
  //
  // Since the concrete heuristic IDs are solver-specific, we conservatively
  // toggle between 0 and 1 if those are used, otherwise leave unchanged.
  if (active_arm == 2 || active_arm == 3) {
    // If the solver uses a binary heuristic selector (common in Kissat forks),
    // toggle it; otherwise do nothing.
    if (solver->heuristic == 0u)
      solver->heuristic = 1u;
    else if (solver->heuristic == 1u)
      solver->heuristic = 0u;
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
