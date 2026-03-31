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
  /* This project already defines `kissat_restarting` in this file.
     The injected code accidentally redefined it and also referenced
     non-existing statistics fields.  The MAB logic belongs into a
     separate helper called from `kissat_restart`.  */

  assert (solver);
  assert (solver->stable);
  assert (solver->mab);

  /* Thompson-sampling state (kept across restarts). */
  enum { ARMS = 4, WINDOW_TS = 100, WINDOW_MINMAX = 500, WINDOW_ENTROPY = 50 };

  const uint64_t LUBY_BASE_A1 = 64;
  const uint64_t LUBY_BASE_A2 = 512;
  const double GLUCOSE_MARGIN_A3 = 1.1;
  const double GLUCOSE_MARGIN_A4 = 1.25;
  const double PI_VAL = 3.14159265358979323846;

  /* Local helper: iterative Luby (avoids extra global symbol). */
  auto uint64_t luby (uint64_t i) -> uint64_t {
    /* Standard Luby sequence (1-indexed). */
    uint64_t k = 1;
    while (1) {
      uint64_t p2k = (uint64_t) 1 << k;
      if (i == p2k - 1)
        return (uint64_t) 1 << (k - 1);
      if (i < p2k - 1) {
        i = i - ((uint64_t) 1 << (k - 1)) + 1;
        k = 1;
        continue;
      }
      k++;
      if (k >= 63)
        return 1;
    }
  }

  static bool init = false;
  static uint64_t last_restart_count = 0;
  static int active_arm = 0;

  /* Saved context from the restart that just happened. */
  static double saved_level = 0.0;
  static double saved_lbd = 0.0;

  /* Reward history for min-max normalization. */
  static double r_history[WINDOW_MINMAX];
  static int r_hist_ptr = 0;
  static int r_hist_total = 0;

  /* Entropy history for context shift detection. */
  static double h_history[WINDOW_ENTROPY];
  static int h_hist_ptr = 0;
  static int h_hist_total = 0;

  /* Sliding window for TS updates. */
  static double window_R[WINDOW_TS];
  static int window_arm[WINDOW_TS];
  static int window_ptr = 0;
  static int window_total = 0;

  if (!init) {
    active_arm = 0;
    last_restart_count = solver->statistics.restarts;
    /* Initialize a reasonable first limit for arm 0. */
    solver->limits.restart.conflicts =
        CONFLICTS + LUBY_BASE_A1 * luby (1);
    init = true;
    return;
  }

  /* We are called from `kissat_restart` after INC(restarts) and
     ADD(restarts_levels, solver->level).  So a restart has just
     completed and statistics.restarts has increased. */
  if (solver->statistics.restarts <= last_restart_count)
    return;

  /* --- Step 3: reward (learning efficiency proxy) --- */
  double avg_level = 0.0;
  if (solver->statistics.restarts)
    avg_level = (double) solver->statistics.restarts_levels /
                (double) solver->statistics.restarts;

  const double current_lbd = (saved_lbd < 1.0) ? 1.0 : saved_lbd;
  const double denom = current_lbd * log2 (saved_level + 2.0);
  const double R = (denom > 0.0) ? (avg_level / denom) : 0.0;

  /* --- Step 4: rolling min-max normalization --- */
  r_history[r_hist_ptr] = R;
  r_hist_ptr = (r_hist_ptr + 1) % WINDOW_MINMAX;
  if (r_hist_total < WINDOW_MINMAX)
    r_hist_total++;

  double min_r = r_history[0], max_r = r_history[0];
  for (int i = 1; i < r_hist_total; i++) {
    if (r_history[i] < min_r)
      min_r = r_history[i];
    if (r_history[i] > max_r)
      max_r = r_history[i];
  }

  double norm_R =
      (max_r > min_r + 1e-9) ? (R - min_r) / (max_r - min_r) : 0.5;
  if (norm_R < 0.0)
    norm_R = 0.0;
  if (norm_R > 1.0)
    norm_R = 1.0;

  window_R[window_ptr] = norm_R;
  window_arm[window_ptr] = active_arm;
  window_ptr = (window_ptr + 1) % WINDOW_TS;
  if (window_total < WINDOW_TS)
    window_total++;

  /* --- Step 2: search entropy (trail polarity balance) --- */
  double pos = 0.0, neg = 0.0;
  for (all_stack (unsigned, lit, solver->trail)) {
    if (lit & 1)
      neg++;
    else
      pos++;
  }
  const double total = pos + neg;
  double h = 0.0;
  if (total > 0.0) {
    const double p = pos / total;
    if (p > 0.0 && p < 1.0)
      h = -p * log2 (p) - (1.0 - p) * log2 (1.0 - p);
  }

  /* --- Step 5: context shift trigger --- */
  if (h_hist_total >= WINDOW_ENTROPY) {
    double sum_h = 0.0, sum_sq_h = 0.0;
    for (int i = 0; i < WINDOW_ENTROPY; i++) {
      sum_h += h_history[i];
      sum_sq_h += h_history[i] * h_history[i];
    }
    const double mean_h = sum_h / WINDOW_ENTROPY;
    const double var_h = (sum_sq_h / WINDOW_ENTROPY) - mean_h * mean_h;
    const double std_h = sqrt (var_h > 0.0 ? var_h : 0.0);
    if (std_h > 1e-6 && fabs (h - mean_h) > 2.0 * std_h) {
      window_total = 0;
      window_ptr = 0;
    }
  }
  h_history[h_hist_ptr] = h;
  h_hist_ptr = (h_hist_ptr + 1) % WINDOW_ENTROPY;
  if (h_hist_total < WINDOW_ENTROPY)
    h_hist_total++;

  /* --- Thompson sampling policy (normal approx to Beta) --- */
  double best_sample = -1e300;
  int best_arm = 0;

  for (int i = 0; i < ARMS; i++) {
    double alpha = 1.0, beta = 1.0;
    for (int j = 0; j < window_total; j++) {
      if (window_arm[j] == i) {
        alpha += window_R[j];
        beta += (1.0 - window_R[j]);
      }
    }

    const double ab = alpha + beta;
    const double mu = alpha / ab;
    const double sigma =
        sqrt ((alpha * beta) / (ab * ab * (ab + 1.0)));

    const double u1 = kissat_pick_double (&solver->random);
    const double u2 = kissat_pick_double (&solver->random);
    const double z =
        sqrt (-2.0 * log (u1 + 1e-9)) * cos (2.0 * PI_VAL * u2);

    const double sample = mu + z * sigma;
    if (sample > best_sample) {
      best_sample = sample;
      best_arm = i;
    }
  }

  active_arm = best_arm;
  last_restart_count = solver->statistics.restarts;

  /* Update the next restart conflict limit depending on chosen arm.
     (Arms 2/3 are "glucose-style" checks, so keep a small throttle.) */
  if (active_arm == 0)
    solver->limits.restart.conflicts =
        CONFLICTS + LUBY_BASE_A1 * luby (last_restart_count + 1);
  else if (active_arm == 1)
    solver->limits.restart.conflicts =
        CONFLICTS + LUBY_BASE_A2 * luby (last_restart_count + 1);
  else
    solver->limits.restart.conflicts = CONFLICTS + 50;

  /* Also update heuristic selection based on the arm.
     Keep it simple: 0/2 -> keep current heuristic, 1/3 -> toggle. */
  if (active_arm == 1 || active_arm == 3)
    solver->heuristic ^= 1;

  /* Save context for reward computation at the next restart. */
  saved_level = (double) solver->level;
  saved_lbd = AVERAGE (fast_glue);
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
