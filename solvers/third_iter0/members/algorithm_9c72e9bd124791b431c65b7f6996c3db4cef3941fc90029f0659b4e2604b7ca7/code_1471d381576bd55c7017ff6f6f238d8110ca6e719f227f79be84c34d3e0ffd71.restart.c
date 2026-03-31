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
  /* Multi-armed bandit restart heuristic switcher.
     This function is called from 'kissat_restart' (stable mode only).
     It updates solver->heuristic based on a discounted Thompson-sampling
     style update using only statistics/averages that exist in this code base.

     NOTE: We intentionally do NOT use AVERAGE(jump) since 'jump' is not a
     member of 'averages' in this repository version (compile error). */

  /* Persistent bandit state (process-wide, like the injected code). */
  static double alphas[4] = {1.0, 1.0, 1.0, 1.0};
  static double betas[4]  = {1.0, 1.0, 1.0, 1.0};
  static double h_history[50];
  static int h_idx = 0, h_cnt = 0;

  static uint64_t last_restarts = 0;
  static uint64_t last_restart_conflicts = 0;

  /* Arm selection state. */
  static int selected_arm = 0;
  static uint64_t arm_restarts[4] = {1, 1, 1, 1};

  /* Only update after an actual restart happened since last time. */
  if (solver->statistics.restarts <= last_restarts)
    return;

  /* --- Reward computation (uses existing averages) ----------------------- */

  /* Use averages that exist: fast_glue, level.  Approximate "progress"
     by conflicts since last restart (avoid non-existing 'jump'). */
  const double avg_glue = fmax (1.0, AVERAGE (fast_glue));
  const double avg_lvl = fmax (1.1, AVERAGE (level));

  const uint64_t conflicts_since_last =
      (last_restart_conflicts ? (CONFLICTS - last_restart_conflicts) : 0);

  /* Prefer arms that achieve lower glue and/or allow longer productive runs.
     Keep reward in [0,1]. */
  const double denom = avg_glue * log2 (avg_lvl);
  const double raw_R = denom > 0 ? ((double) conflicts_since_last) / denom : 0.0;
  const double R = raw_R / (raw_R + 10.0);

  /* Update Beta parameters for the arm that was active. */
  alphas[selected_arm] += R;
  betas[selected_arm] += (1.0 - R);

  /* --- Context signal: trail polarity entropy --------------------------- */

  double H = 0.0;
  const size_t trail_size = SIZE_STACK (solver->trail);
  if (trail_size) {
    unsigned pos_count = 0;
    /* Trail stores unsigned literals in this code base; iterate accordingly
       to avoid signedness warnings. */
    for (all_stack (unsigned, ulit, solver->trail))
      if (ulit) /* positive literal encoding is non-zero */
        pos_count++;

    const double p = (double) pos_count / (double) trail_size;
    if (p > 0.0 && p < 1.0)
      H = -(p * log2 (p) + (1.0 - p) * log2 (1.0 - p));
  }

  /* If entropy shifts strongly, reset bandit (simple change detector). */
  if (h_cnt == 50) {
    double h_sum = 0.0, h_sq_sum = 0.0;
    for (int i = 0; i < 50; i++) {
      h_sum += h_history[i];
      h_sq_sum += h_history[i] * h_history[i];
    }
    const double h_avg = h_sum / 50.0;
    const double var = (h_sq_sum / 50.0) - (h_avg * h_avg);
    const double h_std = sqrt (var > 0.0 ? var : 0.0);

    if (h_std > 0.0 && fabs (H - h_avg) > 2.0 * h_std) {
      for (int i = 0; i < 4; i++) {
        alphas[i] = 1.0;
        betas[i] = 1.0;
      }
    }
  }

  h_history[h_idx] = H;
  h_idx = (h_idx + 1) % 50;
  if (h_cnt < 50)
    h_cnt++;

  /* --- Discount + Thompson-like sampling to pick next arm --------------- */

  double best_sample = -1.0;
  for (int i = 0; i < 4; i++) {
    /* Discount factor gamma. */
    alphas[i] *= 0.92;
    betas[i] *= 0.92;
    if (alphas[i] < 1.0)
      alphas[i] = 1.0;
    if (betas[i] < 1.0)
      betas[i] = 1.0;

    const double a = alphas[i], b = betas[i];
    const double mean = a / (a + b);
    const double variance =
        (a * b) / ((a + b) * (a + b) * (a + b + 1.0));

    /* Crude sample around mean using solver RNG (keeps dependencies local). */
    const double noise = (kissat_pick_double (&solver->random) - 0.5);
    const double sample = mean + noise * sqrt (variance);

    if (sample > best_sample) {
      best_sample = sample;
      selected_arm = i;
    }
  }

  arm_restarts[selected_arm]++;

  /* Map arms to existing heuristic IDs.
     We keep it conservative: use 0/1 if available, otherwise clamp. */
  unsigned new_heuristic = (unsigned) selected_arm;
  if (new_heuristic > 1)
    new_heuristic = (new_heuristic & 1u);

  solver->heuristic = new_heuristic;

  last_restarts = solver->statistics.restarts;
  last_restart_conflicts = CONFLICTS;
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
