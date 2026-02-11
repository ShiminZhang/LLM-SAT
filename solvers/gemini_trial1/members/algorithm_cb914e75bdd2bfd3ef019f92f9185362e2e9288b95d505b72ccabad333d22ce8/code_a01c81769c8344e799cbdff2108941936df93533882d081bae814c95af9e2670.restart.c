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

  // Algorithm: Efficiency-Coasting Protected Restarts

  // Static state variables to track history between restarts.
  // We use static variables because we cannot modify the solver struct.
  static uint64_t last_props = 0;
  static uint64_t last_confl = 0;
  static double avg_eff = 1.0;
  static uint64_t last_binary = 0;
  static uint64_t last_glue2_conflict = 0;
  static bool initialized = false;

  // Initialize state on the very first call
  if (!initialized)
    {
      last_props = GET (propagations);
      last_confl = CONFLICTS;
      last_binary = GET (clauses_binary);
      initialized = true;
    }

  // Step 1: Calculate interval_efficiency
  uint64_t current_props = GET (propagations);
  uint64_t current_confl = CONFLICTS;

  uint64_t delta_props = current_props - last_props;
  uint64_t delta_confl = current_confl - last_confl;

  double interval_efficiency = 1.0;
  if (delta_confl > 0)
    interval_efficiency = (double) delta_props / (double) delta_confl;

  // Step 3: Determine dynamic scaling factor S
  // (We use the historical avg_eff before updating it)
  double S = 1.0;
  if (interval_efficiency > 1.1 * avg_eff)
    S = 1.2;
  else if (interval_efficiency < 0.9 * avg_eff)
    S = 0.8;

  // Step 4: Check standard Glucose restart condition with scaling S
  // trigger = fast_LBD_EMA > (S * 0.8 * slow_LBD_EMA)
  const double fast = AVERAGE (fast_glue);
  const double slow = AVERAGE (slow_glue);
  bool trigger = (fast > (S * 0.8 * slow));

  // Step 5: Implement 'Glue Protection'
  // "If a clause with LBD <= 2 was learned within the last 50 conflicts..."
  // Since we cannot iterate learned clauses efficiently, we use the count of
  // binary clauses (LBD=2) as a faithful proxy.
  uint64_t current_binary = GET (clauses_binary);
  if (current_binary > last_binary)
    {
      last_binary = current_binary;
      last_glue2_conflict = current_confl;
    }

  if (trigger)
    {
      if (current_confl >= last_glue2_conflict)
        {
          if (current_confl - last_glue2_conflict < 50)
            {
              trigger = false;
              kissat_extremely_verbose (solver,
                                        "restart inhibited by glue protection (LBD<=2 learned recently)");
            }
        }
    }

  // Step 6: Implement 'Propagation Coasting'
  // If trigger is true but efficiency is very high, override to maintain phase.
  if (trigger)
    {
      if (interval_efficiency > 1.5 * avg_eff)
        {
          trigger = false;
          kissat_extremely_verbose (solver,
                                    "restart inhibited by propagation coasting (high efficiency)");
        }
    }

  // Finalize: If restarting, update history and averages
  if (trigger)
    {
      // Step 2: Update global average_efficiency (EMA alpha=0.01)
      avg_eff += 0.01 * (interval_efficiency - avg_eff);

      // Update last counters for the next interval
      last_props = current_props;
      last_confl = current_confl;

      kissat_extremely_verbose (solver,
                                "ECPR trigger: eff=%.2f avg=%.2f S=%.1f fast=%.2f slow=%.2f",
                                interval_efficiency, avg_eff, S, fast, slow);
    }

  return trigger;
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
    // Reset MAB tracking variables
    unsigned stable_restarts = 0;
    solver->mab_reward[solver->heuristic] += log2(solver->mab_decisions) / log2(solver->mab_conflicts);
    
    // Clear per-variable MAB data
    for (all_variables(idx)) {
        solver->mab_chosen[idx] = 0;
    }
    solver->mab_chosen_tot = 0;
    solver->mab_decisions = 0;
    solver->mab_conflicts = 0;
    
    // Count stable restarts across all heuristics
    for (unsigned i = 0; i < solver->mab_heuristics; i++) {
        stable_restarts += solver->mab_select[i];
    }

    // Track recent gains with momentum
    static double recent_gains[10] = {0};
    static int gain_index = 0;
    static double momentum = 1.0;

    double current_gain = solver->mab_reward[solver->heuristic] / solver->mab_select[solver->heuristic];
    recent_gains[gain_index] = current_gain;
    gain_index = (gain_index + 1) % 10;

    // Compute average gain over recent window
    double avg_gain = 0;
    for (int i = 0; i < 10; i++) {
        avg_gain += recent_gains[i];
    }
    avg_gain /= 10;

    // Update momentum based on performance
    if (current_gain > avg_gain) {
        momentum *= 1.1;
    } else {
        momentum *= 0.9;
    }

    // Compute adaptive exploration parameter
    double adaptive_c = solver->mabc / (momentum * (stable_restarts + 1));

    // Select next heuristic
    if (stable_restarts < solver->mab_heuristics) {
        // Exploration phase: alternate between first two heuristics
        solver->heuristic = solver->heuristic == 0 ? 1 : 0;
    } else {
        // UCB-based selection
        double ucb[2];
        solver->heuristic = 0;
        for (unsigned i = 0; i < solver->mab_heuristics; i++) {
            ucb[i] = solver->mab_reward[i] / solver->mab_select[i] 
                   + sqrt(adaptive_c * log(stable_restarts + 1) / solver->mab_select[i]);
            if (i != 0 && ucb[i] > ucb[solver->heuristic]) {
                solver->heuristic = i;
            }
        }
    }
    
    // Update selection count for chosen heuristic
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
