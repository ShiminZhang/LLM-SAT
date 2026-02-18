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
    // Persistent state for the bandit algorithm
    // We use static variables because we cannot modify the solver struct.
    // Assuming max 2 heuristics (0=VSIDS, 1=CHB) as per standard Kissat MAB.
    static double fast_ema[2] = {0};
    static double slow_ema[2] = {0};
    static int stagnation[2] = {0};
    
    // Snapshot of the glue histogram to calculate interval deltas
    // MAX_GLUE_USED is typically 127 in Kissat.
    #ifndef MAX_GLUE_USED
    #define MAX_GLUE_USED 127
    #endif
    static uint64_t last_glue_snapshot[MAX_GLUE_USED + 1] = {0};
    static bool initialized = false;

    // Current active heuristic/mode
    unsigned h = solver->heuristic;
    if (h >= 2) h = 0; // Safety fallback

    // --- Step 1: Calculate Reward R ---
    // R = log2(1 + conflicts) * (sum(1 / LBD) / max(1, conflicts))
    
    double sum_inv_lbd = 0.0;
    uint64_t new_glue_clauses = 0; // For stagnation (LBD <= 2)
    
    // Access statistics for the current mode. 
    // MAB typically runs in stable mode, so we look at solver->stable (usually 1).
    int stats_mode = solver->stable ? 1 : 0;
    
    // Iterate through LBD values to calculate the sum of 1/LBD for this interval
    for (int i = 0; i <= MAX_GLUE_USED; i++) {
        uint64_t current_count = solver->statistics.used[stats_mode].glue[i];
        
        // Handle initialization
        if (!initialized) {
            last_glue_snapshot[i] = current_count;
            continue;
        }
        
        uint64_t prev_count = last_glue_snapshot[i];
        uint64_t delta = 0;
        
        if (current_count > prev_count) {
            delta = current_count - prev_count;
        }
        
        // Update snapshot for next interval
        last_glue_snapshot[i] = current_count;
        
        if (delta > 0) {
            // LBD value approximation: index i corresponds to LBD i.
            // Clamp lower bound to 1.0 to avoid division by zero.
            double lbd = (double)i;
            if (lbd < 1.0) lbd = 1.0;
            
            sum_inv_lbd += delta * (1.0 / lbd);
            
            // Step 3 (partial): Count new glue clauses (LBD <= 2)
            if (i <= 2) {
                new_glue_clauses += delta;
            }
        }
    }
    
    // If this was the first run, we just initialized snapshots and return
    if (!initialized) {
        initialized = true;
        // Reset counters to ensure next interval is clean
        solver->mab_decisions = 0;
        solver->mab_conflicts = 0;
        for (all_variables(idx)) {
            solver->mab_chosen[idx] = 0;
        }
        solver->mab_chosen_tot = 0;
        return;
    }

    // Conflicts since last MAB update
    double conflicts = (double)solver->mab_conflicts;
    
    // Calculate R
    double R = 0.0;
    if (conflicts > 0.0) {
        double denom = (conflicts < 1.0) ? 1.0 : conflicts;
        R = log2(1.0 + conflicts) * (sum_inv_lbd / denom);
    }

    // --- Step 2: Update EMAs ---
    // Fast_EMA (alpha=0.3), Slow_EMA (alpha=0.05)
    // We update the EMAs for the currently active mode 'h'
    
    // Check if EMAs are zero (first update), initialize directly to R to avoid ramp-up
    if (fast_ema[h] == 0.0 && slow_ema[h] == 0.0 && R > 0.0) {
        fast_ema[h] = R;
        slow_ema[h] = R;
    } else {
        fast_ema[h] = 0.3 * R + 0.7 * fast_ema[h];
        slow_ema[h] = 0.05 * R + 0.95 * slow_ema[h];
    }

    // --- Step 3: Update Stagnation Counter ---
    if (new_glue_clauses == 0) {
        stagnation[h]++;
    } else {
        stagnation[h] = 0;
    }

    // --- Step 4: Calculate Priority P ---
    unsigned best_h = 0;
    double best_P = -1e100; // Negative infinity
    
    // Iterate over available heuristics (typically 2)
    unsigned num_heuristics = solver->mab_heuristics;
    if (num_heuristics < 2) num_heuristics = 2; // Default fallback

    for (unsigned i = 0; i < num_heuristics; i++) {
        // P = Slow_EMA + 1.5 * (Fast_EMA - Slow_EMA)
        double P = slow_ema[i] + 1.5 * (fast_ema[i] - slow_ema[i]);
        
        // --- Step 5: Apply Penalty ---
        // If Stagnation_Counter > 6 and the mode is currently active (i == h)
        if (i == h && stagnation[i] > 6) {
            P = P * 0.5;
        }
        
        // Select max P
        if (P > best_P) {
            best_P = P;
            best_h = i;
        }
    }

    // --- Step 6: Select Next Mode ---
    solver->heuristic = best_h;

    // --- Housekeeping (Standard Kissat MAB Reset) ---
    // Increment selection count for the chosen heuristic
    solver->mab_select[best_h]++;
    
    // Reset MAB tracking variables for the next interval
    solver->mab_decisions = 0;
    solver->mab_conflicts = 0;
    
    // Reset per-variable chosen counts
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
