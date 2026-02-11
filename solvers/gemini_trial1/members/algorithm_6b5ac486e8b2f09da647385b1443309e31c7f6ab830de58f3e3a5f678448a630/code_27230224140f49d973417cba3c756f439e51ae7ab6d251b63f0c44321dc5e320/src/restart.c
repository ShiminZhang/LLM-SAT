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

  // --- Step 1 & 2: Maintain Custom EMAs and Calculate Momentum ---

  // Persistent state for the "Hybrid Glue-Protected Momentum" heuristic.
  // Using static variables as we cannot modify the solver struct.
  // Note: This assumes a single solver instance or sequential usage.
  
  static double h_fast_ema = 0;   // alpha = 0.05
  static double h_slow_ema = 0;   // alpha = 0.002
  static uint64_t last_conflicts_seen = 0;
  static uint64_t last_restart_at = 0;
  
  // State to back-calculate the raw LBD from the solver's internal EMAs
  // We track the last seen EMA value for both Focused (0) and Stable (1) modes
  static double last_solver_ema_val[2] = {0}; 
  static bool initialized = false;

  const uint64_t conflicts = CONFLICTS;

  // Detect solver reset or new instance
  if (conflicts < last_conflicts_seen) {
      initialized = false;
      last_conflicts_seen = 0;
      last_restart_at = 0;
      h_fast_ema = 0;
      h_slow_ema = 0;
  }

  // Determine current mode (0=Focused, 1=Stable)
  int mode = solver->stable ? 1 : 0;
  
  // Access solver's current fast glue EMA
  // EMA(fast_glue) expands to the smooth struct in the current mode's averages
  double current_solver_val = EMA(fast_glue).value;
  double current_solver_alpha = EMA(fast_glue).alpha;
  
  // We need the raw LBD of the *very last learned clause*.
  // Since we cannot access it directly, we reverse-engineer it from the change in the solver's EMA.
  static double last_raw_lbd = 0;

  if (!initialized) {
      // Initialize EMAs with the current system average
      h_fast_ema = current_solver_val;
      h_slow_ema = current_solver_val;
      
      // Initialize tracker for back-calculation
      last_solver_ema_val[0] = solver->averages[0].fast_glue.value;
      last_solver_ema_val[1] = solver->averages[1].fast_glue.value;
      
      last_raw_lbd = current_solver_val;
      last_conflicts_seen = conflicts;
      last_restart_at = conflicts; // Delay first restart
      initialized = true;
  } 
  else if (conflicts > last_conflicts_seen) {
      // A new clause (or clauses) was learned since last call.
      // Back-calculate the input LBD that caused the change in the solver's EMA.
      // Formula: new = old + alpha * (input - old)  =>  input = old + (new - old) / alpha
      
      double prev_val = last_solver_ema_val[mode];
      double lbd = current_solver_val; // Fallback
      
      if (current_solver_alpha > 1e-9) {
          lbd = prev_val + (current_solver_val - prev_val) / current_solver_alpha;
      }
      
      // Sanity check: LBD must be >= 1
      if (lbd < 1.0) lbd = 1.0;
      
      last_raw_lbd = lbd;

      // Update our custom persistent EMAs (Step 1)
      // fast_ema (alpha=0.05), slow_ema (alpha=0.002)
      h_fast_ema += 0.05 * (lbd - h_fast_ema);
      h_slow_ema += 0.002 * (lbd - h_slow_ema);

      // Update history for next iteration
      last_solver_ema_val[mode] = current_solver_val;
      last_conflicts_seen = conflicts;
  }

  // Calculate Momentum Ratio (Step 2)
  double R = 1.0;
  if (h_slow_ema > 1e-9) {
      R = h_fast_ema / h_slow_ema;
  }

  // --- Step 3: Enforce Minimum Run Length ---
  if (conflicts - last_restart_at < 50) {
      return false;
  }

  bool trigger = false;

  // --- Step 4 & 5: Mode-Specific Logic ---
  if (solver->stable) {
      // Step 5: Stable Mode
      // Trigger if R > 1.35 or Reluctant Doubling limit exceeded
      if (R > 1.35) {
          trigger = true;
      } else if (kissat_reluctant_triggered (&solver->reluctant)) {
          trigger = true;
      }
  } else {
      // Step 4: Focused Mode
      // Glue Clause Override: If last LBD <= 2, inhibit restart.
      // Using 2.05 to account for floating point back-calculation noise.
      if (last_raw_lbd <= 2.05) {
          trigger = false;
      } else {
          // Trigger if R > 1.08 (tightened tolerance)
          if (R > 1.08) {
              trigger = true;
          } 
          // Inhibit if R < 0.90 (widened protection)
          else if (R < 0.90) {
              trigger = false;
          }
          // Implicitly: if 0.90 <= R <= 1.08, do not trigger (default false)
      }
  }

  if (trigger) {
      last_restart_at = conflicts;
      kissat_extremely_verbose (solver,
          "restart triggered: mode=%s R=%.2f (fast=%.2f slow=%.2f) lbd=%.2f",
          solver->stable ? "stable" : "focused",
          R, h_fast_ema, h_slow_ema, last_raw_lbd);
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
