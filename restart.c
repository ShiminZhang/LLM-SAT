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

bool kissat_restarting(kissat *solver) {
  // Bandit Restarts with Dynamic Window

  // --- Parameters and defines ---
  enum { NUM_STRATS = 4 };
  enum { STRAT_LUBY = 0, STRAT_GEOM = 1, STRAT_UNIF = 2, STRAT_ADAPT = 3 };

  // Bandit state (static for persistence across calls in single-threaded Kissat)
  static unsigned strat = STRAT_LUBY;
  static unsigned last_strat = STRAT_LUBY;
  static uint64_t last_conflicts = 0;
  static uint64_t last_decisions = 0;
  static uint64_t last_restart_conflicts = 0;
  static uint64_t last_restart_decisions = 0;

  // Sliding window of NUM_WINS restart intervals
  enum { WIN_SIZE_MIN = 8, WIN_SIZE_MAX = 16, WIN_BUF_MAX = WIN_SIZE_MAX };
  static struct {
    double cpd;           // conflicts per decision
    unsigned strat;       // which strategy was used
    uint64_t confs;       // conflicts at window end
    uint64_t decs;        // decisions at window end
  } win[WIN_BUF_MAX];
  static unsigned win_count = 0;
  static unsigned win_head = 0;   // points to next window to overwrite

  // Bandit reward tracking: for each strategy, a list of recent rewards
  static double strat_rewards[NUM_STRATS][WIN_BUF_MAX];
  static unsigned strat_counts[NUM_STRATS] = {0, 0, 0, 0};

  // For dynamic window size
  static unsigned window_size = WIN_SIZE_MAX;

  // For epsilon-greedy
  static double epsilon = 0.20;

  // For "long" restart injection
  static bool long_restart_pending = false;
  static unsigned long_restart_left = 0;

  // --- Step 1: Is a restart allowed by basic Kissat policy? ---
  if (!solver->unassigned) return false;
  if (!GET_OPTION(restart)) return false;
  if (!solver->level) return false;
  if (CONFLICTS < solver->limits.restart.conflicts) return false;

  // --- Step 2: Dynamic adjust window size based on conflict variability ---
  // Compute variance of last window_size cpd values
  unsigned eff_win_size = win_count < window_size ? win_count : window_size;
  double sum = 0, sum2 = 0;
  for (unsigned i = 0; i < eff_win_size; i++) {
    double v = win[(win_head + WIN_BUF_MAX - i - 1) % WIN_BUF_MAX].cpd;
    sum += v;
    sum2 += v*v;
  }
  double mean = eff_win_size > 0 ? sum / eff_win_size : 0;
  double variance = eff_win_size > 1 ? (sum2 - sum*mean) / (eff_win_size - 1) : 0;
  // Window shrinks with high variance, grows with low
  if (eff_win_size >= 4) {
    if (variance > 2.0) window_size = WIN_SIZE_MIN;
    else if (variance < 0.25) window_size = WIN_SIZE_MAX;
    else window_size = WIN_SIZE_MIN + (unsigned)((WIN_SIZE_MAX-WIN_SIZE_MIN)*(2.0-variance)/1.75);
    if (window_size > WIN_SIZE_MAX) window_size = WIN_SIZE_MAX;
    if (window_size < WIN_SIZE_MIN) window_size = WIN_SIZE_MIN;
  }

  // --- Step 3: Compute conflicts-per-decision for last interval ---
  uint64_t cur_conflicts = CONFLICTS;
  uint64_t cur_decisions = solver->statistics.decisions;
  uint64_t interval_conflicts = cur_conflicts - last_restart_conflicts;
  uint64_t interval_decisions = cur_decisions - last_restart_decisions;
  double cpd = interval_decisions ? (double)interval_conflicts / (double)interval_decisions : (double)interval_conflicts;
  // Save stats for next interval
  last_restart_conflicts = cur_conflicts;
  last_restart_decisions = cur_decisions;

  // --- Step 4: Store window stats ---
  win[win_head].cpd = cpd;
  win[win_head].strat = strat;
  win[win_head].confs = cur_conflicts;
  win[win_head].decs = cur_decisions;
  win_head = (win_head + 1) % WIN_BUF_MAX;
  if (win_count < WIN_BUF_MAX) win_count++;

  // --- Step 5: Compute median cpd in window ---
  double median_cpd = cpd;
  if (win_count > 0) {
    // Copy window cpd vals
    double tmp[WIN_BUF_MAX];
    for (unsigned i = 0; i < win_count; i++) tmp[i] = win[i].cpd;
    // Insertion sort for small N
    for (unsigned i = 1; i < win_count; i++) {
      double v = tmp[i]; int j = i-1;
      while (j >= 0 && tmp[j] > v) { tmp[j+1]=tmp[j]; j--; }
      tmp[j+1]=v;
    }
    if (win_count % 2 == 1)
      median_cpd = tmp[win_count/2];
    else
      median_cpd = 0.5*(tmp[win_count/2-1]+tmp[win_count/2]);
  }

  // --- Step 6: Compute reward for this strat: improvement over median ---
  double reward = 0.0;
  if (win_count > 1) {
    reward = median_cpd > 0 ? (median_cpd - cpd) / median_cpd : 0;
    strat_rewards[strat][strat_counts[strat] % WIN_BUF_MAX] = reward;
    strat_counts[strat]++;
  }

  // --- Step 7: Epsilon-greedy strategy selection (decaying epsilon) ---
  // Epsilon shrinks from 0.20 to 0.10 in first 10,000 conflicts
  uint64_t total_conflicts = CONFLICTS;
  double eps_decay = (total_conflicts < 10000) ? 0.20 - 0.10 * ((double)total_conflicts/10000.0) : 0.10;
  epsilon = eps_decay;
  bool explore = (kissat_pick_double(&solver->random) < epsilon);

  // Compute mean reward for each strat
  double mean_rew[NUM_STRATS];
  for (unsigned s = 0; s < NUM_STRATS; s++) {
    unsigned n = strat_counts[s] < WIN_BUF_MAX ? strat_counts[s] : WIN_BUF_MAX;
    double sumr = 0; for (unsigned i = 0; i < n; i++) sumr += strat_rewards[s][i];
    mean_rew[s] = n ? sumr / n : 0.0;
  }

  // Select next strat: explore or exploit
  unsigned next_strat = strat;
  if (explore) {
    next_strat = kissat_pick_random(&solver->random, 0, NUM_STRATS);
  } else {
    // Pick strat with best mean reward
    unsigned best = 0;
    double best_val = mean_rew[0];
    for (unsigned s = 1; s < NUM_STRATS; s++) {
      if (mean_rew[s] > best_val) { best = s; best_val = mean_rew[s]; }
    }
    next_strat = best;
  }
  last_strat = strat;
  strat = next_strat;

  // --- Step 8: Schedule next restart limit based on selected strategy ---
  uint64_t base = 0, next_limit = 0;
  switch (strat) {
    case STRAT_LUBY: {
      static unsigned luby_idx = 1;
      base = GET_OPTION(restartint);
      // Luby sequence: luby(1), luby(2),...
      unsigned u = luby_idx;
      // Compute Luby value
      unsigned p = 1;
      while (p <= u) p <<= 1;
      p >>= 1;
      unsigned luby = (p == u) ? p : luby_idx - p + 1;
      next_limit = CONFLICTS + base * luby;
      luby_idx++;
      break;
    }
    case STRAT_GEOM: {
      static unsigned geom_cnt = 0;
      base = GET_OPTION(restartint);
      double ratio = 1.5;
      next_limit = CONFLICTS + (uint64_t)(base * pow(ratio, (double)geom_cnt));
      geom_cnt++;
      break;
    }
    case STRAT_UNIF: {
      base = GET_OPTION(restartint);
      unsigned mult = kissat_pick_random(&solver->random, 1, 5); // 1x-4x
      next_limit = CONFLICTS + base * mult;
      break;
    }
    case STRAT_ADAPT: {
      // Use median window cpd: if cpd gets worse, increase interval
      base = GET_OPTION(restartint);
      double factor = (cpd > median_cpd && median_cpd > 0) ? 2.0 : 1.0;
      next_limit = CONFLICTS + (uint64_t)(base * factor);
      break;
    }
    default:
      // Fallback: 1x
      base = GET_OPTION(restartint);
      next_limit = CONFLICTS + base;
      break;
  }

  // --- Step 9: Inject occasional long restart if no improvement in 2 windows ---
  static double last_rewards[2] = {0,0};
  bool no_improve = false;
  if (win_count >= 2) {
    last_rewards[0] = last_rewards[1];
    last_rewards[1] = reward;
    if (last_rewards[0] <= 0 && last_rewards[1] <= 0)
      no_improve = true;
  }
  bool inject_long = false;
  if (no_improve && kissat_pick_double(&solver->random) < 0.08) {
    // 8x base interval
    inject_long = true;
    next_limit = CONFLICTS + base * 8;
  }

  // --- Step 10: Set next restart limit ---
  solver->limits.restart.conflicts = next_limit;

  // --- Step 11: Run Kissat's glue-based emergency restart as fallback ---
  if (solver->stable) {
    const double fast = AVERAGE(fast_glue);
    const double slow = AVERAGE(slow_glue);
    const double margin = (100.0 + GET_OPTION(restartmargin)) / 100.0;
    const double limit = margin * slow;
    kissat_extremely_verbose(solver,
      "bandit restart glue limit %g = %.02f * %g (slow glue) %c %g (fast glue)",
      limit, margin, slow,
      (limit > fast    ? '>' : limit == fast ? '=' : '<'),
      fast
    );
    if (limit <= fast) return true;
  } else {
    // Focused mode: keep classic behaviour for now
    return (CONFLICTS >= solver->limits.restart.conflicts);
  }

  // --- Step 12: Always restart if over limit ---
  if (CONFLICTS >= solver->limits.restart.conflicts)
    return true;
  return false;
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
