#include "bump.h"
#include "analyze.h"
#include "inlineheap.h"
#include "inlinequeue.h"
#include "inlinevector.h"
#include "internal.h"
#include "logging.h"
#include "print.h"
#include "rank.h"
#include "sort.h"

#define RANK(A) ((A).rank)
#define SMALLER(A, B) (RANK (A) < RANK (B))

#define RADIX_SORT_BUMP_LIMIT 32

static void sort_bump (kissat *solver) {
  const size_t size = SIZE_STACK (solver->analyzed);
  if (size < RADIX_SORT_BUMP_LIMIT) {
    LOG ("quick sorting %zu analyzed variables", size);
    SORT_STACK (datarank, solver->ranks, SMALLER);
  } else {
    LOG ("radix sorting %zu analyzed variables", size);
    RADIX_STACK (datarank, unsigned, solver->ranks, RANK);
  }
}

void kissat_rescale_scores (kissat *solver) {
  INC (rescaled);
  heap *scores = &solver->scores;
  const double max_score = kissat_max_score_on_heap (scores);
  kissat_phase (solver, "rescale", GET (rescaled),
                "maximum score %g increment %g", max_score, solver->scinc);
  const double rescale = MAX (max_score, solver->scinc);
  assert (rescale > 0);
  const double factor = 1.0 / rescale;
  kissat_rescale_heap (solver, scores, factor);
  solver->scinc *= factor;
  kissat_phase (solver, "rescale", GET (rescaled), "rescaled by factor %g",
                factor);
}
void kissat_bump_score_increment (kissat *solver) {
  // Step 1: Calculate the search efficiency ratio R = (EMA_short_LBD / EMA_long_LBD)
  // In Kissat, fast_glue is the short-term EMA and slow_glue is the long-term EMA.
  const double lbd_short = solver->averages[solver->stable].fast_glue.value;
  const double lbd_long = solver->averages[solver->stable].slow_glue.value;
  const double R = (lbd_long > 0) ? (lbd_short / lbd_long) : 1.0;

  // Step 2: Determine the conflict depth impact ratio D = (current_conflict_level / EMA_conflict_level)
  // solver->level is the current decision level where the conflict was found.
  const double current_level = (double) solver->level;
  const double level_ema = solver->averages[solver->stable].level.value;
  const double D = (level_ema > 0) ? (current_level / level_ema) : 1.0;

  // Step 3: Initialize the base decay variable delta using the standard option value
  double delta = GET_OPTION (decay) * 1e-3;

  // Step 4: If R > 1.10, increase delta by 15% to force aggressive reorganization
  if (R > 1.10) {
    delta *= 1.15;
  }

  // Ensure delta stays within the stable range (0, 0.5] as per critical rules
  if (delta > 0.5) delta = 0.5;
  if (delta < 0.0) delta = 0.0;

  // Step 5: If D < 0.80, calculate a quadratic depth-boost factor B
  double B = 1.0;
  if (D < 0.80) {
    const double diff = 1.0 - D;
    B = 1.0 + (0.15 * diff * diff);
  }

  // Step 6: Compute the final increment factor F
  const double factor_decay = 1.0 / (1.0 - delta);
  const double factor_depth = (D < 0.80) ? B : 1.0;
  const double F = factor_decay * factor_depth;

  // Step 7: Update solver->scinc and check for rescale
  const double old_scinc = solver->scinc;
  const double new_scinc = old_scinc * F;

  LOG ("quadratic adaptive VSIDS: R=%g D=%g delta=%g B=%g factor=%g scinc=%g", 
       R, D, delta, B, F, new_scinc);

  solver->scinc = new_scinc;

  if (new_scinc > MAX_SCORE) {
    kissat_rescale_scores (solver);
  }
}

static inline void bump_analyzed_variable_score (kissat *solver,
                                                 unsigned idx) {
  heap *scores = &solver->scores;
  const double old_score = kissat_get_heap_score (scores, idx);
  const double inc = solver->scinc;
  const double new_score = old_score + inc;
  LOG ("new score[%u] = %g = %g + %g", idx, new_score, old_score, inc);
  kissat_update_heap (solver, scores, idx, new_score);
  if (new_score > MAX_SCORE)
    kissat_rescale_scores (solver);
}

void kissat_bump_variable (kissat *solver, unsigned idx) {
  bump_analyzed_variable_score (solver, idx);
}

static void bump_analyzed_variable_scores (kissat *solver) {
  flags *flags = solver->flags;

  for (all_stack (unsigned, idx, solver->analyzed))
    if (flags[idx].active)
      bump_analyzed_variable_score (solver, idx);

  kissat_bump_score_increment (solver);
}

static void move_analyzed_variables_to_front_of_queue (kissat *solver) {
  assert (EMPTY_STACK (solver->ranks));
  const links *const links = solver->links;
  for (all_stack (unsigned, idx, solver->analyzed)) {
    // clang-format off
    const datarank rank = { .data = idx, .rank = links[idx].stamp };
    // clang-format on
    PUSH_STACK (solver->ranks, rank);
  }

  sort_bump (solver);

  flags *flags = solver->flags;
  unsigned idx;

  for (all_stack (datarank, rank, solver->ranks))
    if (flags[idx = rank.data].active)
      kissat_move_to_front (solver, idx);

  CLEAR_STACK (solver->ranks);
}

void kissat_bump_analyzed (kissat *solver) {
  START (bump);
  const size_t bumped = SIZE_STACK (solver->analyzed);
  if (!solver->stable)
    move_analyzed_variables_to_front_of_queue (solver);
  else
    bump_analyzed_variable_scores (solver);
  ADD (literals_bumped, bumped);
  STOP (bump);
}

void kissat_update_scores (kissat *solver) {
  assert (solver->stable);
  heap *scores = kissat_get_scores(solver);
  for (all_variables (idx))
    if (ACTIVE (idx) && !kissat_heap_contains (scores, idx))
      kissat_push_heap (solver, scores, idx);
}

// CHB

void kissat_bump_chb(kissat * solver, unsigned v, double multiplier) {
  int64_t age = solver->statistics.conflicts - solver->conflicted_chb[v] + 1;
  double reward_chb = multiplier / age;
  double old_score = kissat_get_heap_score (&solver->scores_chb, v);
  double new_score = solver->step_chb * reward_chb + (1 - solver->step_chb) * old_score;
  LOG ("new score[%u] = %g vs %g",
     v, new_score, old_score);
  kissat_update_heap (solver, &solver->scores_chb, v, new_score);
}

void kissat_decay_chb(kissat * solver){
  if (solver->step_chb > solver->step_min_chb) solver->step_chb -= solver->step_dec_chb;
}

void
kissat_update_conflicted_chb (kissat * solver)
{
flags *flags = solver->flags;

for (all_stack (unsigned, idx, solver->analyzed))
  if (flags[idx].active)
      solver->conflicted_chb[idx]=solver->statistics.conflicts;
}
