// OpenEvolve baseline for the LLM-SAT comparison experiment.
//
// Only the function inside the EVOLVE block is injected into
// solvers/base/src/decide.c.  The surrounding Kissat source tree is fixed.

// # EVOLVE-BLOCK-START
int kissat_decide_phase (kissat *solver, unsigned idx) {
  bool force = GET_OPTION (forcephase);

  value *target;
  if (force)
    target = 0;
  else if (!GET_OPTION (target))
    target = 0;
  else if (solver->stable || GET_OPTION (target) > 1)
    target = solver->phases.target + idx;
  else
    target = 0;

  value *saved;
  if (force)
    saved = 0;
  else if (GET_OPTION (phasesaving))
    saved = solver->phases.saved + idx;
  else
    saved = 0;

  value res = 0;

  if (!solver->stable) {
    switch ((solver->statistics.switched >> 1) & 7) {
    case 1:
      res = INITIAL_PHASE;
      break;
    case 3:
      res = -INITIAL_PHASE;
      break;
    }
  }

  if (!res && target && (res = *target)) {
    LOG ("%s uses target decision phase %d", LOGVAR (idx), (int) res);
    INC (target_decisions);
  }

  if (!res && saved && (res = *saved)) {
    LOG ("%s uses saved decision phase %d", LOGVAR (idx), (int) res);
    INC (saved_decisions);
  }

  if (!res) {
    res = INITIAL_PHASE;
    LOG ("%s uses initial decision phase %d", LOGVAR (idx), (int) res);
    INC (initial_decisions);
  }
  assert (res);

  return res < 0 ? -1 : 1;
}
// # EVOLVE-BLOCK-END
