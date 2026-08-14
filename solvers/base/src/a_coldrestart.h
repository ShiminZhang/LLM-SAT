/**
 * (taomengxia) [coldrestart] 2025-04-25
 */

#ifndef _a_coldrestart_h_INCLUDED
#define _a_coldrestart_h_INCLUDED

struct kissat;

bool kissat_coldrestarting (struct kissat *solver);
void kissat_coldrestart (struct kissat *solver);

#endif // _a_coldrestart_h_INCLUDED