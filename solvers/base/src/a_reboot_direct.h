/**
 * (taomengxia) [reboot: direct] 2025-04-16
 */

#ifdef ENABLE_REBOOT

#ifndef _a_reboot_direct_h_INCLUDED
#define _a_reboot_direct_h_INCLUDED

#include <stdbool.h>

struct kissat;

bool kissat_rebooting_direct(struct kissat *);
int kissat_reboot_direct(struct kissat *);

#endif //_reboot_direct_h_INCLUDED

#endif