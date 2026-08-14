/**
 * (taomengxia) [reboot: file] 2025-04-14
 */

#ifdef ENABLE_REBOOT

#ifndef _a_reboot_file_h_INCLUDED
#define _a_reboot_file_h_INCLUDED

#include <stdbool.h>

struct kissat;

void kissat_reboot_file_add_useful_binary(struct kissat *solver, unsigned first, unsigned second);

bool kissat_rebooting_file(struct kissat *solver);
void kissat_reboot_file(struct kissat *solver, int argc, char **argv);

#endif  // _reboot_file_h_INCLUDED

#endif
