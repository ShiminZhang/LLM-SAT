/**
 * (taomengxia) [reboot: file] 2025-04-14
 */

#ifdef ENABLE_REBOOT

#include "internal.h"
#include "inline.h"
#include "error.h"
#include "print.h"

#include <string.h>
#include <unistd.h>

bool kissat_rebooting_file(kissat *solver) {
    if (GET_OPTION(reboot) != 2)
        return false;
    if (CONFLICTS < (unsigned)GET_OPTION(rebootconflicts))
        return false;
    
    return true;
}

void kissat_reboot_file_add_useful_binary(struct kissat *solver, unsigned first, unsigned second) {
    if (GET_OPTION(reboot) != 2)
        return;

    const int elit_first = kissat_export_literal(solver, first);
    const unsigned eidx_first = ABS(elit_first);
    if (eidx_first > solver->original_variables)
        return;

    const int elit_second = kissat_export_literal(solver, second);
    const unsigned eidx_second = ABS(elit_second);
    if (eidx_second > solver->original_variables) {
        return;
    }

    assert(eidx_first <= solver->original_variables && eidx_second <= solver->original_variables);
    PUSH_STACK (solver->reboot_useful_binaries, elit_first);
    PUSH_STACK (solver->reboot_useful_binaries, elit_second);
}


static char *getFileName(const char *file_path) {
    char *file_name = strrchr(file_path, '/');
    if (file_name) {
        return file_name + 1;
    }
    return file_name;
}

static char* kissat_reboot_get_new_cnf_file_name(const char *old_path) {
    const char *prefix = "reboot_";
    const char *filename = getFileName(old_path);
    size_t prefix_len = strlen(prefix);
    size_t filename_len = strlen(filename);
    char *new_file_name = malloc(prefix_len + filename_len + 1);
    if (!new_file_name)
        return NULL;

    memcpy(new_file_name, prefix,  prefix_len);
    memcpy(new_file_name + prefix_len, filename,  filename_len + 1);
    return new_file_name;
}

static inline void kissat_reboot_check_clause_useful(struct kissat *solver, clause *c) {
    // Init used = 0
    c->used = 0;

    // Skip clause: 1. garbage; 2. irredundant; 3. size > rebootclausesize.
    if (c->garbage || !c->redundant || c->size > (unsigned)GET_OPTION(rebootclausesize)){
        return;
    }

    // Skip clause: has extension-variable
    for (all_literals_in_clause(lit, c)) {
        const int elit = kissat_export_literal(solver, lit);
        const unsigned eidx = ABS(elit);
        if (eidx > solver->original_variables)
            return;
    }

    // Change used in useful-caluses to 1
    c->used = 1;
}

static unsigned kissat_reboot_useful_learn_clauses_cnt(struct kissat *solver) {
    unsigned res = 0;

    // count of useful unit-clause
    const unsigned unit_stack_size = SIZE_STACK(solver->units);
    for (unsigned idx = 0; idx < unit_stack_size; idx++) {
        const int unit = PEEK_STACK(solver->units, idx);
        const unsigned eidx_unit = ABS(unit);
        if (eidx_unit > solver->original_variables)
            continue;
        res++;
    }

    // count of useful binary-clause
    assert(!(SIZE_STACK(solver->reboot_useful_binaries) % 2));
    res += SIZE_STACK(solver->reboot_useful_binaries) / 2;

    // count of useful larget-clause
    for (all_clauses(c)) {
        kissat_reboot_check_clause_useful(solver, c);
        if (c->used == 1) {
            res++;
        }
    }
    return res;
}

static void kissat_reboot_generate_new_cnf_file(struct kissat *solver) {
    assert(GET_OPTION(reboot) == 2);

    // Open input file: orignal input file  => used to read old clauses
    const char *input_path = solver->input_path;
    FILE *input_file = fopen(input_path, "r");
    // Open new file: generate new cnf-file
    const char *new_path = kissat_reboot_get_new_cnf_file_name(input_path);
    FILE *new_file = fopen(new_path, "w");

    if (!input_file || !new_file) {
        kissat_fatal("open file : %s or %s error", input_path, new_path);
    }

    // Set buffer_size
    const size_t BUFFER_SIZE = 8 * 1024 * 1024; // 8MB
    char* buffer = malloc(sizeof(char*) * BUFFER_SIZE);
    setvbuf(new_file, buffer, _IOFBF, BUFFER_SIZE);

    const unsigned useful_learn_cnt = kissat_reboot_useful_learn_clauses_cnt(solver);
    const unsigned all_clauses_cnt = solver->original_clauses + useful_learn_cnt;

    /// 1. Write cnf header into new_file
    fprintf(new_file, "p cnf %u %u\n", solver->original_variables, all_clauses_cnt);

    /// 2. Write orignal clauses into new_file
    char line[256];
    bool found_head = false;
    while (fgets(line, sizeof(line), input_file)) {
        if (!found_head && line[0] == 'p') {
            found_head = true;
            continue;
        } else {
            fputs(line, new_file);
        }
    }
    fclose(input_file);

    /// 3. Write useful learn-clauses into new_file
    //  3.1 Write useful unit-clause
    const unsigned unit_stack_size = SIZE_STACK(solver->units);
    for (unsigned idx = 0; idx < unit_stack_size; idx++) {
        const int unit = PEEK_STACK(solver->units, idx);
        const unsigned eidx_unit = ABS(unit);
        if (eidx_unit > solver->original_variables)
            continue;
        fprintf(new_file, "%d 0\n", unit);
    }
    //  3.2 Write useful binary-clause
    const unsigned useful_binary_cnt = SIZE_STACK(solver->reboot_useful_binaries) / 2;
    for (unsigned idx = 0; idx < useful_binary_cnt; idx++) {
        unsigned actual_idx = idx * 2;
        int first = PEEK_STACK(solver->reboot_useful_binaries, actual_idx);
        int second = PEEK_STACK(solver->reboot_useful_binaries, actual_idx + 1);
        fprintf(new_file, "%d %d 0\n", first, second);
    }
    //  3.3 Write useful large-clause
    for (all_clauses(c)) {
        if (!c->used)
            continue;
        
        assert(c->used == 1);
        for (all_literals_in_clause(lit, c)) {
            const int elit = kissat_export_literal(solver, lit);
            fprintf(new_file, "%d ", elit);
        }
        fprintf(new_file, "0\n");
    }

    fflush(new_file);
    fclose(new_file);
    free(buffer);
}

void kissat_reboot_file(struct kissat *solver, int argc, char **argv) {
    assert (GET_OPTION(reboot) == 2);

    // Generate new cnf file
    kissat_reboot_generate_new_cnf_file(solver);

    // Process options:   1. replace input_path to new_cnf_file; 2 add --reboot=false
    int new_argc = argc + 2;
    char **new_argv = malloc(new_argc * sizeof(char*));
    int idx = 0;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(solver->input_path, argv[i])) {
            // Replace old input_path with new reboot file
            new_argv[idx++] = kissat_reboot_get_new_cnf_file_name(solver->input_path);
        } else {
            new_argv[idx++] = strdup(argv[i]);
        }
    }
    new_argv[idx++] = "--reboot=0";
    new_argv[idx++] = NULL;

    kissat_section (solver, "reboot start");
    kissat_release(solver);

    // Reboot
    if (execv("/proc/self/exe", new_argv) == -1) {
        kissat_fatal("reboot failed");
    }
}

#endif