/*
 * checkmodel - verify a SAT solver's reported model against a DIMACS CNF.
 *
 * Usage: checkmodel <formula.cnf> <solver.log>
 *
 * The solver log is scanned (streaming) for the "s SATISFIABLE" status line
 * and for the "v ..." witness lines (kissat / DIMACS output format: signed
 * literals separated by whitespace, possibly spread across many "v" lines,
 * terminated by a "0").  The CNF is then streamed clause by clause and every
 * clause is checked to contain at least one literal satisfied by the model.
 * A clause with no satisfied literal (including clauses that depend only on
 * unassigned variables) means the model is NOT a witness.
 *
 * Exit codes:
 *   0  model verified                        (prints "MODEL VERIFIED")
 *   1  model missing/inconsistent/falsified  (prints "MODEL FAILED: ...")
 *   2  usage, I/O, or CNF parse error        (prints "ERROR: ...")
 *
 * The "p cnf <vars> <clauses>" header is treated tolerantly: the actual
 * clauses in the file are what get checked; a count mismatch only produces
 * an informational comment line.
 *
 * Plain C99, no external dependencies.  Both inputs are streamed through a
 * fixed-size buffer, so multi-hundred-MB logs and ~GB CNF files are fine;
 * the only memory that scales with input is one byte per model variable.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RD_BUF_SIZE (1u << 20)     /* 1 MiB streaming buffer */
#define VAR_LIMIT   (1LL << 28)    /* max variable index accepted (~268M) */
#define SHOW_LITS   16             /* literals of a failing clause to print */

/* ------------------------------------------------------------------ */
/* Buffered byte-stream reader                                        */
/* ------------------------------------------------------------------ */

typedef struct {
    FILE *f;
    unsigned char *buf;
    size_t len;
    size_t pos;
    long long line;                /* current 1-based line number */
} Reader;

static int rd_open(Reader *r, const char *path) {
    r->f = fopen(path, "rb");
    if (!r->f)
        return 0;
    r->buf = (unsigned char *)malloc(RD_BUF_SIZE);
    if (!r->buf) {
        fclose(r->f);
        r->f = NULL;
        return 0;
    }
    r->len = 0;
    r->pos = 0;
    r->line = 1;
    return 1;
}

static void rd_close(Reader *r) {
    if (r->f)
        fclose(r->f);
    free(r->buf);
    r->f = NULL;
    r->buf = NULL;
}

static int rd_getc(Reader *r) {
    if (r->pos >= r->len) {
        r->len = fread(r->buf, 1, RD_BUF_SIZE, r->f);
        r->pos = 0;
        if (r->len == 0)
            return -1;
    }
    int c = r->buf[r->pos++];
    if (c == '\n')
        r->line++;
    return c;
}

/* Skip the rest of the current line (fast path for comment/stat lines). */
static void rd_skip_line(Reader *r) {
    for (;;) {
        int c = rd_getc(r);
        if (c == -1 || c == '\n')
            return;
    }
}

static int is_blank(int c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\f' || c == '\v';
}

/* ------------------------------------------------------------------ */
/* Model (assignment) storage                                         */
/* ------------------------------------------------------------------ */

typedef struct {
    signed char *val;              /* val[v] in {-1, 0, +1}; index 1..cap */
    long long cap;                 /* highest allocatable variable index */
    long long max_var;             /* highest variable actually assigned */
    long long assigned;            /* number of assigned variables */
    int saw_sat_status;
    int saw_unsat_status;
    int saw_vline;
    int terminated;                /* saw the trailing 0 of the witness */
} Model;

static int model_reserve(Model *m, long long var) {
    if (var <= m->cap)
        return 1;
    if (var > VAR_LIMIT)
        return 0;
    long long ncap = m->cap ? m->cap : (1LL << 16);
    while (ncap < var) {
        ncap *= 2;
        if (ncap > VAR_LIMIT) {
            ncap = VAR_LIMIT;
            break;
        }
    }
    signed char *nv = (signed char *)realloc(m->val, (size_t)(ncap + 1));
    if (!nv)
        return 0;
    long long old = m->cap ? m->cap + 1 : 0;
    memset(nv + old, 0, (size_t)(ncap + 1 - old));
    m->val = nv;
    m->cap = ncap;
    return 1;
}

static int model_value(const Model *m, long long var) {
    if (var < 1 || var > m->cap)
        return 0;
    return m->val[var];
}

/* ------------------------------------------------------------------ */
/* Result reporting                                                   */
/* ------------------------------------------------------------------ */

static int fail_model(const char *fmt, ...);
static int fail_error(const char *fmt, ...);

#include <stdarg.h>

static int fail_model(const char *fmt, ...) {
    va_list ap;
    fputs("MODEL FAILED: ", stdout);
    va_start(ap, fmt);
    vfprintf(stdout, fmt, ap);
    va_end(ap);
    fputc('\n', stdout);
    return 1;
}

static int fail_error(const char *fmt, ...) {
    va_list ap;
    fputs("ERROR: ", stderr);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    return 2;
}

/* ------------------------------------------------------------------ */
/* Solver log parsing                                                 */
/* ------------------------------------------------------------------ */

/*
 * Returns 0 on success (model fields populated) or an exit code (1/2)
 * after printing a diagnostic.
 */
static int parse_log(const char *path, Model *m) {
    Reader r;
    if (!rd_open(&r, path))
        return fail_error("cannot open solver log '%s'", path);

    int rc = 0;
    for (;;) {
        int c = rd_getc(&r);
        if (c == -1)
            break;
        if (c == '\n')
            continue;                       /* empty line */

        if (c == 's') {
            /* Possible status line: "s SATISFIABLE" / "s UNSATISFIABLE". */
            c = rd_getc(&r);
            if (c == -1)
                break;
            if (!is_blank(c)) {
                if (c != '\n')
                    rd_skip_line(&r);
                continue;
            }
            while (is_blank(c))
                c = rd_getc(&r);
            char token[32];
            size_t n = 0;
            while (c != -1 && c != '\n' && !is_blank(c)) {
                if (n + 1 < sizeof(token))
                    token[n++] = (char)c;
                c = rd_getc(&r);
            }
            token[n] = '\0';
            if (strcmp(token, "SATISFIABLE") == 0)
                m->saw_sat_status = 1;
            else if (strcmp(token, "UNSATISFIABLE") == 0)
                m->saw_unsat_status = 1;
            if (c != -1 && c != '\n')
                rd_skip_line(&r);
            continue;
        }

        if (c == 'v') {
            long long vline = r.line;
            c = rd_getc(&r);
            if (c != -1 && c != '\n' && !is_blank(c)) {
                /* Some other word starting with 'v'; not a witness line. */
                rd_skip_line(&r);
                continue;
            }
            m->saw_vline = 1;
            /* Parse whitespace-separated signed literals to end of line. */
            while (c != -1 && c != '\n') {
                if (is_blank(c)) {
                    c = rd_getc(&r);
                    continue;
                }
                int neg = 0;
                if (c == '-') {
                    neg = 1;
                    c = rd_getc(&r);
                }
                if (c < '0' || c > '9') {
                    rc = fail_model(
                        "malformed witness ('v') line at log line %lld",
                        vline);
                    goto done;
                }
                long long v = 0;
                while (c >= '0' && c <= '9') {
                    v = v * 10 + (c - '0');
                    if (v > VAR_LIMIT) {
                        rc = fail_model(
                            "literal magnitude %lld exceeds limit at log "
                            "line %lld", v, vline);
                        goto done;
                    }
                    c = rd_getc(&r);
                }
                if (c != -1 && c != '\n' && !is_blank(c)) {
                    rc = fail_model(
                        "malformed witness ('v') line at log line %lld",
                        vline);
                    goto done;
                }
                if (v == 0) {
                    m->terminated = 1;
                } else if (!m->terminated) {
                    if (!model_reserve(m, v)) {
                        rc = fail_error(
                            "out of memory storing model (variable %lld)",
                            v);
                        goto done;
                    }
                    signed char want = neg ? -1 : 1;
                    if (m->val[v] != 0 && m->val[v] != want) {
                        rc = fail_model(
                            "contradictory assignment for variable %lld "
                            "in witness", v);
                        goto done;
                    }
                    if (m->val[v] == 0)
                        m->assigned++;
                    m->val[v] = want;
                    if (v > m->max_var)
                        m->max_var = v;
                }
                /* literal after the terminating 0 on the same line: ignore */
            }
            continue;
        }

        /* Anything else ('c' comments, stats, huge padding lines): skip. */
        rd_skip_line(&r);
    }

    if (ferror(r.f)) {
        rc = fail_error("read error on solver log '%s'", path);
        goto done;
    }

    if (!m->saw_sat_status) {
        if (m->saw_unsat_status)
            rc = fail_model(
                "solver log reports 's UNSATISFIABLE'; no model to check");
        else
            rc = fail_model("no 's SATISFIABLE' status line in solver log");
        goto done;
    }
    if (!m->saw_vline) {
        rc = fail_model(
            "'s SATISFIABLE' found but no 'v' witness lines in solver log");
        goto done;
    }
    if (!m->terminated) {
        rc = fail_model(
            "incomplete witness: 'v' lines not terminated by 0");
        goto done;
    }
    if (m->assigned == 0) {
        rc = fail_model("empty witness: no literals before terminating 0");
        goto done;
    }

done:
    rd_close(&r);
    return rc;
}

/* ------------------------------------------------------------------ */
/* CNF streaming + clause checking                                    */
/* ------------------------------------------------------------------ */

static int check_cnf(const char *path, const Model *m) {
    Reader r;
    if (!rd_open(&r, path))
        return fail_error("cannot open CNF file '%s'", path);

    long long declared_vars = -1, declared_clauses = -1;
    long long clause_idx = 0;      /* clauses completed so far */
    long long clause_line = 1;     /* line where current clause started */
    int cur_sat = 0;
    long long cur_len = 0;
    long long first_unassigned = 0;
    long long shown[SHOW_LITS];
    int shown_n = 0;
    int rc = 0;
    int at_line_start = 1;
    int done_parsing = 0;

    for (;;) {
        int c = rd_getc(&r);
        if (c == -1)
            break;
        if (c == '\n') {
            at_line_start = 1;
            continue;
        }
        if (is_blank(c))
            continue;               /* keeps at_line_start over indentation */

        if (at_line_start && c == 'c') {
            rd_skip_line(&r);
            at_line_start = 1;
            continue;
        }
        if (at_line_start && c == 'p') {
            char header[256];
            size_t n = 0;
            c = rd_getc(&r);
            while (c != -1 && c != '\n') {
                if (n + 1 < sizeof(header))
                    header[n++] = (char)c;
                c = rd_getc(&r);
            }
            header[n] = '\0';
            /* Tolerant: a malformed header is simply ignored. */
            if (sscanf(header, " cnf %lld %lld",
                       &declared_vars, &declared_clauses) != 2) {
                declared_vars = -1;
                declared_clauses = -1;
            }
            at_line_start = 1;
            continue;
        }
        if (at_line_start && c == '%') {
            /* SATLIB-style end-of-formula marker: stop parsing. */
            done_parsing = 1;
            break;
        }
        at_line_start = 0;

        /* Expect a signed integer literal. */
        int neg = 0;
        if (c == '-') {
            neg = 1;
            c = rd_getc(&r);
        }
        if (c < '0' || c > '9') {
            rc = fail_error(
                "malformed CNF '%s': unexpected character '%c' (0x%02x) "
                "at line %lld", path,
                (c >= 32 && c < 127) ? c : '?', (unsigned)(c & 0xff),
                r.line);
            goto done;
        }
        long long v = 0;
        while (c >= '0' && c <= '9') {
            v = v * 10 + (c - '0');
            if (v > VAR_LIMIT) {
                rc = fail_error(
                    "malformed CNF '%s': literal magnitude %lld exceeds "
                    "limit at line %lld", path, v, r.line);
                goto done;
            }
            c = rd_getc(&r);
        }
        if (c == '\n')
            at_line_start = 1;
        else if (c != -1 && !is_blank(c)) {
            rc = fail_error(
                "malformed CNF '%s': unexpected character after literal "
                "at line %lld", path, r.line);
            goto done;
        }

        if (v == 0) {
            /* End of clause. */
            clause_idx++;
            if (!cur_sat) {
                fputs("MODEL FAILED: clause ", stdout);
                printf("%lld", clause_idx);
                fputs(" (starting at CNF line ", stdout);
                printf("%lld", clause_line);
                fputs(") not satisfied by model:", stdout);
                for (int i = 0; i < shown_n; i++)
                    printf(" %lld", shown[i]);
                if (cur_len > shown_n)
                    printf(" ... (%lld literals)", cur_len);
                fputs(" 0", stdout);
                if (first_unassigned)
                    printf(" [variable %lld unassigned in model]",
                           first_unassigned);
                fputc('\n', stdout);
                rc = 1;
                goto done;
            }
            cur_sat = 0;
            cur_len = 0;
            first_unassigned = 0;
            shown_n = 0;
            clause_line = r.line;
        } else {
            long long lit = neg ? -v : v;
            if (cur_len == 0)
                clause_line = r.line;
            cur_len++;
            if (shown_n < SHOW_LITS)
                shown[shown_n++] = lit;
            int assigned = model_value(m, v);
            if (assigned == 0) {
                if (!first_unassigned)
                    first_unassigned = v;
            } else if ((assigned > 0) == !neg) {
                cur_sat = 1;
            }
        }
    }

    if (!done_parsing && ferror(r.f)) {
        rc = fail_error("read error on CNF file '%s'", path);
        goto done;
    }

    /* Tolerate a final clause missing its terminating 0 at EOF. */
    if (cur_len > 0) {
        clause_idx++;
        if (!cur_sat) {
            printf("MODEL FAILED: clause %lld (unterminated at EOF) "
                   "not satisfied by model\n", clause_idx);
            rc = 1;
            goto done;
        }
    }

    if (declared_clauses >= 0 && declared_clauses != clause_idx)
        printf("c checkmodel: note: header declares %lld clauses, "
               "file contains %lld (checked actual clauses)\n",
               declared_clauses, clause_idx);
    if (declared_vars >= 0 && m->max_var > declared_vars)
        printf("c checkmodel: note: model assigns up to variable %lld, "
               "header declares %lld variables\n",
               m->max_var, declared_vars);

    printf("c checkmodel: %lld variable(s) assigned, %lld clause(s) "
           "checked\n", m->assigned, clause_idx);

done:
    rd_close(&r);
    return rc;
}

/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr,
                "usage: checkmodel <formula.cnf> <solver.log>\n"
                "  Verifies the model printed in <solver.log> "
                "('s SATISFIABLE' + 'v' lines)\n"
                "  against every clause of the DIMACS <formula.cnf>.\n"
                "  Exit codes: 0 verified, 1 model failed, 2 usage/IO "
                "error.\n");
        return 2;
    }

    Model m;
    memset(&m, 0, sizeof(m));

    /* Parse the (potentially huge) solver log first: it defines the model. */
    int rc = parse_log(argv[2], &m);
    if (rc != 0) {
        free(m.val);
        return rc;
    }

    rc = check_cnf(argv[1], &m);
    free(m.val);
    if (rc != 0)
        return rc;

    printf("MODEL VERIFIED\n");
    return 0;
}
