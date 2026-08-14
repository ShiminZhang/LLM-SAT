/**
 * (taomengxia) [dynamic sat] 2025-04-27
 */

#ifndef _a_dynamicsat_h_INCLUDED
#define _a_dynamicsat_h_INCLUDED

struct kissat;

// DynamicSAT hyperparameters
#define DSAT_SAMPLE_CNT 1000
#define DSAT_CHANGE_THRESHOLD (0.3 * CLAUSES)
#define DSAT_INT_CONFIG_STEP 0.1
#define DSAT_DECISION_CNT 100

// DynamicSAT tuneable configurations
#define DSAT_NO_ACTIONS 6           // Total number of actions

struct dsat_config {
    const char *name;
    const int type;                 // DSAT_CONFIG_TYPE_BOOL/DSAT_CONFIG_TYPE_INT
    const int value_default;
    const int value_min;
    const int value_max;
};

#define DSAT_CONFIG_TYPE_BOOL 0
#define DSAT_CONFIG_TYPE_INT 1

typedef struct dsat_config dsat_config;

static const dsat_config dsat_config_table[] = {
  //  {"chrono",          DSAT_CONFIG_TYPE_BOOL,    0,   0,   1},
  //  {"chronolevels",    DSAT_CONFIG_TYPE_INT,     10, 10, 100},
  {"compact", DSAT_CONFIG_TYPE_BOOL, 1,0,1},
  {"compactlim",  DSAT_CONFIG_TYPE_INT,     10, 0, 100},
};

#define DSAT_CONFIG_SIZE ((sizeof(dsat_config_table) / sizeof (dsat_config)))

#define DSAT_CONFIG_NAME(idx)           (dsat_config_table[idx].name)
#define DSAT_CONFIG_TYPE(idx)           (dsat_config_table[idx].type)
#define DSAT_CONFIG_VALUE_DEFAULT(idx)  (dsat_config_table[idx].value_default)
#define DSAT_CONFIG_VALUE_MIN(idx)      (dsat_config_table[idx].value_min)
#define DSAT_CONFIG_VALUE_MAX(idx)      (dsat_config_table[idx].value_max)


void kissat_dynamicsat_init (struct kissat *solver);
void kissat_dynamicsat_update_learn_info(struct kissat *solver, const size_t glue);
void kissat_dynamicsat (struct kissat *solver);

#endif  // _a_dynamicsat_h_INCLUDED
