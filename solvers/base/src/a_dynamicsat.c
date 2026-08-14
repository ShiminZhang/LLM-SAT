/**
 * (taomengxia) [dynamic sat] 2025-04-27
 */

#include "internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static void kissat_dynamicsat_get_actions(const int _action_val, int *action_list) {
    int action_val = _action_val;
    for (size_t i = 0; i < DSAT_CONFIG_SIZE; i++) {
        const int config_type = DSAT_CONFIG_TYPE(i);
        if (config_type == DSAT_CONFIG_TYPE_BOOL) {
            action_list[i] = action_val % 2;
            action_val /= 2;
        } else {
            assert (config_type == DSAT_CONFIG_TYPE_INT);
            action_list[i] = action_val % 3;
            action_val /= 3;
        }
    }
}

static int kissat_dynamicsat_weighted_random(kissat *solver) {
    const double *weights = solver->ucb_D;

    double r = (double)rand() / RAND_MAX;
    double weight_sum = 0; 
    double norm_weights[DSAT_NO_ACTIONS];
    for (int i = 0; i < DSAT_NO_ACTIONS; i++) weight_sum += weights[i];
    for (int i = 0; i < DSAT_NO_ACTIONS; i++) norm_weights[i] = weights[i] / weight_sum;

    double cumulative_prob = 0.0;
    for (int i = 0; i < DSAT_NO_ACTIONS - 1; i++) {
        cumulative_prob += norm_weights[i]; 
        if (r < cumulative_prob) {
            return i; 
        }
    }
    return DSAT_NO_ACTIONS - 1; 
}

void kissat_dynamicsat_init (kissat *solver) {
    memset(solver->mab_reward_D, 0, sizeof(double) * DSAT_NO_ACTIONS);
    memset(solver->mab_selected_D, 0, sizeof(double) * DSAT_NO_ACTIONS);
    memset(solver->ucb_D, 0, sizeof(double) * DSAT_NO_ACTIONS);

    for (size_t i = 0; i < DSAT_CONFIG_SIZE; i++) {
        solver->cur_config_values[i] = DSAT_CONFIG_VALUE_DEFAULT(i);
    }
    solver->last_action = -1;

    // Stone add
    solver->last_clauses_added = 0;
    solver->last_clauses_deleted = 0;
    //Dynamic SAT variables
    solver->num_decisions_D = 0;
    //solver->tot_mab_chosen = 0;
    solver->num_of_sampling_D = 0;
    solver->learned = 0;
    solver->tot_glue = 0;
    solver->mab_in_process = 1e7+1;
    solver->mab_reset_threshold = CLAUSES * 10;
}

void kissat_dynamicsat_update_learn_info (kissat *solver, const size_t glue) {
    solver->tot_glue += glue;
    solver->learned++;
}

static void kissat_dynamicsat_action_execute (kissat *solver, int cur_action) {
    int action_list[DSAT_CONFIG_SIZE];
    kissat_dynamicsat_get_actions(cur_action, action_list);
    for (size_t i = 0; i < DSAT_CONFIG_SIZE; i++) {
        const int config_type = DSAT_CONFIG_TYPE(i);
        int new_config_value = 0;
        if (config_type == DSAT_CONFIG_TYPE_BOOL) {
            new_config_value = action_list[i];
        } else {
            assert (config_type == DSAT_CONFIG_TYPE_INT);

            new_config_value = solver->cur_config_values[i];
            new_config_value += (action_list[i] - 1) * DSAT_INT_CONFIG_STEP * new_config_value;
            const int config_value_min = DSAT_CONFIG_VALUE_MIN(i);
            const int config_value_max = DSAT_CONFIG_VALUE_MAX(i);
            if (new_config_value < config_value_min) {
                new_config_value = config_value_min;
            } else if (new_config_value > config_value_max) {
                new_config_value = config_value_max;
            }
        }
        const char *config_name = DSAT_CONFIG_NAME(i);
        kissat_set_option(solver, config_name, new_config_value);
        solver->cur_config_values[i] = new_config_value;
    }
}

static void kissat_dynamicsat_update_mab (kissat *solver, int cur_action, const int avg_glue) {
    // Update MAB
    if (solver->last_action >= 0) {
        solver->mab_selected_D[solver->last_action]++;
        solver->mab_reward_D[solver->last_action] += (avg_glue > 5) ? 0 : (5 - avg_glue);
        solver->learned = 0;
        solver->tot_glue = 0;
    }
    solver->last_action = cur_action;
}

void kissat_dynamicsat (kissat *solver) {
    static double avg_glue = 0;
    // Get the score of changed, to determine whether call MAB
    int delta_clauses_added = solver->statistics.clauses_redundant - solver->last_clauses_added;
    int delta_clauses_deleted = solver->statistics.reductions - solver->last_clauses_deleted;
    int change_score = delta_clauses_added + delta_clauses_deleted * 100;

    // Step 1. Sampling
    if (solver->num_decisions_D < DSAT_SAMPLE_CNT && solver->num_decisions_D % 100 == 0) {
        // Get reward
        avg_glue = solver->learned ? (double)solver->tot_glue / solver->learned : 0;
        // Action execution
        int cur_action = rand() % DSAT_NO_ACTIONS;
        kissat_dynamicsat_action_execute (solver, cur_action);
        // Update MAB
        kissat_dynamicsat_update_mab (solver, cur_action, avg_glue);
    }
    // Step 2. Decision
    else if (solver->num_decisions_D >= DSAT_SAMPLE_CNT &&
             (change_score > DSAT_CHANGE_THRESHOLD || solver->mab_in_process < DSAT_SAMPLE_CNT) ) {
        // Change of clauses exceed threshold
        if (change_score > DSAT_CHANGE_THRESHOLD) {
            solver->last_clauses_added = solver->statistics.clauses_redundant;
            solver->last_clauses_deleted = solver->statistics.reductions;
            solver->mab_in_process = 0;
        }
        // mab in process for sampling % 100 times
        else {
            solver->mab_in_process++;
            solver->mab_in_process %= (int)1e7+1;
        }
        // Get reward
        if (solver->mab_in_process % DSAT_DECISION_CNT == 0) {
            avg_glue = solver->learned ? (double)solver->tot_glue / solver->learned : 0;
        }
        // Action execution
        solver->num_of_sampling_D++;
        int cur_action = kissat_dynamicsat_weighted_random(solver);
        kissat_dynamicsat_action_execute (solver, cur_action);
        // Update MAB
        kissat_dynamicsat_update_mab (solver, cur_action, avg_glue);
        // Update UCB
        for (int i = 0; i < DSAT_NO_ACTIONS; i++) {
            solver->ucb_D[i] = solver->mab_reward_D[i] / solver->mab_selected_D[i] +
                               sqrt(log(solver->num_of_sampling_D) / solver->mab_selected_D[i]);
        }
    }
    // MAB Reset
    solver->num_decisions_D++;
    unsigned tot_change_score = solver->statistics.clauses_redundant - solver->last_clauses_added +
                                (solver->statistics.reductions - solver->last_clauses_deleted) * 100;
    if (tot_change_score > solver->mab_reset_threshold) {
        solver->mab_reset_threshold += CLAUSES * 10;
        if (solver->num_decisions_D > DSAT_SAMPLE_CNT) {
            solver->num_decisions_D = 0;
        }
    }
}