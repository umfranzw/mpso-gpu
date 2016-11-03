#include "config_mcs.h"

cl_int get_next_config_mcs(
    FILE *config_file,
    config_mcs *config
    )
{
    cl_int ret_val = 0;
    if (!feof(config_file))
    {
        ret_val = parse_config_mcs(
            config_file, 
            config
            );
        
        if (!ret_val)
        {
            printf("Error parsing config file.\n");
        }
    }

    return ret_val;
}

cl_int parse_config_mcs(
    FILE *file, 
    config_mcs *conf
    )
{
    return parse_param(file, "%u,", (void *) &(conf->num_swarms)) &&
        parse_param(file, "%u,", (void *) &(conf->num_sparticles)) &&
        parse_param(file, "%u,", (void *) &(conf->num_dims)) &&
        parse_param(file, "%u,", (void *) &(conf->max_iters)) &&
        parse_param(file, "%f,", (void *) &(conf->omega)) &&
        parse_param(file, "%f,", (void *) &(conf->omega_decay)) &&
        parse_param(file, "%f,", (void *) &(conf->c1)) &&
        parse_param(file, "%f,", (void *) &(conf->c2)) &&
        parse_param(file, "%u,", (void *) &(conf->exchange_iters)) &&
        parse_param(file, "%u,", (void *) &(conf->num_exchange)) &&
        parse_param(file, "%u,", (void *) &(conf->num_reps)) &&
        parse_param(file, "%u,", (void *) &(conf->bench_fcn)) &&
        parse_param(file, "%u,", (void *) &(conf->m)) &&
        parse_param(file, "%u,", (void *) &(conf->cross_iters)) &&
        parse_param(file, "%f,", (void *) &(conf->unhealthy_ratio)) &&
        parse_param(file, "%u,", (void *) &(conf->unhealthy_iters)) &&
        parse_param(file, "%f,", (void *) &(conf->mut_prob)) &&
        parse_param(file, "%f,", (void *) &(conf->max_vel)) &&
        parse_param(file, "%u", (void *) &(conf->fitness_sample_interval));
}
