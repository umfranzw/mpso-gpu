#include "config_reg.h"

cl_int get_next_config_reg(
    FILE *config_file,
    config_reg *config
    )
{
    cl_int ret_val = 0;
    if (!feof(config_file))
    {
        ret_val = parse_config_reg(
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

cl_int parse_config_reg(
    FILE *file, 
    config_reg *conf
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
        parse_param(file, "%f,", (void *) &(conf->max_vel)) &&
        parse_param(file, "%u", (void *) &(conf->fitness_sample_interval));
}
